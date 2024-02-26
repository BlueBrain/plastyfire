"""
Single cell simulations in bluecellulab
last modified: András Ecker, 02.2024
"""

import re
import logging
import multiprocessing
import numpy as np
import pandas as pd
from functools import lru_cache
from bluepysnap import Simulation
import bluecellulab
from conntility.io.synapse_report import get_presyn_mapping

SPIKE_THRESHOLD = -30  # mV
with_cache = lru_cache(128)  # set cache for spiking thresholds
logger = logging.getLogger(__name__)  # configure logger
DEBUG = False
# because of the constant ping-pong between GluSynapse.mod, SONATA parameter names
# and synapse helpers both in `neurodamus` and in bluecellulab.synapses.synapse_types/GluSynapse() mimicking it
# some variable names have to be patched (to match the current state of GluSynapse.mod)
PARAM_MAP = {"Use_d_TM": "Use_d", "Use_p_TM": "Use_p", "Use0_TM": "Use",
             "Dep_TM": "Dep", "Fac_TM": "Fac", "Nrrp_TM": "Nrrp"}


def _get_spikes(t, v, dt_int=0.025):
    """Interprets fix `dt` time (and voltage) from CVode results, and finds spike times (skips 200 ms)"""
    tdense = np.linspace(min(t), max(t), int((max(t)-min(t))/dt_int))
    vdense = np.interp(tdense, t, v)
    return np.array([tdense[i+1] for i in range(int(200 / dt_int), len(vdense) - 1)
                     if vdense[i] < SPIKE_THRESHOLD and vdense[i+1] >= SPIKE_THRESHOLD])


def _runsinglecell_proc(sim_config, post_gid, stimulus, results, node_pop, fixhp):
    """Multiprocessing subprocess for `runsinglecell()`"""
    ssim = bluecellulab.SSim(sim_config)
    bluecellulab.neuron.h.cvode.atolscale("v", .1)
    ssim.instantiate_gids([(node_pop, post_gid)])
    cell = ssim.cells[(node_pop, post_gid)]
    if fixhp:  # hyperpolarization workaround
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # add stimuli and run sim
    tstim = bluecellulab.neuron.h.TStim(0.5, sec=cell.soma)
    stim_duration = (stimulus["nspikes"] - 1) * 1000. / stimulus["freq"] + stimulus["width"]
    tstim.train(stimulus["offset"], stim_duration, stimulus["amp"], stimulus["freq"], stimulus["width"])
    cell.persistent.append(tstim)
    ssim.run(stimulus["offset"] + stim_duration + 200., cvode=True)
    # get soma voltage and simulation time vector and extract spike times
    t = np.array(ssim.get_time())
    v = np.array(ssim.get_voltage_trace((node_pop, post_gid)))
    spikes = _get_spikes(t, v)
    # store results
    results["t"] = t
    results["v"] = v
    results["t_spikes"] = spikes
    results["t_stimuli"] = np.array(tstim.tvec)[:-1:4]
    results.update(stimulus)


def runsinglecell(sim_config, post_gid, stimulus, node_pop, fixhp):
    """Runs single cell simulation with given stimulus"""
    manager = multiprocessing.Manager()
    logger.debug("Submitting simulation: post_gid={}, f={}, a={}".format(post_gid, stimulus["freq"], stimulus["amp"]))
    results = manager.dict()
    p = multiprocessing.Process(target=_runsinglecell_proc, args=(sim_config, post_gid, stimulus, results, node_pop, fixhp))
    p.start()
    p.join()
    return dict(results)


@with_cache
def spike_threshold_finder(sim_config, post_gid, nspikes, freq, width, offset, min_amp, max_amp, nlevels,
                           node_pop="S1nonbarrel_neurons", fixhp=False):
    """
    Finds the min amplitude of stimulus current (within the range [`min_amp`, `max_amp`])
    that makes the `post_gid` fire `nspikes` APs (given `freq`, `width`, `offset`)
    using a binary search (parametrized by `nlevels`).
    """
    candidate_amp = np.linspace(min_amp, max_amp, nlevels)
    # find suitable amplitude (binary search, leftmost element)
    L = 0
    R = nlevels
    simres = {}
    while L < R:
        m = int(np.floor((L+R)/2.))
        stim = {"nspikes": nspikes,
                "freq": freq,
                "width": width,
                "offset": offset,
                "amp": candidate_amp[m]}
        simres[m] = runsinglecell(sim_config, post_gid, stim, node_pop, fixhp)
        logger.debug("Number of spikes = %d" % len(simres[m]["t_spikes"]))
        if len(simres[m]["t_spikes"]) < nspikes:
            L = m + 1
        else:
            R = m
    # L is the index of the best amplitude, but we don't know if the match was exact
    t_stim = 1000. / freq * np.array(range(nspikes)) + offset
    t_lim = t_stim + width + 5
    if L == nlevels:
        logger.debug("Max stimulation intensity too weak")
    elif len(simres[L]["t_spikes"]) != nspikes:
        logger.debug("Search grid too coarse")
    elif np.any(simres[L]["t_spikes"] < t_stim) or np.any(simres[L]["t_spikes"] > t_lim):
        logger.debug("Spikes out of order")
    else:
        logger.debug("Correct spike count for stimulation amplitude = %.3f nA" % simres[L]["amp"])
        return simres[L]
    return None


def _set_global_params(allparams):
    """Sets global parameters of the simulation"""
    logger.debug("Setting global parameters")
    for param_name, param_val in allparams.items():
        if re.match(".*_GluSynapse$", param_name):
            setattr(bluecellulab.neuron.h, param_name, param_val)
            logger.debug("\t%s = %f", param_name, getattr(bluecellulab.neuron.h, param_name))


def _set_local_params(synapse, fit_params, extra_params, c_pre=0., c_post=0.):
    """Sets synaptic parameters in bluecellulab"""
    for key, val in extra_params.items():  # update basic synapse parameters
        if key in PARAM_MAP:
            setattr(synapse.hsynapse, PARAM_MAP[key], val)
        else:
            if key  == "loc":
                continue
            setattr(synapse.hsynapse, key, val)
    if fit_params is not None:  # update thresholds
        if all(key in fit_params for key in ["a00", "a01"]) and extra_params["loc"] == "basal":
            # set basal depression threshold
            synapse.hsynapse.theta_d_GB = fit_params["a00"] * c_pre + fit_params["a01"] * c_post
        if all(key in fit_params for key in ["a10", "a11"]) and extra_params["loc"] == "basal":
            # set basal potentiation threshold
            synapse.hsynapse.theta_p_GB = fit_params["a10"] * c_pre + fit_params["a11"] * c_post
        if all(key in fit_params for key in ["a20", "a21"]) and extra_params["loc"] == "apical":
            # set apical depression threshold
            synapse.hsynapse.theta_d_GB = fit_params["a20"] * c_pre + fit_params["a21"] * c_post
        if all(key in fit_params for key in ["a30", "a31"]) and extra_params["loc"] == "apical":
            # set apical potentiation threshold
            synapse.hsynapse.theta_p_GB = fit_params["a30"] * c_pre + fit_params["a31"] * c_post


def _map_syn_idx(sim_config, post_gid, syn_idx, edge_pop):
    """Compared to `bluepysnap` which has one global synapse ID for all synapses in the circuit,
    `bluecellulab` (just as `neurodamus`) re-indexes synapses for each postsynaptic cell starting at 0
    (which ID is used for seeding the synapses). This helper gets the mapping between the two indexing versions
    It's unfortunate that this is here (and gets called so many times)... but I couldn't find a better way"""
    sim = Simulation(sim_config)
    return get_presyn_mapping(sim.circuit, edge_pop,
                              pd.MultiIndex.from_tuples([(post_gid, syn_id) for syn_id in syn_idx]))


def _c_pre_finder_process(sim_config, fit_params, syn_extra_params, pre_gid, post_gid,
                          node_pop, edge_pop, fixhp):
    """
    Multiprocessing subprocess for `c_pre_finder()`
    Delivers spike from `pre_gid` and measures the Ca++ transient at the synapses on `post_gid`
    """
    logger.debug("c_pre finder process")
    ssim = bluecellulab.SSim(sim_config)
    bluecellulab.neuron.h.cvode.atolscale("v", .1)
    ssim.instantiate_gids([(node_pop, post_gid)], add_synapses=True, add_minis=False,
                          pre_spike_trains={(node_pop, pre_gid): [1000.]},
                          intersect_pre_gids=[(node_pop, pre_gid)])
    cell = ssim.cells[(node_pop, post_gid)]
    if fixhp:  # hyperpolarization workaround
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    if fit_params is not None:  # setup global parameters
        _set_global_params(fit_params)
    # initialize [Ca^{2+}]_i recorders
    recorder, syn_idx = {}, []
    for syn_id, synapse in cell.synapses.items():
        syn_idx.append(syn_id[1])
        recorder[syn_id[1]] = bluecellulab.neuron.h.Vector()
        recorder[syn_id[1]].record(synapse.hsynapse._ref_effcai_GB)
    # setup synapses and run simulation
    df = _map_syn_idx(sim_config, post_gid, syn_idx, edge_pop)
    for syn_id, synapse in cell.synapses.items():
        logger.debug("Configuring synapse %d", syn_id[1])
        if syn_extra_params is not None:  # configure local parameters
            _set_local_params(synapse, fit_params,
                              syn_extra_params[df.loc[df["local_syn_idx"] == syn_id[1]].index[0]])
        synapse.hsynapse.rho0_GB = 1  # override rho (not sure if it's needed)
        synapse.hsynapse.Use_p = 1  # override potentiated U_SE (to guarantee release)
        synapse.hsynapse.Use = synapse.hsynapse.Use_p  # override U_SE
        synapse.hsynapse.gmax0_AMPA = synapse.hsynapse.gmax_p_AMPA  # override conductance
        synapse.hsynapse.theta_d_GB = -1  # disable LTD
        synapse.hsynapse.theta_p_GB = -1  # disable LTP
    ssim.run(2500, cvode=True)
    logger.debug("Simulation completed")
    # compute c_pre and store results
    c_pre = {df.loc[df["local_syn_idx"] == syn_id].index[0]: recorder[syn_id].max() for syn_id in syn_idx}
    results = {"c_pre": c_pre,
               "c_trace": {df.loc[df["local_syn_idx"] == syn_id].index[0]:
                           np.array(recorder[syn_id]) for syn_id in syn_idx}}
    return results


def c_pre_finder(sim_config, fit_params, syn_extra_params, pre_gid, post_gid,
                 node_pop="S1nonbarrel_neurons", edge_pop="S1nonbarrel_neurons__S1nonbarrel_neurons__chemical",
                 fixhp=False):
    """Replays spike from `pre_gid` and measures Ca++ transient in synapses on `post_gid`"""
    pool = multiprocessing.Pool(processes=1)
    results = pool.apply(_c_pre_finder_process, [sim_config, fit_params, syn_extra_params, pre_gid, post_gid,
                                               node_pop, edge_pop, fixhp])
    pool.terminate()
    logger.debug("C_pre: %s", str(results["c_pre"]))
    return results["c_pre"]


def _c_post_finder_process(sim_config, stimulus, fit_params, syn_extra_params, pre_gid, post_gid,
                           node_pop, edge_pop, fixhp):
    """
    Multiprocessing subprocess for `c_post_finder()`
    Injects (precalculated) stimulus to the `post_gid` to make it fire 1 AP and measures the Ca++ transient
    (from the backpropagiting AP) at the synapses made by `pre_gid`
    """
    logger.debug("c_post finder process")
    ssim = bluecellulab.SSim(sim_config)
    bluecellulab.neuron.h.cvode.atolscale("v", .1)
    ssim.instantiate_gids([(node_pop, post_gid)], add_synapses=True, add_minis=False,
                          intersect_pre_gids=[(node_pop, pre_gid)])
    cell = ssim.cells[(node_pop, post_gid)]
    if fixhp:  # hyperpolarization workaround
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # add stimuli
    tstim = bluecellulab.neuron.h.TStim(0.5, sec=cell.soma)
    stim_duration = (stimulus["nspikes"] - 1) * 1000./stimulus["freq"] + stimulus["width"]
    tstim.train(stimulus["offset"], stim_duration, stimulus["amp"], stimulus["freq"], stimulus["width"])
    cell.persistent.append(tstim)
    if fit_params is not None:  # setup global parameters
        _set_global_params(fit_params)
    # initialize [Ca^{2+}]_i recorders
    recorder, syn_idx = {}, []
    for syn_id, synapse in cell.synapses.items():
        syn_idx.append(syn_id[1])
        recorder[syn_id[1]] = bluecellulab.neuron.h.Vector()
        recorder[syn_id[1]].record(synapse.hsynapse._ref_effcai_GB)
    # setup synapses and run simulation
    df = _map_syn_idx(sim_config, post_gid, syn_idx, edge_pop)
    for syn_id, synapse in cell.synapses.items():
        logger.debug("Configuring synapse %d", syn_id[1])
        if syn_extra_params is not None:  # configure local parameters
            _set_local_params(synapse, fit_params,
                              syn_extra_params[df.loc[df["local_syn_idx"] == syn_id[1]].index[0]])
        synapse.hsynapse.theta_d_GB = -1  # disable LTD
        synapse.hsynapse.theta_p_GB = -1  # disable LTP
    ssim.run(1500, cvode=True)
    logger.debug("Simulation completed")
    # get soma voltage and simulation time vector and extract spike times
    t = np.array(ssim.get_time())
    v = np.array(ssim.get_voltage_trace((node_pop, post_gid)))
    spikes = _get_spikes(t, v)
    # compute c_post and store results
    c_post = {df.loc[df["local_syn_idx"] == syn_id].index[0]: recorder[syn_id].max() for syn_id in syn_idx}
    results = {"c_post": c_post,
               "c_trace": {df.loc[df["local_syn_idx"] == syn_id].index[0]:
                           np.array(recorder[syn_id]) for syn_id in syn_idx},
               "t": t,
               "v": v,
               "t_spikes": spikes,
               "t_stimuli": np.array(tstim.tvec)[:-1:4]}
    return results


def c_post_finder(sim_config, fit_params, syn_extra_params, pre_gid, post_gid, stimulus,
                  node_pop="S1nonbarrel_neurons", edge_pop="S1nonbarrel_neurons__S1nonbarrel_neurons__chemical",
                  fixhp=False):
    """
    Finds c_post - the calcium transient from a postsynaptic spike in `post_gid` at synapses made by `pre_gid`.
    Injects current that makes the cell fire a single AP and then measures the Ca++ transient of the backpropagating AP.
    (To do so the necessary current that makes `post_gid` fire a single AP must be calculated beforehand
    - see `spike_threshold_finder()` above - and passed as a `stimulus` dictionary.)
    """
    # find c_post
    logger.debug("Stimulating cell with {} nA pulse ({} ms)".format(stimulus["amp"], stimulus["width"]))
    pool = multiprocessing.Pool(processes=1)
    results = pool.apply(_c_post_finder_process, [sim_config, stimulus, fit_params, syn_extra_params,
                                                  pre_gid, post_gid, node_pop, edge_pop, fixhp])
    pool.terminate()
    # validate number of spikes
    logger.debug("Spike timing: {}".format(results["t_spikes"]))
    logger.debug("C_post: %s", str(results["c_post"]))
    if len(results["t_spikes"]) < 1:
        # special case, small integration differences with threshold detection sim
        logger.debug("Cell not spiking as expected during c_post, "
                     "attempting to bump stimulus amplitude before failing...")
        # find c_post
        amp = stimulus["amp"] + 0.05
        logger.debug("Stimulating cell with %f nA pulse", amp)
        stimulus = {"nspikes": 1, "freq": 0.1, "width": stimulus["width"], "offset": 1000., "amp": amp}
        pool = multiprocessing.Pool(processes=1)
        results = pool.apply(_c_post_finder_process, [sim_config, stimulus, fit_params, syn_extra_params,
                                                      pre_gid, post_gid, node_pop, edge_pop, fixhp])
        pool.terminate()
        logger.debug("C_post: %s", str(results["c_post"]))
        return results["c_post"] if len(results["t_spikes"]) == 1 else None
    return results["c_post"]

