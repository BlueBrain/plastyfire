"""
Single cell simulations in BGLibPy
last modified: Giuseppe Chindemi, 03.2020
+ deleting parts that aren't used for generalization, adding input impedance calculation (for ML)
and minor changes for BGLibPy compatibility by András Ecker, 04.2021
"""

import re
import logging
import multiprocessing
import numpy as np
import bglibpy
from plastyfire.synapse import Synapse
from functools import lru_cache

bglibpy.neuron.h.cvode.atolscale("v", .1)
DEBUG = False
# set cache for spiking thresholds
with_cache = lru_cache(128)
# configure logger
logger = logging.getLogger(__name__)
# patch bglibpy
bglibpy.Synapse = Synapse


def _runsinglecell_proc(bc, post_gid, stimulus, results, fixhp):
    """Multiprocessing subprocess for `runsinglecell()`"""
    # create simulation
    ssim = bglibpy.SSim(bc)
    ssim.instantiate_gids([post_gid])
    cell = ssim.cells[post_gid]
    # hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # add stimuli
    tstim = bglibpy.neuron.h.TStim(0.5, sec=cell.soma)
    stim_duration = (stimulus["nspikes"] - 1) * 1000./stimulus["freq"] + stimulus["width"]
    tstim.train(stimulus["offset"], stim_duration, stimulus["amp"], stimulus["freq"], stimulus["width"])
    cell.persistent.append(tstim)
    # runsim
    ssim.run(stimulus["offset"] + stim_duration + 200., cvode=True)
    # get soma voltage and simulation time vector
    t = np.array(ssim.get_time())
    v = np.array(ssim.get_voltage_traces()[post_gid])
    # get spike timing (skip 200 ms)
    dt_int = 0.025
    tdense = np.linspace(min(t), max(t), int((max(t)-min(t))/dt_int))
    vdense = np.interp(tdense, t, v)
    spikes = np.array([tdense[i+1] for i in range(int(200/dt_int), len(vdense) - 1)
                       if vdense[i] < -30 and vdense[i+1] >= -30])
    # store results
    results["t"] = t
    results["v"] = v
    results["t_spikes"] = spikes
    results["t_stimuli"] = np.array(tstim.tvec)[:-1:4]
    results.update(stimulus)


def runsinglecell(bc, post_gid, stimulus, fixhp):
    """Runs single cell simulation with given stimulus"""
    manager = multiprocessing.Manager()
    logger.debug("Submitting simulation: post_gid={}, f={}, a={}".format(post_gid, stimulus["freq"], stimulus["amp"]))
    results = manager.dict()
    p = multiprocessing.Process(target=_runsinglecell_proc, args=(bc, post_gid, stimulus, results, fixhp))
    p.start()
    p.join()
    return dict(results)


@with_cache
def spike_threshold_finder(bc, post_gid, nspikes, freq, width, offset, min_amp, max_amp, nlevels, fixhp=True):
    """
    Finds the min amplitude of stimulus current (within the range [`min_amp`, `max_amp`])
    that makes the `post_gid` fire `nspikes` APs (given `freq`, `width`, `offset`)
    using a binary search (parametrized by `nlevels`).
    """
    # initialize search grid
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
        simres[m] = runsinglecell(bc, post_gid, stim, fixhp)
        logger.debug("Number of spikes = %d" % len(simres[m]["t_spikes"]))
        if len(simres[m]["t_spikes"]) < nspikes:
            L = m + 1
        else:
            R = m
    # L is the index of the best amplitude, but we don't know if the match was exact
    t_stim = 1000/freq * np.array(range(nspikes)) + offset
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


def _inp_imp_finder_process(bc, post_gid, synapse_locations, fixhp):
    """
    Multiprocessing subprocess for `inp_imp_finder()`
    Instantiates cell and uses NEURON's built in Impedance class to get input impedance at all synapse locations
    """
    logger.debug("input impedance finder process")
    ssim = bglibpy.SSim(bc)
    ssim.instantiate_gids([post_gid])
    cell = ssim.cells[post_gid]
    # hyperpolarization workaround (just for consistency)
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # calculate input impedance at evey synapse location
    inp_imps = {}
    for syn_id, row in synapse_locations.iterrows():
        pos = row["pos"]
        sec = cell.get_hsection(row["sec_id"])
        imp = bglibpy.neuron.h.Impedance()
        imp.loc(pos, sec=sec)
        imp.compute(0, 1)
        inp_imps[syn_id] = imp.input(pos, sec=sec)
    return inp_imps


def inp_imp_finder(bc, post_gid, synapse_locations, fixhp=True):
    """Calculates input impedance at all synapse locations"""
    pool = multiprocessing.Pool(processes=1)
    inp_imps = pool.apply(_inp_imp_finder_process, [bc, post_gid, synapse_locations, fixhp])
    pool.terminate()
    return inp_imps


def _set_global_params(allparams):
    """Sets global parameters of the simulation"""
    logger.debug("Setting global parameters")
    for param_name, param_val in allparams.items():
        if re.match(".*_GluSynapse$", param_name):
            setattr(bglibpy.neuron.h, param_name, param_val)
            logger.debug("\t%s = %f", param_name, getattr(bglibpy.neuron.h, param_name))


def _set_local_params(synapse, fit_params, extra_params, c_pre=0., c_post=0.):
    """Sets synaptic parameters in BGLibPy"""
    # update basic synapse parameters
    for param in extra_params:
        if param  == "loc":
            continue
        setattr(synapse.hsynapse, param, extra_params[param])
    # update thresholds
    if fit_params is not None:
        if all(key in fit_params for key in ["a00", "a01"]) and extra_params["loc"] == "basal":
            # set basal depression threshold
            synapse.hsynapse.theta_d_GB = fit_params["a00"]*c_pre + fit_params["a01"]*c_post
        if all(key in fit_params for key in ["a10", "a11"]) and extra_params["loc"] == "basal":
            # set basal potentiation threshold
            synapse.hsynapse.theta_p_GB = fit_params["a10"]*c_pre + fit_params["a11"]*c_post
        if all(key in fit_params for key in ["a20", "a21"]) and extra_params["loc"] == "apical":
            # set apical depression threshold
            synapse.hsynapse.theta_d_GB = fit_params["a20"]*c_pre + fit_params["a21"]*c_post
        if all(key in fit_params for key in ["a30", "a31"]) and extra_params["loc"] == "apical":
            # set apical potentiation threshold
            synapse.hsynapse.theta_p_GB = fit_params["a30"]*c_pre + fit_params["a31"]*c_post


def _c_pre_finder_process(bc, fit_params, syn_extra_params, pre_gid, post_gid, fixhp):
    """
    Multiprocessing subprocess for `c_pre_finder()`
    Delivers spike from `pre_gid` and measures the Ca++ transient at the synapses on `post_gid`
    """
    logger.debug("c_pre finder process")
    ssim = bglibpy.SSim(bc)
    ssim.instantiate_gids([post_gid], synapse_detail=1, add_synapses=True,
                          pre_spike_trains={pre_gid: [1000.]},
                          intersect_pre_gids=[pre_gid])
    cell = ssim.cells[post_gid]
    # hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # setup global parameters
    if fit_params is not None:
        _set_global_params(fit_params)
    # initialize effcai recorder
    recorder = {}
    # setup synapses
    syn_idx = []
    for syn_id, synapse in cell.synapses.items():
        syn_idx.append(syn_id)
        logger.debug("Configuring synapse %d", syn_id)
        # configure local parameters
        _set_local_params(synapse, fit_params, syn_extra_params[syn_id])
        # set recorder
        recorder[syn_id] = bglibpy.neuron.h.Vector()
        recorder[syn_id].record(synapse.hsynapse._ref_effcai_GB)
        # override rho
        synapse.hsynapse.rho0_GB = 1
        # override Use
        synapse.hsynapse.Use0_TM = 1
        synapse.hsynapse.Use_p_TM = 1
        # override gmax_AMPA
        synapse.hsynapse.gmax0_AMPA = synapse.hsynapse.gmax_p_AMPA
        # disable LTP / LTD
        synapse.hsynapse.theta_d_GB = -1
        synapse.hsynapse.theta_p_GB = -1
    # run
    ssim.run(2500, cvode=True)
    logger.debug("Simulation completed")
    # compute calcium peak
    return {syn_id: recorder[syn_id].max() for syn_id in syn_idx}


def c_pre_finder(bc, fit_params, syn_extra_params, pre_gid, post_gid, fixhp=True):
    """Replays spike from `pre_gid` and measures Ca++ transient in synapses on `post_gid`"""
    pool = multiprocessing.Pool(processes=1)
    c_pre = pool.apply(_c_pre_finder_process, [bc, fit_params, syn_extra_params, pre_gid, post_gid, fixhp])
    pool.terminate()
    logger.debug("C_pre: %s", str(c_pre))
    return c_pre


def _c_post_finder_process(bc, stimulus, fit_params, syn_extra_params, pre_gid, post_gid, fixhp):
    """
    Multiprocessing subprocess for `c_post_finder()`
    Injects (precalculated) stimulus to the `post_gid` to make it fire 1 AP and measures the Ca++ transient
    (from the backpropagiting AP) at the synapses made by `pre_gid`
    """
    logger.debug("c_post finder process")
    ssim = bglibpy.SSim(bc)
    ssim.instantiate_gids([post_gid], synapse_detail=1, add_synapses=True,
                          intersect_pre_gids=[pre_gid])
    cell = ssim.cells[post_gid]
    # hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # add stimuli
    tstim = bglibpy.neuron.h.TStim(0.5, sec=cell.soma)
    stim_duration = (stimulus["nspikes"] - 1) * 1000./stimulus["freq"] + stimulus["width"]
    tstim.train(stimulus["offset"], stim_duration, stimulus["amp"], stimulus["freq"], stimulus["width"])
    cell.persistent.append(tstim)
    # setup global parameters
    if fit_params is not None:
        _set_global_params(fit_params)
    # initialize effcai recorder
    recorder = {}
    # setup synapses
    syn_idx = []
    for syn_id, synapse in cell.synapses.items():
        syn_idx.append(syn_id)
        logger.debug("Configuring synapse %d", syn_id)
        # configure local parameters
        _set_local_params(synapse, fit_params, syn_extra_params[syn_id])
        # set recorder
        recorder[syn_id] = bglibpy.neuron.h.Vector()
        recorder[syn_id].record(synapse.hsynapse._ref_effcai_GB)
        # disable LTP / LTD
        synapse.hsynapse.theta_d_GB = -1
        synapse.hsynapse.theta_p_GB = -1
    # run
    ssim.run(1500, cvode=True)
    logger.debug("Simulation completed")
    # get soma voltage and simulation time vector
    t = np.array(ssim.get_time())
    v = np.array(ssim.get_voltage_traces()[post_gid])
    # get spike timing (skip 200 ms)
    dt_int = 0.025
    tdense = np.linspace(min(t), max(t), int((max(t)-min(t))/dt_int))
    vdense = np.interp(tdense, t, v)
    spikes = np.array([tdense[i+1] for i in range(int(200/dt_int), len(vdense) - 1)
                       if vdense[i] < -30 and vdense[i+1] >= -30])
    # compute c_post
    c_post = {syn_id: recorder[syn_id].max() for syn_id in syn_idx}
    # store results
    results = {"c_post": c_post,
               "c_trace": {syn_id: np.array(recorder[syn_id]) for syn_id in syn_idx},
               "t": t,
               "v": v,
               "t_spikes": spikes,
               "t_stimuli": np.array(tstim.tvec)[:-1:4]}
    return results


def c_post_finder(bc, fit_params, syn_extra_params, pre_gid, post_gid, stimulus, fixhp=True):
    """
    Finds c_post - the calcium transient in `post_gid` at synapses made by `pre_gid`.
    To do so it calculates the necessary current to make `post_gid` fire a single AP
    and then measures the Ca++ transient of the backpropagating AP.
    """
    # find c_post
    logger.debug("Stimulating cell with {} nA pulse ({} ms)".format(stimulus["amp"], stimulus["width"]))
    pool = multiprocessing.Pool(processes=1)
    results = pool.apply(_c_post_finder_process, [bc, stimulus, fit_params, syn_extra_params,
                                                  pre_gid, post_gid, fixhp])
    pool.terminate()
    # validate number of spikes
    logger.debug("Spike timing: {}".format(results["t_spikes"]))
    if len(results["t_spikes"]) < 1:
        # special case, small integration differences with threshold detection sim
        logger.debug("Cell not spiking as expected during c_post, "
                     "attempting to bump stimulus amplitude before failing...")
        # find c_post
        amp = stimulus["amp"] + 0.05
        logger.debug("Stimulating cell with %f nA pulse", amp)
        stimulus = {"nspikes": 1, "freq": 0.1, "width": stimulus["width"], "offset": 1000., "amp": amp}
        pool = multiprocessing.Pool(processes=1)
        results = pool.apply(_c_post_finder_process, [bc, stimulus, fit_params, syn_extra_params,
                                                      pre_gid, post_gid, fixhp])
        pool.terminate()
        return results["c_post"] if len(results["t_spikes"]) == 1 else None
    return results["c_post"]

