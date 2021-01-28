"""
Single cell simulations in BGLibPy
author: Giuseppe Chindemi, last update 03.2020
"""

import os
import re
import warnings
import logging
import multiprocessing
import numpy as np
import bglibpy
from plastifyre.epg import ParamsGenerator
from plastifyre.synapse import Synapse
from functools import lru_cache

# Default parameters, can be overriten by CL arguments
circuitpath = "/gpfs/bbp.cscs.ch/project/proj64/circuits/O1.v6a/20181207/CircuitConfig"
extra_recipe = "/gpfs/bbp.cscs.ch/project/proj32/glusynapse_20190926_release/recipe.csv"
default_synrec = ["rho_GB", "Use_TM", "gmax_AMPA",
                  "cai_CR", "vsyn", "ica_NMDA", "ica_VDCC", "effcai_GB"]
default_syn_prop = ["Cpre", "Cpost", "loc", "Use0_TM", "Dep_TM", "Fac_TM",
                   "Nrrp_TM", "gmax0_AMPA", "gmax_NMDA", "volume_CR",
                   "synapseID", "theta_d_GB", "theta_p_GB"]
default_modprop = ["tau_exp_GB", "gamma_d_GB", "gamma_p_GB"]
bglibpy.neuron.h.cvode.atolscale("v", .1)
DEBUG = False

# Set cache for spiking thresholds
with_cache = lru_cache(128)

# Configure logger
logger = logging.getLogger(__name__)

# Patch bglibpy
bglibpy.Synapse = Synapse


def _set_global_params(allparams):
    """Sets global parameters of the simulation"""
    logger.debug("Setting global parameters")
    for param_name, param_val in allparams.items():
        if re.match(".*_GluSynapse$", param_name):
            setattr(bglibpy.neuron.h, param_name, param_val)
            logger.debug("\t%s = %f", param_name, getattr(bglibpy.neuron.h, param_name))


def _set_local_params(synapse, fit_params, extra_params, c_pre=0., c_post=0.):
    """Sets synaptic parameters in BGLibPy"""
    # Update basic synapse parameters
    for param in extra_params:
        if param == "loc":
            continue
        setattr(synapse.hsynapse, param, extra_params[param])
    # Update thresholds
    if fit_params is not None:
        if all(key in fit_params for key in ["a00", "a01"]) and extra_params["loc"] == "basal":
            # Set basal depression threshold
            synapse.hsynapse.theta_d_GB = fit_params["a00"]*c_pre + fit_params["a01"]*c_post
        if all(key in fit_params for key in ["a10", "a11"]) and extra_params["loc"] == "basal":
            # Set basal potentiation threshold
            synapse.hsynapse.theta_p_GB = fit_params["a10"]*c_pre + fit_params["a11"]*c_post
        if all(key in fit_params for key in ["a20", "a21"]) and extra_params["loc"] == "apical":
            # Set apical depression threshold
            synapse.hsynapse.theta_d_GB = fit_params["a20"]*c_pre + fit_params["a21"]*c_post
        if all(key in fit_params for key in ["a30", "a31"]) and extra_params["loc"] == "apical":
            # Set apical potentiation threshold
            synapse.hsynapse.theta_p_GB = fit_params["a30"]*c_pre + fit_params["a31"]*c_post


def _c_pre_finder_process(bcpath, fit_params, invivo, pre_gid, post_gid, fixhp):
    """
    Multiprocessing subprocess for `c_pre_finder()`
    Replays spike from `pre_gid` (atm it's in out.dat written elsewhere - TODO: update that)
    and measures the Ca++ transient at the synapses on `post_gid`
    """
    logger.debug("Cpre finder process")
    ssim = bglibpy.SSim(bcpath)
    if pre_gid is None:
        pre_gid = list(ssim.all_targets_dict["PreCell"])[0]
    if post_gid is None:
        post_gid = list(ssim.all_targets_dict["PostCell"])[0]
    ssim.instantiate_gids([post_gid], synapse_detail=2, add_stimuli=True,
                          add_replay=True, add_synapses=True,
                          intersect_pre_gids=[pre_gid])
    cell = ssim.cells[post_gid]
    # Hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # Generate supplementary model parameters
    pgen = ParamsGenerator(circuitpath, extra_recipe)
    syn_extra_params = pgen.generate_params(pre_gid, post_gid)
    # Setup global parameters
    if fit_params is not None:
        _set_global_params(fit_params)
    # Enable in vivo mode (global)
    if invivo:
        bglibpy.neuron.h.cao_CR_GluSynapse = 1.2  # mM
        # TODO Set global cao
    # Initialize effcai recorder
    recorder = {}
    # Setup synapses
    for syn_id in cell.synapses.keys():
        synapse = cell.synapses[syn_id]
        logger.debug("Configuring synapse %d", syn_id)
        # Configure local parameters
        _set_local_params(synapse, fit_params, syn_extra_params[(post_gid, syn_id)])
        # Set recorder
        recorder[syn_id] = bglibpy.neuron.h.Vector()
        recorder[syn_id].record(synapse.hsynapse._ref_effcai_GB)
        # Override Rho
        synapse.hsynapse.rho0_GB = 1
        # Override Use
        synapse.hsynapse.Use0_TM = 1
        synapse.hsynapse.Use_p_TM = 1
        # Override gmax_AMPA
        synapse.hsynapse.gmax0_AMPA = synapse.hsynapse.gmax_p_AMPA
        # Disable LTP / LTD
        synapse.hsynapse.theta_d_GB = -1
        synapse.hsynapse.theta_p_GB = -1
    # Run
    ssim.run(3000, cvode=True)
    logger.debug("Simulation completed")
    # Compute calcium peak
    return {(post_gid, syn_id): recorder[syn_id].max() for syn_id in cell.synapses.keys()}


def c_pre_finder(basedir, fit_params=None, invivo=False, pre_gid=None, post_gid=None, fixhp=True):
    """Replays spike from `pre_gid` and measures Ca++ transient in synapses on `post_gid`"""
    logger.info("Calibrating Cpre...")
    # Load simulation
    bcpath = os.path.join(basedir, "BlueConfig")
    ssim = bglibpy.SSim(bcpath)
    pre_gids = [pre_gid]
    # Special case: multiple connections
    if "ExtraPreCell" in ssim.all_targets_dict:
        pre_gids = pre_gids + list(ssim.all_targets_dict["ExtraPreCell"])
    c_pre = {}
    for pg in pre_gids:
        pool = multiprocessing.Pool(processes=1)
        c_pre.update(pool.apply(_c_pre_finder_process, [bcpath, fit_params, invivo, pg, post_gid, fixhp]))
        pool.terminate()
    logger.debug("C_pre: %s", str(c_pre))
    return c_pre


def _runsinglecell_proc(bc, post_gid, stimulus, results, fixhp):
    """Multiprocessing subprocess for `runsinglecell()`"""
    # Create simulation
    ssim = bglibpy.SSim(bc)
    if post_gid is None:
        post_gid = list(ssim.all_targets_dict["PostCell"])[0]
    ssim.instantiate_gids([post_gid])
    cell = ssim.cells[post_gid]
    # Hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # Add stimuli
    tstim = bglibpy.neuron.h.TStim(0.5, sec=cell.soma)
    stim_duration = (stimulus["nspikes"] - 1) * 1000./stimulus["freq"] + stimulus["width"]
    tstim.train(stimulus["offset"], stim_duration, stimulus["amp"], stimulus["freq"], stimulus["width"])
    cell.persistent.append(tstim)
    # Runsim
    ssim.run(stimulus["offset"] + stim_duration + 200., cvode=True)
    # Get soma voltage and simulation time vector
    t = np.array(ssim.get_time())
    v = np.array(ssim.get_voltage_traces()[post_gid])
    # Get spike timing (skip 200 ms)
    dt_int = 0.025
    tdense = np.linspace(min(t), max(t), int((max(t)-min(t))/dt_int))
    vdense = np.interp(tdense, t, v)
    spikes = np.array([tdense[i+1] for i in range(int(200/dt_int), len(vdense) - 1)
                       if vdense[i] < -30 and vdense[i+1] >= -30])
    # Store results
    results["t"] = t
    results["v"] = v
    results["t_spikes"] = spikes
    results["t_stimuli"] = np.array(tstim.tvec)[:-1:4]
    results.update(stimulus)


def runsinglecell(bc, post_gid, stimulus, fixhp=True):
    """Runs single cell simulation with given stimulus"""
    manager = multiprocessing.Manager()
    logger.debug("Submitting simulation: post_gid={}, f={}, a={}".format(post_gid, stimulus["freq"], stimulus["amp"]))
    results = manager.dict()
    p = multiprocessing.Process(target=_runsinglecell_proc, args=(bc, post_gid, stimulus, results, fixhp))
    p.start()
    p.join()
    return dict(results)


def _c_post_finder_process(basedir, stimulus, fit_params, invivo, pre_gid, post_gid, fixhp):
    """
    Multiprocessing subprocess for `c_post_finder()`
    Injects (precalculated) stimulus to the `post_gid` to make it fire 1 AP and measures the Ca++ transient
    (from the backpropagiting AP) at the synapses made by `pre_gid`
    """
    logger.debug("Cpost finder process")
    bcpath = os.path.join(basedir, "BlueConfig")
    ssim = bglibpy.SSim(bcpath)
    if pre_gid is None:
        pre_gid = list(ssim.all_targets_dict["PreCell"])[0]
    if post_gid is None:
        post_gid = list(ssim.all_targets_dict["PostCell"])[0]
    ssim.instantiate_gids([post_gid], synapse_detail=2,
                          add_synapses=True,
                          intersect_pre_gids=[pre_gid])
    cell = ssim.cells[post_gid]
    # Hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # Add stimuli
    tstim = bglibpy.neuron.h.TStim(0.5, sec=cell.soma)
    stim_duration = (stimulus["nspikes"] - 1) * 1000./stimulus["freq"] + stimulus["width"]
    tstim.train(stimulus["offset"], stim_duration, stimulus["amp"], stimulus["freq"], stimulus["width"])
    cell.persistent.append(tstim)
    # Generate supplementary model parameters
    pgen = ParamsGenerator(circuitpath, extra_recipe)
    syn_extra_params = pgen.generate_params(pre_gid, post_gid)
    # Setup global parameters
    if fit_params is not None:
        _set_global_params(fit_params)
    # Enable in vivo mode (global)
    if invivo:
        bglibpy.neuron.h.cao_CR_GluSynapse = 1.2  # mM
        # TODO Set global cao
    # Initialize effcai recorder
    recorder = {}
    # Setup synapses
    for syn_id in cell.synapses.keys():
        synapse = cell.synapses[syn_id]
        logger.debug("Configuring synapse %d", syn_id)
        # Configure local parameters
        _set_local_params(synapse, fit_params, syn_extra_params[(post_gid, syn_id)])
        # Set recorder
        recorder[syn_id] = bglibpy.neuron.h.Vector()
        recorder[syn_id].record(synapse.hsynapse._ref_effcai_GB)
        # Disable LTP / LTD
        synapse.hsynapse.theta_d_GB = -1
        synapse.hsynapse.theta_p_GB = -1
    # Run
    ssim.run(1500, cvode=True)
    logger.debug("Simulation completed")
    # Get soma voltage and simulation time vector
    t = np.array(ssim.get_time())
    v = np.array(ssim.get_voltage_traces()[post_gid])
    # Get spike timing (skip 200 ms)
    dt_int = 0.025
    tdense = np.linspace(min(t), max(t), int((max(t)-min(t))/dt_int))
    vdense = np.interp(tdense, t, v)
    spikes = np.array([tdense[i+1] for i in range(int(200/dt_int), len(vdense) - 1)
                       if vdense[i] < -30 and vdense[i+1] >= -30])
    # Compute c_post
    c_post = {(post_gid, syn_id): recorder[syn_id].max() for syn_id in cell.synapses.keys()}
    # Store results
    results = {"c_post": c_post,
               "c_trace": {(post_gid, syn_id): np.array(recorder[syn_id]) for syn_id in cell.synapses.keys()},
               "t": t,
               "v": v,
               "t_spikes": spikes,
               "t_stimuli": np.array(tstim.tvec)[:-1:4]}
    return results


@with_cache
def spike_threshold_finder(bctest_path, post_gid, nspikes,
                           freq, width, offset, min_amp, max_amp,
                           nlevels, fixhp=True):
    """
    Finds the min amplitude of stimulus current (within the range [`min_amp`, `max_amp`])
    that makes the `post_gid` fire `nspikes` APs (given `freq`, `width`, `offset`)
    using a binary search (parametrized by `nlevels`).
    """
    # Initialize search grid
    candidate_amp = np.linspace(min_amp, max_amp, nlevels)
    # Find suitable amplitude (binary search, leftmost element)
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
        simres[m] = runsinglecell(bctest_path, post_gid, stim, fixhp)
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


def c_post_finder(basedir, fit_params=None, invivo=False, pre_gid=None, post_gid=None, fixhp=True):
    """
    Finds c_post - the calcium transient in `post_gid` at synapses made by `pre_gid`.
    To do so it calculates the necessary current to make `post_gid` fire a single AP
    and then measures the Ca++ transient of the backpropagating AP.
    """
    logger.info("Calibrating Cpost...")
    # Calibrate current pulse amplitude
    bc_path = os.path.join(basedir, "BlueConfig")
    for pulse_width in [1.5, 3]:
        simres = spike_threshold_finder(bc_path, post_gid, 1, 0.1, pulse_width, 1000., 0.05, 5., 100, fixhp)
        if simres is not None:
            break
    else:
        raise RuntimeError("Could not find a suitable stimulation amplitude for sim %s" % basedir)
    # Find Cpost
    amp = simres["amp"]
    width = simres["width"]
    logger.debug("Stimulating cell with {} nA pulse ({} ms)".format(amp, width))
    stimulus = {"nspikes": 1, "freq": 0.1, "width": width, "offset": 1000., "amp": amp}  # Default stimulus
    # Special case: multiple connections
    ssim = bglibpy.SSim(bc_path)
    pre_gids = [pre_gid]
    if "ExtraPreCell" in ssim.all_targets_dict:
        pre_gids = pre_gids + list(ssim.all_targets_dict["ExtraPreCell"])
    c_post = {}
    for pg in pre_gids:
        pool = multiprocessing.Pool(processes=1)
        results = pool.apply(_c_post_finder_process, [basedir, stimulus, fit_params, invivo, pg, post_gid, fixhp])
        pool.terminate()
        # Validate number of spikes
        logger.debug("Spike timing: {}".format(results["t_spikes"]))
        if len(results["t_spikes"]) < 1:
            # Special case, small integration differences with threshold detection sim
            warnings.warn("Cell not spiking as expected during Cpost,"
                          "attempting to bump stimulus amplitude before failing...")
            # Find Cpost
            amp = simres["amp"] + 0.05
            logger.debug("Stimulating cell with %f nA pulse", amp)
            stimulus = {"nspikes": 1, "freq": 0.1, "width": 1.5, "offset": 1000., "amp": amp}  # Default stimulus
            pool = multiprocessing.Pool(processes=1)
            results = pool.apply(_c_post_finder_process, [basedir, stimulus, fit_params, invivo, pg, post_gid, fixhp])
            pool.terminate()
        assert len(results["t_spikes"]) == 1
        # Return Cpost
        c_post.update(results["c_post"])
    return c_post


def _runconnectedpair_process(results, basedir, c_pre, c_post, fit_params, synrec, fastforward, invivo, fixhp):
    """
    Multiprocessing subprocess for `runconnectedpair()`
    Replays spike from pre_gid (atm pre_gid is specified in BlueConfig
    and spikes are in out.dat written elsewhere - TODO: update those)
    and runs single cell sim of post_gid (also specified in BlueConfig) with `fastforward` option.
    """
    bcpath = os.path.join(basedir, "BlueConfig")
    ssim = bglibpy.SSim(bcpath)
    logger.debug("Loaded simulation")
    post_gid = list(ssim.all_targets_dict["PostCell"])[0]
    pre_gid = list(ssim.all_targets_dict["PreCell"])[0]
    pre_gids = [pre_gid]
    # Special case: multiple connections
    if "ExtraPreCell" in ssim.all_targets_dict:
        pre_gids = pre_gids + list(ssim.all_targets_dict["ExtraPreCell"])
    ssim.instantiate_gids([post_gid], synapse_detail=2, add_stimuli=True,
                          add_replay=True, add_synapses=True,
                          intersect_pre_gids=pre_gids)
    cell = ssim.cells[post_gid]
    prespikes = np.unique(np.loadtxt(os.path.join(basedir, "out.dat"),
        skiprows=1)[:, 0])
    # Hyperpolarization workaround
    if fixhp:
        for sec in cell.somatic + cell.axonal:
            sec.uninsert("SK_E2")
    # NMDA Spike injection workaround
    if "Stimulus_NMDASpike" in ssim.bc:
        # Find position of first synapse
        syn_id = list(cell.synapses.keys())[0]
        segx = cell.synapses[syn_id].post_segx
        sec = cell.get_hsection(cell.synapses[syn_id].isec)  # syn_description[2]
        # Create stimulus
        tstim = bglibpy.neuron.h.TStim(segx, sec=sec)
        tstim.train(float(ssim.bc["Stimulus_NMDASpike"]["Delay"]),
                float(ssim.bc["Stimulus_NMDASpike"]["Duration"]),
                float(ssim.bc["Stimulus_NMDASpike"]["AmpStart"]),
                float(ssim.bc["Stimulus_NMDASpike"]["Frequency"]),
                float(ssim.bc["Stimulus_NMDASpike"]["Width"]))
        cell.persistent.append(tstim)
    # Generate supplementary model parameters
    pgen = ParamsGenerator(circuitpath, extra_recipe)
    syn_extra_params = {}
    for pg in pre_gids:
        syn_extra_params.update(pgen.generate_params(pg, post_gid))
    # Set fitted model parameters
    if fit_params is not None:
        _set_global_params(fit_params)
    # Enable in vivo mode (global)
    if invivo:
        bglibpy.neuron.h.cao_CR_GluSynapse = 1.2  # mM
        # TODO Set global cao
    # Store model properties
    modprop = {key: getattr(bglibpy.neuron.h, "%s_GluSynapse" % key) for key in default_modprop}
    # Allocate recording vectors
    actual_synrec = default_synrec if synrec is None else synrec
    time_series = {key: list() for key in actual_synrec}
    syn_prop = {key: list() for key in default_syn_prop}
    # Setup synapses
    for syn_id in cell.synapses.keys():
        logger.debug("Configuring synapse %d", syn_id)
        synapse = cell.synapses[syn_id]
        # Set local parameters
        _set_local_params(synapse, fit_params, syn_extra_params[(post_gid, syn_id)],
                          c_pre[post_gid, syn_id], c_post[post_gid, syn_id])
        # Enable in vivo mode (synapse)
        if invivo:
            synapse.hsynapse.Use0_TM = 0.15*synapse.hsynapse.Use0_TM
            synapse.hsynapse.Use_d_TM = 0.15*synapse.hsynapse.Use_d_TM
            synapse.hsynapse.Use_p_TM = 0.15*synapse.hsynapse.Use_p_TM
        # Setting up recordings
        for key, lst in time_series.items():
            recorder = bglibpy.neuron.h.Vector()
            recorder.record(getattr(synapse.hsynapse, "_ref_%s" % key))
            lst.append(recorder)
        # Store synapse properties
        for key, lst in syn_prop.items():
            if key == "Cpre":
                lst.append(c_pre[post_gid, syn_id])
            elif key == "Cpost":
                lst.append(c_post[post_gid, syn_id])
            elif key == "loc":
                lst.append(syn_extra_params[(post_gid, syn_id)]["loc"])
            else:
                lst.append(getattr(synapse.hsynapse, key))
        # Show all params
        for attr in dir(synapse.hsynapse):
            if re.match("__.*", attr) is None:
                logger.debug("%s = %s", attr, str(getattr(synapse.hsynapse, attr)))
    # Run
    endtime = 30000 if DEBUG else float(ssim.bc.Run.Duration)
    if fastforward is not None:
        # Run until fastforward point
        logger.debug("Fastforward enabled, simulating %d seconds...", fastforward/1000.)
        ssim.run(fastforward, cvode=True)
        # Fastforward synapses
        logger.debug("Updating synapses...")
        for syn_id in cell.synapses.keys():
            logger.debug("Configuring synapse %d", syn_id)
            synapse = cell.synapses[syn_id]
            if synapse.hsynapse.rho_GB >= 0.5:
                synapse.hsynapse.rho_GB = 1.
                synapse.hsynapse.Use_TM = synapse.hsynapse.Use_p_TM
                synapse.hsynapse.gmax_AMPA = synapse.hsynapse.gmax_p_AMPA
            else:
                synapse.hsynapse.rho_GB = 0.
                synapse.hsynapse.Use_TM = synapse.hsynapse.Use_d_TM
                synapse.hsynapse.gmax_AMPA = synapse.hsynapse.gmax_d_AMPA
        # Complete run
        logger.debug("Simulating remaining %d seconds...", (endtime-fastforward)/1000.)
        bglibpy.neuron.h.cvode_active(1)
        bglibpy.neuron.h.continuerun(endtime)
    else:
        logger.debug("Simulating %d seconds...", endtime/1000.)
        ssim.run(endtime, cvode=True)
    logger.debug("Simulation completed")
    # Collect all properties
    syn_prop.update(modprop)
    # Collect Results
    results["t"] = np.array(ssim.get_time())
    results["v"] = np.array(ssim.get_voltage_traces()[post_gid])
    results["prespikes"] = np.array(prespikes)
    results["syn_prop"] = syn_prop
    for key, lst in time_series.items():
        results[key] = np.transpose([np.array(rec) for rec in lst])


def runconnectedpair(basedir, fit_params=None, synrec=None, fastforward=None, invivo=False, fixhp=True):
    """High level function with `fastforward` option (most of the parameters are read from BlueConfig)"""
    # Get reference thetas
    c_pre = c_pre_finder(basedir, fit_params, invivo)
    c_post = c_post_finder(basedir, fit_params, invivo)
    # Run main simulation
    logger.info("Simulating %s...", basedir)
    manager = multiprocessing.Manager()
    results = manager.dict()
    child_proc = multiprocessing.Process(target=_runconnectedpair_process,
                                         args=[results, basedir, c_pre, c_post, fit_params,
                                               synrec, fastforward, invivo, fixhp])
    child_proc.start()
    child_proc.join()
    return dict(results)

