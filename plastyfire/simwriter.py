"""
Writes files for model optimization (more involved)
and simple ones for generalization (AKA finding thresholds based on the optimized parameters)
last modified: András Ecker 06.2024
"""

import os
import h5py
import pickle
import pathlib
import shutil
import warnings
import json
from tqdm import tqdm
import numpy as np
import pandas as pd
from bluepysnap import Circuit
from conntility.connectivity import ConnectivityMatrix

from plastyfire.config import OptConfig, Config
from plastyfire.simulator import spike_threshold_finder

MIN2MS = 60 * 1000.
OPT_CPU_TIME = 2.  # heuristics: it takes ~twice as much compute time (with CVode) to simulate biological time
CPU_TIME = 1.  # heuristics: c_pre and c_post for a single connection takes ~1 minute to simulate/calculate


def check_geom_constraint(conn_mat, pre_mtype, post_gid, max_dist):
    """Check if cell has any presynaptic partners within `max_dists`"""
    nrn = conn_mat.vertices
    post_mtype = nrn.loc[nrn["node_ids"] == post_gid, "mtype"].to_numpy()[0]
    coords = nrn.loc[nrn["node_ids"] == post_gid, ["ss_flat_x", "depth", "ss_flat_y"]]
    dists = (nrn.loc[nrn["mtype"].isin(pre_mtype), ["ss_flat_x", "depth", "ss_flat_y"]] - coords.to_numpy()).abs()
    if post_mtype in pre_mtype:
        dists.drop(coords.index, inplace=True)
    idx = dists.loc[(dists["ss_flat_x"] < max_dist[0]) &
                    (dists["depth"] < max_dist[1]) &
                    (dists["ss_flat_y"] < max_dist[2])].index.to_numpy()
    valid_gids = nrn.loc[idx, "node_ids"].to_numpy()
    sub_mat = conn_mat.submatrix(valid_gids, sub_gids_post=[post_gid])
    if sub_mat.size:
        return valid_gids[sub_mat.tocoo().row]
    else:
        return None


def check_electrical_constraint(sim_config, gid, stim_config, save_dir):
    """Check if the cell can fire correctly at every stim. frequency
    (and save params. of current injection that makes it fire)"""
    pklf_name = os.path.join(save_dir, "%i.pkl" % gid)
    if os.path.isfile(pklf_name):  # check if these sims were already run...
        return True
    results = {}
    for freq in stim_config["freq"]:
        simres = spike_threshold_finder(sim_config, gid, stim_config["nspikes"],
                                        freq, stim_config["width"], stim_config["offset"], stim_config["amp_min"],
                                        stim_config["amp_max"], stim_config["amp_lev"], fixhp=True)
        if simres is None:
            return False
        else:
            results[freq] = simres
    # Store results for manual validation and simulation setup
    with open(pklf_name, "wb") as f:
        pickle.dump(results, f, -1)
    return True


def save_spikes(h5f_name, prefix, spike_times, spiking_gids):
    """Save spikes to SONATA format"""
    assert (spiking_gids.shape == spike_times.shape)
    with h5py.File(h5f_name, "w") as h5f:
        grp = h5f.require_group("spikes/%s" % prefix)
        grp.create_dataset("timestamps", data=spike_times)
        grp["timestamps"].attrs["units"] = "ms"
        grp.create_dataset("node_ids", data=spiking_gids, dtype=int)


def get_cpu_time(n_afferents):
    """CPU time heuristics: 5 min setup and stimulus calculation + 10 mins. extra just to make sure +
    gid specific simulation time, based on the number of its afferent gids"""
    cpu_time_sec = (15 + n_afferents * CPU_TIME) * 60
    h, m = np.divmod(cpu_time_sec, 3600)
    m, s = np.divmod(m, 60)
    cpu_time_str = "%.2i:%.2i:%.2i" % (h, m, s)
    qos = "#SBATCH --qos=longjob" if h >= 24 else ""
    return cpu_time_str, qos


class OptSimWriter(OptConfig):
    """Class to setup single cell simulations for the optimization of model parameters"""
    def find_pairs(self):
        """Finds connected pairs of gids based on constraints specified in the config"""
        save_dir = os.path.join(self.out_dir, "single_cells")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # Write simple simulation config (for checking electrical constraints)
        sim_config = {"run": {"dt": 0.025, "tstop": self.T, "random_seed": self.seed},
                      "network": self.circuit_config,
                      "node_sets_file": self.node_set,
                      "node_set": self.target,
                      "output": {"output_dir": os.path.join(self.sims_dir, "out")}}
        sim_config_path = os.path.join(save_dir, "simulation_config.json")
        with open(sim_config_path, "w", encoding="utf-8") as f:
            json.dump(sim_config, f, indent=4)

        c = Circuit(self.circuit_config)
        # Get connectivity matrix and flatmap locations (used for distance based filtering) with `conntility`
        load_cfg = {"loading": {"base_target": self.target,
                                "properties": ["mtype", "x", "y", "z",
                                               "ss_flat_x", "ss_flat_y", "depth"]},
                    "filtering": [{"column": "mtype",
                                   "values": np.unique(self.pre_mtype + self.post_mtype).tolist()}]}
        conn_mat = ConnectivityMatrix.from_bluepy(c, load_cfg, connectome=self.edge_pop)
        nrn = conn_mat.vertices
        post_gids = nrn.loc[nrn["mtype"].isin(self.post_mtype), "node_ids"].to_numpy()
        np.random.seed(self.seed)
        np.random.shuffle(post_gids)
        # Find pairs (TODO: parallelize)
        pairs = []
        pbar = tqdm(total=self.npairs, desc="Finding pairs")
        for i, post_gid in enumerate(post_gids):
            # Check if post_gid fulfills constraints
            pre_gids = check_geom_constraint(conn_mat, self.pre_mtype, post_gid, self.max_dist)
            if pre_gids is not None:
                if check_electrical_constraint(sim_config_path, post_gid, self.config["stimulus"], save_dir):
                    np.random.seed(self.seed + i)
                    pairs.append((np.random.choice(pre_gids, 1)[0], post_gid))  # select a random presynaptic partner
                    pbar.update(1)
            if len(pairs) == self.npairs:
                break
        pbar.close()
        if len(pairs) < self.npairs:
            warnings.warn("Not enough pairs found")
        return pairs

    def write_batch_sript(self, f_name, templ, cpu_time):
        """Writes single cell batch script"""
        workdir = os.path.dirname(f_name)
        tmp = os.path.split(workdir)
        name = "%s_%s" % (tmp[1], os.path.split(tmp[0])[1])
        fastforward = self.fastforward
        if fastforward is None:
            fastforward = self.C01_duration * MIN2MS + self.nreps * self.T
        with open(f_name, "w+", encoding="latin1") as f:
            f.write(templ.format(name=name, cpu_time=cpu_time, qos="#SBATCH --chdir=%s" % workdir, log=name,
                                 env=self.env, run=self.run, args="--fastforward=%.1f" % fastforward))

    def write_sim_files(self, pairs):
        """Writes pair, frequency, and dt specific `simulation_config.json` used by `bluecellulab`
        and batch scripts to launch single cell sims"""
        # Copy config file to sims dir (just to make sure that one can more or less know what was run...)
        basedir = os.path.split(os.path.split(self.out_dir)[0])[0]
        if not os.path.exists(basedir):
            os.makedirs(basedir)
        shutil.copyfile(self._config_path, os.path.join(basedir, "%s.yaml" % self.label))
        # Generate presynaptic spike times (for C01 and C02)
        n_spikes_before = int(self.C01_duration * MIN2MS / self.C01_T)
        n_spikes_after = int(self.C02_duration * MIN2MS / self.C02_T)
        before_pre_spikes = [self.offset + i * self.C01_T for i in range(n_spikes_before)]
        after_pre_spikes = [self.offset + n_spikes_before * self.C01_T + self.nreps * self.T +
                            i * self.C02_T for i in range(n_spikes_after)]
        before_duration = n_spikes_before * self.C01_T
        pairing_duration = self.nreps * self.T
        t_stop = before_duration + pairing_duration + n_spikes_after * self.C02_T
        # CPU time heuristics
        h, m = np.divmod(OPT_CPU_TIME * t_stop / 1000., 3600)
        m, s = np.divmod(m, 60)
        cpu_time = "%.2i:%.2i:%.2i" % (h, m, s)
        with open(os.path.join("templates", "simulation.batch.tmpl"), "r") as f:
            templ = f.read()

        all_sims = []
        for freq in self.freq:
            for dt in self.dt:
                for pre_gid, post_gid in pairs:
                    workdir = os.path.join(self.out_dir, "%i-%i" % (pre_gid, post_gid),
                                           "%iHz_%ims" % (int(freq), int(dt)))
                    if not os.path.exists(workdir):
                        os.makedirs(workdir)
                    # Write node set with idx of pre- and postsynaptic neurons
                    jsonf_name = os.path.join(workdir, "node_sets.json")
                    node_sets = {"precell": {"node_id": [int(pre_gid)], "population": self.node_pop},
                                 "postcell": {"node_id": [int(post_gid)], "population": self.node_pop}}
                    with open(jsonf_name, "w", encoding="utf-8") as f:
                        json.dump(node_sets, f, indent=4)
                    try:  # load pulse amplitude and compute spike delays at the given frequency (independent of dt...)
                        with open(os.path.join(self.out_dir, "single_cells", "%i.pkl" % post_gid), "rb") as f:
                            simres = pickle.load(f)
                        amplitude = simres[freq]["amp"]
                        spike_delay = simres[freq]["t_spikes"] - simres[freq]["t_stimuli"]
                    except IOError:  # fallback to default current pulse
                        warnings.warn("Cannot read stimulus amplitude from cache, using 1 nA")
                        spike_delay = self.nspikes * [self.width / 2.]
                        amplitude = 1.0  # nA
                    # Generate postsynaptic stimulus (only one period, as the stimulus will be periodic)
                    isi = 1000.0 / freq  # Inter Spike Interval (ms)
                    post_spikes = np.array([self.offset + before_duration + i * isi for i in range(self.nspikes)])
                    inputs = {"pulse%i" % i: {"input_type": "current_clamp", "module": "pulse",
                                              "node_set": self.target,  # "postcell" (to be fixed in `bluecellulab`)
                                              "delay": post_spike, "duration": pairing_duration, "amp_start": amplitude,
                                              "width": self.width, "frequency": 1000. / self.T}
                              for i, post_spike in enumerate(post_spikes)}
                    # Generate (full) presynaptic spike train used as spike replay stimulus
                    pre_spikes = [self.offset + before_duration + i * isi - dt + spike_delay[i] + j * self.T
                                  for j in range(self.nreps) for i in range(self.nspikes)]
                    pre_spikes = np.array(before_pre_spikes + pre_spikes + after_pre_spikes)
                    h5f_name = os.path.join(workdir, "prespikes.h5")
                    save_spikes(h5f_name, self.node_pop, pre_spikes, pre_gid * np.ones(len(pre_spikes), dtype=int))
                    # Not adding spike replay to the `inputs` because one has to do it manually in `bluecellulab`
                    # (could be kept for `neurodamus`, but that breaks the current version of `bluecellulab`)
                    # inputs["prespikes"] = {"input_type": "spikes", "module": "synapse_replay", "node_set": "postcell",
                    #                        "delay": 0., "duration": t_stop, "spike_file": h5f_name}
                    # Write simulation config
                    sim_config = {"run": {"dt": 0.025, "tstop": t_stop, "random_seed": np.random.randint(1, 999999)},
                                  "network": self.circuit_config,
                                  "node_sets_file": jsonf_name,
                                  "node_set": "postcell",
                                  "output": {"output_dir": os.path.join(workdir, "out")},
                                  "inputs": inputs,
                                  "connection_overrides": [
                                      {"name": "plasticity", "source": "precell", "target": "postcell",
                                       "modoverride": "GluSynapse", "weight": 1.0}]}
                    with open(os.path.join(workdir, "simulation_config.json"), "w", encoding="utf-8") as f:
                        json.dump(sim_config, f, indent=4)
                    # Write launch script (to be able to run it separately with `pairrunner.py`)
                    f_name = os.path.join(workdir, "simulation.batch")
                    self.write_batch_sript(f_name, templ, cpu_time)
                    all_sims.append((pre_gid, post_gid, freq, dt, f_name))
        sim_idx = pd.DataFrame(all_sims, columns=["pregid", "postgid", "frequency", "dt", "path"])
        sim_idx.to_csv(os.path.join(basedir, "index_%s.csv" % self.label), index=False)


class SimWriter(Config):
    """Class to setup single cell simulations for finding C_pre and C_post for all synapses"""
    def write_batch_sript(self, f_name, templ, gid, cpu_time, qos):
        """Writes single cell batch script"""
        with open(f_name, "w+", encoding="latin1") as f:
            f.write(templ.format(name="plast_%i" % gid, cpu_time=cpu_time, qos=qos, log=gid,
                                 env=self.env, run=self.run, args="%s %s" % (self._config_path, gid)))

    def write_sim_files(self):
        """Writes simple `simulation_config.json` used by `bluecellulab` and batch scripts for single cell sims"""
        # create and write simple simulation config
        pathlib.Path(self.sims_dir).mkdir(exist_ok=True)
        sim_config = {"run": {"dt": 0.025, "tstop": 3000.0, "random_seed": 12345},
                      "network": self.circuit_config,
                      "node_sets_file": self.node_set,
                      "node_set": self.target,
                      "output": {"output_dir": "out"},
                      "connection_overrides": [{"name": "plasticity", "source": self.target, "target": self.target,
                                               "modoverride": "GluSynapse", "weight": 1.0}]}
        with open(self.sim_config, "w", encoding="utf-8") as f:
            json.dump(sim_config, f, indent=4)

        # create folders for batch scripts and output csv files
        sbatch_dir = os.path.join(self.sims_dir, "sbatch")
        pathlib.Path(sbatch_dir).mkdir(exist_ok=True)
        pathlib.Path(os.path.join(self.sims_dir, "out")).mkdir(exist_ok=True)
        # get all EXC gids and write sbatch scripts for all of them
        c = Circuit(self.circuit_config)
        df = c.nodes[self.node_pop].get(self.target, "synapse_class")
        gids = df.loc[df == "EXC"].index.to_numpy()  # just to make sure
        with open(os.path.join("templates", "simulation.batch.tmpl"), "r") as f:
            templ = f.read()
        f_names = []
        for gid in tqdm(gids, desc="Writing batch scripts for every EXC gid", miniters=len(gids)/100):
            f_name = os.path.join(sbatch_dir, "sim_%i.batch" % gid)
            f_names.append(f_name)
            n_afferents = len(np.intersect1d(c.edges[self.edge_pop].afferent_nodes(gid), gids))
            cpu_time, qos = get_cpu_time(n_afferents)
            self.write_batch_sript(f_name, templ, gid, cpu_time, qos)
        # write master launch scripts in batches of 5k
        idx = np.arange(0, len(f_names), 5000)
        idx = np.append(idx, len(f_names))
        for i, (start, end) in enumerate(zip(idx[:-1], idx[1:])):
            with open(os.path.join(sbatch_dir, "launch_batch%i.sh" % i), "w") as f:
                for f_name in f_names[start:end]:
                    f.write("sbatch %s\n" % f_name)

    def relaunch_failed_jobs(self, error, verbose=False):
        """Checks output files and if they aren't presents checks logs for specific `error`
        and creates master launch script to relaunch all failed jobs"""
        c = Circuit(self.circuit_config)
        df = c.nodes[self.node_pop].get(self.target, "synapse_class")
        gids = df.loc[df == "EXC"].index.to_numpy()  # just to make sure
        f_names = []
        for gid in tqdm(gids, desc="Checking log files", miniters=len(gids)/100):
            if not os.path.isfile(os.path.join(self.sims_dir, "out", "%i.csv" % gid)):
                f_name = os.path.join(self.sims_dir, "sbatch", "sim_%i.log" % gid)
                if os.path.isfile(f_name):
                    if verbose:
                        print(f_name)
                    with open(f_name, "r") as f:
                        if error in f.readlines()[-1]:
                            f_names.append(os.path.join(self.sims_dir, "sbatch", "sim_%i.batch" % gid))
        if len(f_names):
            with open(os.path.join(self.sims_dir, "sbatch", "relaunch_failed.sh"), "w") as f:
                for f_name in f_names:
                    f.write("sbatch %s\n" % f_name)
            if verbose:
                print("Generated relaunch_failed.sh master launch script with %i jobs" % len(f_names))

    def check_failed_thresholds(self):
        """Check log files and returns statistics about failed threshold calibrations (for L6 PCs)"""
        c = Circuit(self.circuit_config)
        df = c.nodes[self.node_pop].get(self.target, ["synapse_class", "layer", "mtype"])
        df = df.loc[(df["synapse_class"] == "EXC") & (df["layer"] == "6")]
        gids, mtypes = df.index.to_numpy(), df["mtype"].to_numpy()
        not_defined_ths = {}
        for gid, mtype in tqdm(zip(gids, mtypes), total=len(gids),
                               desc="Checking log files", miniters=len(gids) / 100):
            f_name = os.path.join(self.sims_dir, "sbatch", "sim_%i.log" % gid)
            with open(f_name, "r") as f:
                if "setting negative thresholds" in f.readlines()[-3]:
                    if mtype in not_defined_ths:
                        not_defined_ths[mtype] += 1
                    else:
                        not_defined_ths[mtype] = 1
        unique_mtypes, counts = np.unique(mtypes, return_counts=True)
        for mtype, count in not_defined_ths.items():
            n = counts[unique_mtypes == mtype][0]
            print("For %s: %i gids (%.2f%% of total) couldn't be calibrated" % (mtype, count, (count/n) * 100))


if __name__ == "__main__":
    writer = OptSimWriter("../configs/L5TTPC_L5TTPC.yaml")
    pairs = writer.find_pairs()
    writer.write_sim_files(pairs)
    writer = OptSimWriter("../configs/L23PC_L5TTPC.yaml")
    pairs = writer.find_pairs()
    writer.write_sim_files(pairs)
    '''
    writer = SimWriter("../configs/Zenodo_O1.yaml")
    writer.write_sim_files()
    # writer.relaunch_failed_jobs("slurmstepd:", True)
    # writer.check_failed_thresholds()
    '''

