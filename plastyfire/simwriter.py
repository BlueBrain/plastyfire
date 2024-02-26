"""
Writes sbatch scripts for every gid (given a circuit and a target in the config file)
last modified: András Ecker 02.2024
"""

import os
import pathlib
import json
from tqdm import tqdm
import numpy as np
from bluepysnap import Circuit

from plastyfire.config import Config


CPU_TIME = 1.5  # heuristics: c_pre and c_post for a single connection takes ca. 1.3 minutes to simulate/calculate


def get_cpu_time(n_afferents):
    """CPU time heuristics: 5 min setup and stimulus calculation + 10 mins. extra just to make sure +
    gid specific simulation time, based on the number of its afferent gids"""
    cpu_time_sec = (15 + n_afferents * CPU_TIME) * 60
    h, m = np.divmod(cpu_time_sec, 3600)
    m, s = np.divmod(m, 60)
    cpu_time_str = "%.2i:%.2i:%.2i" % (h, m, s)
    qos = "#SBATCH --qos=longjob" if h >= 24 else ""
    return cpu_time_str, qos


class SimWriter(Config):
    """Small class to setup single cell simulations"""
    def write_batch_sript(self, f_name, templ, gid, cpu_time, qos):
        """Writes single cell batch script"""
        with open(f_name, "w+", encoding="latin1") as f:
            f.write(templ.format(name="plast_%i" % gid, cpu_time=cpu_time, qos=qos,
                                 config=self._config_path, gid=gid))

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
        with open(os.path.join(self.sims_dir, "simulation_config.json"), "w", encoding="utf-8") as f:
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
    config_path = "../configs/Zenodo_O1.yaml"
    writer = SimWriter(config_path)
    writer.write_sim_files()
    # writer.relaunch_failed_jobs("slurmstepd:", True)
    # writer.check_failed_thresholds()

