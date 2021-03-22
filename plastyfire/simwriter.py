# -*- coding: utf-8 -*-
"""
Writes sbatch scripts for every gid (given a circuit and a target in the config file)
last modified: András Ecker 03.2021
"""

import os
import yaml
import time
import pathlib
import shutil
from tqdm import tqdm
import numpy as np
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell


class SimWriter(object):
    """Small class to setup single cell simulations"""

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self.config_path = config_path
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

    @property
    def circuit_path(self):
        return self.config["circuit"]["path"]

    @property
    def user_target(self):
        return self.config["circuit"]["user.target"]

    @property
    def target(self):
        return self.config["circuit"]["target"]

    @property
    def sims_dir(self):
        return self.config["sims_dir"]

    def write_batch_sript(self, f_name, templ, gid, cpu_time, qos):
        """Writes single cell batch script"""
        with open(f_name, "w") as f:
            f.write(templ.format(name="plast_%i" % gid, cpu_time=cpu_time, qos=qos,
                                 config=self.config_path, gid=gid))

    def write_sim_files(self):
        """Writes simple BlueConfig used by BGLibPy (and for gid queries) and batch scripts for single cell sims"""
        from plastyfire.bcwriter import BCWriter

        # write BlueConfig and user.target
        pathlib.Path(self.sims_dir).mkdir(exist_ok=True)
        target_fname = os.path.join(self.sims_dir, "user.target")
        shutil.copyfile(self.user_target, target_fname)
        bcw = BCWriter(self.circuit_path, duration=3000, target=self.target, target_file=target_fname, base_seed=12345)
        bc = bcw.write(self.sims_dir)
        # create folders for batch scripts and output csv files
        sbatch_dir = os.path.join(self.sims_dir, "sbatch")
        pathlib.Path(sbatch_dir).mkdir(exist_ok=True)
        pathlib.Path(os.path.join(self.sims_dir, "out")).mkdir(exist_ok=True)

        # get all EXC gids and write sbatch scripts for all of them
        c = Circuit(bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        with open("templates/simulation.batch.tmpl", "r") as f:
            templ = f.read()
        f_names = []
        for gid in tqdm(gids, desc="Writing batch scripts for every EXC gid", miniters=len(gids)/100):
            f_name = os.path.join(sbatch_dir, "sim_%i.batch" % gid)
            f_names.append(f_name)
            n_afferents = len(np.intersect1d(c.connectome.afferent_gids(gid), gids))
            # CPU time heuristics: 5 min setup and stim. calc., 1.5 min sim per connection + 10 min just to make sure
            cpu_time_sec = (5 + n_afferents * 1.5 + 10) * 60
            cpu_time = time.strftime("%H:%M:%S", time.gmtime(cpu_time_sec))
            # in SSCX hex_O1 there aren't EXC cells with >2000 afferents, so it won't go over 1 day,
            # but keeping this here for the future...
            qos = "#SBATCH --qos=longjob" if int(cpu_time.split(':')[0]) > 24 else ""
            self.write_batch_sript(f_name, templ, gid, cpu_time, qos)
        # write master launch scripts in batches of 1k
        idx = np.arange(0, len(f_names), 1000)
        idx = np.append(idx, len(f_names))
        for i, (start, end) in enumerate(zip(idx[:-1], idx[1:])):
            with open(os.path.join(sbatch_dir, "launch_batch%i.sh" % i), "w") as f:
                for f_name in f_names[start:end]:
                    f.write("sbatch %s\n" % f_name)


if __name__ == "__main__":

    config = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/configs/hexO1_v7.yaml"
    writer = SimWriter(config)
    writer.write_sim_files()


