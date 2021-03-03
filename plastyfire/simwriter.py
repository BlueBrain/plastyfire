# -*- coding: utf-8 -*-
"""

last modified: András Ecker 03.2021
"""

import os
import yaml
import pathlib
import shutil
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell


class SimWriter(object):
    """"""

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self._config_path = config_path
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

    def write_sim_files(self):
        """Writes simple BlueConfig used by BGLibPy (and gid queries)"""
        from plastyfire.bcwriter import BCWriter
        pathlib.Path(self.sims_dir).mkdir(exist_ok=True)
        target_fname = os.path.join(self.sims_dir, "user.target")
        shutil.copyfile(self.user_target, target_fname)
        bcw = BCWriter(self.circuit_path, duration=2000, target=self.target, target_file=target_fname, base_seed=1909)
        bcw.write(self.sims_dir)

    def get_valid_gids(self):
        """Gets EXC gids within the specified target (`Circuit()` has to be initialized from a BlueConfig
        to get the extra target from user.target)"""
        c = Circuit(os.path.join(self.sims_dir, "BlueConfig"))
        return c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})


if __name__ == "__main__":

    writer = SimWriter("../configs/hexO1_v7.yaml")
    writer.write_sim_files()


