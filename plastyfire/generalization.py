# -*- coding: utf-8 -*-
"""
...
last modified: András Ecker 01.2021
"""

import os
import yaml
import pathlib
from cached_property import cached_property
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell
from plastyfire.bcwriter import BCWriter
from plastyfire.epg import ParamsGenerator


class SonataWriter(object):
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
        return self.config["circuit_path"]

    @property
    def target(self):
        return self.config["target"]

    @property
    def extra_recipe_path(self):
        return self.config["extra_recipe_path"]

    @property
    def base_dir(self):
        return self.config["base_dir"]

    @property
    def fit_params(self):
        return self.config["fit_params"]

    @cached_property
    def circuit(self):
        return Circuit(self.circuit_path)

    @cached_property
    def valid_gids(self):
        return self.circuit.cells.ids({"target": self.target, Cell.SYNAPSE_CLASS: "EXC"})

    def write_sim_files(self):
        """Writes simple BlueConfig used by BGLibPy"""
        pathlib.Path(self.base_dir).mkdir(exist_ok=True)
        target_fname = os.path.join(self.base_dir, "user.target")
        # Write empty user.target (just because there has to be one)
        with open(target_fname, "w") as f:
            f.write("")
        # Write BlueConfig
        bcw = BCWriter(self.circuit_path, duration=2000, target=self.target, target_file=target_fname, base_seed=1909)
        bcw.write(self.base_dir)

    def debug(self):
        """Stupid function for debugging purpose"""
        self.write_sim_files()
        pgen = ParamsGenerator(self.circuit_path, self.extra_recipe_path)
        default_params = pgen.generate_params(pregid=147984, postgid=147748)


if __name__ == "__main__":

    writer = SonataWriter("../configs/O1_v6.yaml")
    writer.debug()


