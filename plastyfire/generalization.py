# -*- coding: utf-8 -*-
"""
...
last modified: András Ecker 01.2021
"""

import os
import yaml
from cached_property import cached_property
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
    def fit_params(self):
        return self.config["fit_params"]

    @cached_property
    def circuit(self):
        from bluepy.v2 import Circuit
        return Circuit(self.circuit_path)

    @cached_property
    def valid_gids(self):
        from bluepy.v2.enums import Cell
        return self.circuit.cells.ids({"target": self.target, Cell.SYNAPSE_CLASS: "EXC"})

    def debug(self):
        """Stupid function for debugging purpose"""
        pgen = ParamsGenerator(self.circuit_path, self.extra_recipe_path)
        default_params = pgen.generate_params(pregid=147984, postgid=147748)


if __name__ == "__main__":

    writer = SonataWriter("../configs/O1_v6.yaml")
    writer.debug()


