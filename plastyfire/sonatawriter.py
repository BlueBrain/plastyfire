# -*- coding: utf-8 -*-
"""
...
last modified: András Ecker 02.2021
"""

import os
import yaml
import pathlib
from cached_property import cached_property
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell
from plastyfire.bcwriter import BCWriter
from plastyfire.epg import ParamsGenerator
from plastyfire.simulator import c_pre_finder, c_post_finder


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

    def debug(self, pre_gid=147984, post_gid=147748):
        """Stupid function for debugging purpose"""
        self.write_sim_files()
        pgen = ParamsGenerator(self.circuit_path, self.extra_recipe_path)
        default_params = pgen.generate_params(pre_gid, post_gid)

        try:
            cpre = c_pre_finder(self.base_dir, self.fit_params, default_params, pre_gid, post_gid,
                                fixhp=True, invivo=False)
            _, cpost = c_post_finder(self.base_dir, self.fit_params, default_params, pre_gid, post_gid,
                                     stimulus=None, fixhp=True, invivo=False)
        except RuntimeError:
            # Something went wrong, set negative threshols (no plasticity)
            for synapse_id, synapse_params in default_params.items():
                synapse_params["theta_d"] = -1
                synapse_params["theta_p"] = -1
        else:
            # Compute thresholds (this shouldn't be here, but for now ok)
            for synapse_id, synapse_params in default_params.items():
                if synapse_params["loc"] == "basal":
                    synapse_params["theta_d"] = self.fit_params["a00"] * cpre[synapse_id] +\
                                                self.fit_params["a01"] * cpost[synapse_id]
                    synapse_params["theta_p"] = self.fit_params["a10"] * cpre[synapse_id] +\
                                                self.fit_params["a11"] * cpost[synapse_id]
                elif synapse_params["loc"] == "apical":
                    synapse_params["theta_d"] = self.fit_params["a20"] * cpre[synapse_id] +\
                                                self.fit_params["a21"] * cpost[synapse_id]
                    synapse_params["theta_p"] = self.fit_params["a30"] * cpre[synapse_id] +\
                                                self.fit_params["a31"] * cpost[synapse_id]
                else:
                    raise ValueError("Unknown location")


if __name__ == "__main__":

    writer = SonataWriter("../configs/O1_v6.yaml")
    writer.debug()


