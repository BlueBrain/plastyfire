"""
Config file class
author: András Ecker; last update 02.2024
"""

import os
import yaml


class Config(object):
    """Class to store config parameters about the simulations"""

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self._config_path = config_path
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

    @property
    def circuit_config(self):
        return self.config["circuit"]["config"]

    @property
    def node_set(self):
        return self.config["circuit"]["node_set"]

    @property
    def target(self):
        return self.config["circuit"]["target"]

    @property
    def node_pop(self):
        return self.config["circuit"]["node_pop"]

    @property
    def edge_pop(self):
        return self.config["circuit"]["edge_pop"]

    @property
    def extra_recipe_path(self):
        if "extra_recipe_path" in self.config:
            return self.config["extra_recipe_path"]
        else:
            return None

    @property
    def sims_dir(self):
        return self.config["sims_dir"]

    @property
    def sim_config(self):
        return os.path.join(self.sims_dir, "simulation_config.json")

    @property
    def out_dir(self):
        return self.config["out_dir"]

    @property
    def fit_params(self):
        return self.config["fit_params"]
