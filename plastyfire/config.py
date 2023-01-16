"""
Config file class
author: András Ecker; last update 01.2023
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
    def circuit_path(self):
        return self.config["circuit"]["path"]

    @property
    def user_target(self):
        return self.config["circuit"]["user.target"]

    @property
    def target(self):
        return self.config["circuit"]["target"]

    @property
    def extra_recipe_path(self):
        return self.config["extra_recipe_path"]

    @property
    def sims_dir(self):
        return self.config["sims_dir"]

    @property
    def bc_path(self):
        return os.path.join(self.sims_dir, "BlueConfig")

    @property
    def out_dir(self):
        return self.config["out_dir"]

    @property
    def fit_params(self):
        return self.config["fit_params"]
