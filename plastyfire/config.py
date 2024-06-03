"""
Config file class
author: András Ecker; last update 05.2024
"""

import os
import yaml


class BaseConfig(object):
    """Class to store common config parameters about the simulations"""
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
    def sims_dir(self):
        return self.config["sims_dir"]

    @property
    def env(self):
        return self.config["simulator"]["env"]

    @property
    def run(self):
        return self.config["simulator"]["run"]


class Config(BaseConfig):
    """Class to store config parameters about finding C_pre and C_post for all synapses"""
    @property
    def sim_config(self):
        return os.path.join(self.sims_dir, "simulation_config.json")

    @property
    def use_extra_recipe(self):
        return self.config["use_extra_recipe"]

    @property
    def out_dir(self):
        return self.config["out_dir"]

    @property
    def fit_params(self):
        return self.config["fit_params"]


class OptConfig(BaseConfig):
    """Class to store config parameters about the optimization of model parameters"""
    @property
    def label(self):
        return self.config["label"]

    @property
    def seed(self):
        return self.config["seed"]

    @property
    def npairs(self):
        return self.config["npairs"]

    @property
    def out_dir(self):
        return os.path.join(self.sims_dir, "fitting", "n%i" % self.npairs,
                            "seed%i" % self.seed, self.label, "simulations")

    @property
    def pre_mtype(self):
        return self.config["pregid_conf"]["mtype"]

    @property
    def post_mtype(self):
        return self.config["postgid_conf"]["mtype"]

    @property
    def max_dist(self):
        return self.config["geom_cons"]["max_dist"]

    @property
    def freq(self):
        return self.config["stimulus"]["freq"]

    @property
    def amp(self):
        return None if self.config["stimulus"]["amp"] == "find" else self.config["stimulus"]["amp"]

    @property
    def amp_min(self):
        return self.config["stimulus"]["amp_min"]

    @property
    def amp_max(self):
        return self.config["stimulus"]["amp_max"]

    @property
    def amp_lev(self):
        return self.config["stimulus"]["amp_lev"]

    @property
    def nspikes(self):
        return self.config["stimulus"]["nspikes"]

    @property
    def dt(self):
        return self.config["stimulus"]["dt"]

    @property
    def width(self):
        return self.config["stimulus"]["width"]

    @property
    def dt(self):
        return self.config["stimulus"]["dt"]

    @property
    def width(self):
        return self.config["stimulus"]["width"]

    @property
    def T(self):
        return self.config["stimulus"]["T"]

    @property
    def offset(self):
        return self.config["stimulus"]["offset"]

    @property
    def nreps(self):
        return self.config["stimulus"]["nreps"]

    @property
    def C01_duration(self):
        return self.config["stimulus"]["C01_duration"]

    @property
    def C02_duration(self):
        return self.config["stimulus"]["C02_duration"]

    @property
    def C01_T(self):
        return self.config["stimulus"]["C01_T"] if "C01_T" in self.config["stimulus"] else self.T

    @property
    def C02_T(self):
        return self.config["stimulus"]["C02_T"] if "C02_T" in self.config["stimulus"] else self.T

    @property
    def fastforward(self):
        return self.config["simulator"]["fastforward"] if "fastforward" in self.config["simulator"] else None
