# -*- coding: utf-8 -*-
"""

last modified: András Ecker 03.2021
"""

import os
import yaml
import logging
from cached_property import cached_property
import numpy as np
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell


class ThresholdFinder(object):
    """ """

    def __init__(self, config_path):
        """YAML config file based constructor"""
        self._config_path = config_path
        with open(config_path, "r") as f:
            self._config = yaml.load(f, Loader=yaml.SafeLoader)

    @property
    def config(self):
        return self._config

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
    def fit_params(self):
        return self.config["fit_params"]

    @cached_property
    def bc(self):
        return os.path.join(self.sims_dir, "BlueConfig")

    def run(self, post_gid):
        """Finds c_pre and c_post (see `plastyfire.simulator`) for all afferents of `post_gid`,
        calculates thresholds used in GluSynapse and ..."""
        from plastyfire.epg import ParamsGenerator
        from plastyfire.simulator import spike_threshold_finder, c_pre_finder, c_post_finder

        # get afferent gids (of `post_gid` within the given target)
        c = Circuit(self.bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        afferent_gids = np.intersect1d(c.connectome.afferent_gids(post_gid), gids)
        # init Glusynapse parameter generator (with correlations)
        pgen = ParamsGenerator(c, self.extra_recipe_path)

        try:  # first test if gid can be stimulated to elicit a single spike
            for pulse_width in [1.5, 3]:
                simres = spike_threshold_finder(self.bc, post_gid, 1, 0.1, pulse_width, 1000., 0.05, 5., 100, True)
                if simres is not None:
                    break
            stimulus = {"nspikes": 1, "freq": 0.1, "width": simres["width"], "offset": 1000., "amp": simres["amp"]}
        except RuntimeError:  # if not, set negative threshols (no plasticity)
            for pre_gid in afferent_gids:
                default_params = pgen.generate_params(pre_gid, post_gid)
                for synapse_id, synapse_params in default_params.items():
                    synapse_params["theta_d"] = -1.0
                    synapse_params["theta_p"] = -1.0
        else:  # if gid can be stimulated to elicit a single spike find c_pre and c_post and calc. thersholds
            for pre_gid in afferent_gids:
                default_params = pgen.generate_params(pre_gid, post_gid)
                cpre = c_pre_finder(self.bc, self.fit_params, default_params, pre_gid, post_gid, True)
                cpost = c_post_finder(self.bc, self.fit_params, default_params, pre_gid, post_gid, stimulus, True)
                for synapse_id, synapse_params in default_params.items():
                    if synapse_params["loc"] == "basal":
                        synapse_params["theta_d"] = self.fit_params["a00"] * cpre[synapse_id] + \
                                                    self.fit_params["a01"] * cpost[synapse_id]
                        synapse_params["theta_p"] = self.fit_params["a10"] * cpre[synapse_id] + \
                                                    self.fit_params["a11"] * cpost[synapse_id]
                    elif synapse_params["loc"] == "apical":
                        synapse_params["theta_d"] = self.fit_params["a20"] * cpre[synapse_id] + \
                                                    self.fit_params["a21"] * cpost[synapse_id]
                        synapse_params["theta_p"] = self.fit_params["a30"] * cpre[synapse_id] + \
                                                    self.fit_params["a31"] * cpost[synapse_id]
                    else:
                        raise ValueError("Unknown location")


if __name__ == "__main__":

    sim = ThresholdFinder(config_path)
    sim.run(post_gid)

    # debug: pre_gid=8706, post_gid=8473




