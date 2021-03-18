# -*- coding: utf-8 -*-
"""
Runscript for single cell simulations that find the depression and potentiation thresholds used in GluSynapse
last modified: András Ecker 03.2021
"""

import os
import time
import yaml
import argparse
import logging
from cached_property import cached_property
import numpy as np
import pandas as pd
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell


logging.basicConfig(level=logging.INFO)
L = logging.getLogger("thresholdfinder")

# parameters to store (with names corresponding to GluSynapse.mod)
params = ["Use0_TM", "Dep_TM", "Fac_TM", "Nrrp_TM", "gmax0_AMPA", "volume_CR",  # base 5 params + volume for GluSynapse
          "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA", "gmax_p_AMPA",  # generated (see `plastifyre.epg`)
          "gmax_NMDA", "loc",  # not used in further steps but saved anyways (for ML stuff?)
          "theta_d", "theta_p"]  # calculated from c_pre and c_post + optimized GluSynapse params


def init_df(c, pre_gids, post_gid):
    """Initializes an empty MultiIndex DataFrame (to be filled with values in `ThresholdFinder.run()`)"""
    tuples = []
    for pre_gid in pre_gids:
        syn_idx = c.connectome.pair_synapses(pre_gid, post_gid)
        for syn_id in syn_idx:
            tuples.append((pre_gid, syn_id))
    mi = pd.MultiIndex.from_tuples(tuples, names=["pre_gid", "syn_id"])
    df_tmp = mi.to_frame()
    df = df_tmp.drop(columns=["pre_gid", "syn_id"])  # stupid pandas ...
    for param in params:
        if param not in ["rho0_GB", "loc", "theta_d", "theta_p"]:
            fill_value = 0.0
        elif param == "rho0_GB":
            fill_value = 0
        elif param == "loc":
            fill_value = ""
        elif param in ["theta_d", "theta_p"]:
            fill_value = -1.0
        df[param] = fill_value
    return df


def _synid_to_dfid(df, syn_id):
    """Gets df MultiIndex from synapse id (first index is pre_gid and it's only stored for better readability)"""
    return df[df.index.get_level_values(1) == syn_id].index


def store_params(df, conn_params):
    """Adds (generated and calculated) connection parameters to MultiIndex DataFrame"""
    for syn_id, syn_params in conn_params.items():
        df_id = _synid_to_dfid(df, syn_id)
        for param, val in syn_params.items():
            df.loc[df_id, param] = val


class ThresholdFinder(object):
    """Class (to store info about and) to run single cell simulations in BGLibPy"""

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
        calculates thresholds used in GluSynapse, stores them in a MultiIndex DataFrame, and saves them to csv."""
        from plastyfire.epg import ParamsGenerator
        from plastyfire.simulator import spike_threshold_finder, c_pre_finder, c_post_finder

        # get afferent gids (of `post_gid` within the given target)
        c = Circuit(self.bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        pre_gids = np.intersect1d(c.connectome.afferent_gids(post_gid), gids).astype(np.int)
        # init DataFrame to store results
        df = init_df(c, pre_gids, post_gid)
        # init Glusynapse parameter generator (with correlations)
        pgen = ParamsGenerator(c, self.extra_recipe_path)

        try:  # first test if gid can be stimulated to elicit a single spike
            for pulse_width in [1.5, 3]:
                L.info("Finding stimulus for gid %i" % post_gid)
                simres = spike_threshold_finder(self.bc, post_gid, 1, 0.1, pulse_width, 1000., 0.05, 5., 100, True)
                if simres is not None:
                    break
            stimulus = {"nspikes": 1, "freq": 0.1, "width": simres["width"], "offset": 1000., "amp": simres["amp"]}
        except RuntimeError:  # if not, keep negative threshols as initialized in the DataFrame (no plasticity)
            L.info("Stimulus couldn't be calibrated, skipping simulations, setting negative threshold.")
            for pre_gid in pre_gids:
                conn_params = pgen.generate_params(pre_gid, post_gid)
                store_params(df, conn_params)
        else:  # if gid can be stimulated to elicit a single spike find c_pre and c_post and calc. thersholds
            for i, pre_gid in enumerate(pre_gids):
                L.info("Finding c_pre and c_post for %i -> %i (%i/%i)" % (pre_gid, post_gid, i+1, len(pre_gids)))
                conn_params = pgen.generate_params(pre_gid, post_gid)
                c_pre = c_pre_finder(self.bc, self.fit_params, conn_params, pre_gid, post_gid, True)
                c_post = c_post_finder(self.bc, self.fit_params, conn_params, pre_gid, post_gid, stimulus, True)
                for syn_id, syn_params in conn_params.items():
                    if syn_params["loc"] == "basal":
                        syn_params["theta_d"] = self.fit_params["a00"] * c_pre[syn_id] + \
                                                self.fit_params["a01"] * c_post[syn_id]
                        syn_params["theta_p"] = self.fit_params["a10"] * c_pre[syn_id] + \
                                                self.fit_params["a11"] * c_post[syn_id]
                    elif syn_params["loc"] == "apical":
                        syn_params["theta_d"] = self.fit_params["a20"] * c_pre[syn_id] + \
                                                self.fit_params["a21"] * c_post[syn_id]
                        syn_params["theta_p"] = self.fit_params["a30"] * c_pre[syn_id] + \
                                                self.fit_params["a31"] * c_post[syn_id]
                    else:
                        raise ValueError("Unknown location")
                store_params(df, conn_params)
        # save results to csv
        df.to_csv(os.path.join(self.sims_dir, "out", "%i.csv" % post_gid))


if __name__ == "__main__":

    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to YAML config file")
    parser.add_argument("post_gid", type=int, help="post_gid to simulate")
    args = parser.parse_args()

    sim = ThresholdFinder(args.config_path)
    start_time = time.time()
    sim.run(args.post_gid)
    L.info("Elapsed time: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))




