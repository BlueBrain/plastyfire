"""
Runscript for single cell simulations that find the depression and potentiation thresholds used in GluSynapse
The script has 2 modes based on the `use_extra_recipe` flag in the main YAML config:
- If set to 1, and a recipe with pathway specific means and stds. is provided,
  it generates base synaptic parameters with correlations (e.g. between: spine volume, synaptic strength,
  and release probability - see `plasyfire.epg`) and by passing these to `bluecellulab`
  it overrides the parameters stored in the SONATA edge file before running the sims,
  i.e., this mode is used during model development
- If set to 0, then it just reads the parameters from SONATA sets up single cell sims. and runs them,
  i.e., this mode is used for validation (or after modifying `plastyfire.simulator` it could e.g. do extra recording)
last modified: András Ecker 02.2024
"""

import os
import gc
import time
import argparse
import logging
import numpy as np
import pandas as pd
from neurom import NeuriteType
from bluepysnap import Circuit

from plastyfire.config import Config


logging.basicConfig(level=logging.INFO)
L = logging.getLogger("thresholdfinder")
EXTRA_RECIPE_PATH = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/biodata/recipe.csv"
# parameters to store (with names corresponding to GluSynapse.mod)
PARAMS = ["Use0_TM", "Dep_TM", "Fac_TM", "Nrrp_TM", "gmax0_AMPA", "volume_CR",  # base 5 params + volume for GluSynapse
          "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA", "gmax_p_AMPA",  # generated (see `plastifyre.epg`)
          "gmax_NMDA", "loc",  # not used in further steps but saved anyways for ML stuff
          "theta_d", "theta_p"]  # calculated from c_pre and c_post + optimized GluSynapse params
SONATA_PARAMS = ["@source_node", "u_syn", "depression_time", "facilitation_time", "n_rrp_vesicles", "conductance",
                 "volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA", "gmax_p_AMPA",
                 "conductance_scale_factor", "afferent_section_type"]
PARAM_MAP = {"u_syn": "Use0_TM", "depression_time": "Dep_TM", "facilitation_time": "Fac_TM",
             "n_rrp_vesicles": "Nrrp_TM", "conductance": "gmax0_AMPA"}
BRANCH_TYPE_OFFSET = 1


def init_df(edges, pre_gids, post_gid):
    """Initializes an empty MultiIndex DataFrame (to be filled with values in `ThresholdFinder.run()`)"""
    tuples = []
    for pre_gid in pre_gids:
        syn_idx = edges.pair_edges(pre_gid, post_gid)
        for syn_id in syn_idx:
            tuples.append((pre_gid, syn_id))
    mi = pd.MultiIndex.from_tuples(tuples, names=["pre_gid", "syn_id"])
    df_tmp = mi.to_frame()
    df = df_tmp.drop(columns=["pre_gid", "syn_id"])  # stupid pandas ...
    for param in PARAMS:
        if param not in ["Nrrp_TM", "rho0_GB", "loc", "theta_d", "theta_p"]:
            fill_value = 0.0
        elif param in ["Nrrp_TM", "rho0_GB"]:
            fill_value = 0
        elif param == "loc":
            fill_value = ""
        elif param in ["theta_d", "theta_p"]:
            fill_value = -1.0
        df[param] = fill_value
    return df


def read_sonata_params(edges, pre_gids, post_gid):
    """Read parameters from SONATA edge file and get them to follow the naming convention of `plastyfire.epg`"""
    syn_df = edges.pathway_edges(pre_gids, post_gid, SONATA_PARAMS)
    syn_df = syn_df.rename(columns=PARAM_MAP)
    syn_df["gmax_NMDA"] = syn_df["gmax0_AMPA"] * syn_df["conductance_scale_factor"]
    cond = [(syn_df["afferent_section_type"] + BRANCH_TYPE_OFFSET == NeuriteType.basal_dendrite),
            (syn_df["afferent_section_type"] + BRANCH_TYPE_OFFSET == NeuriteType.apical_dendrite)]
    syn_df["loc"] = np.select(cond, ["basal", "apical"])
    return syn_df[["@source_node"] + PARAMS[:-2]]


def _synid_to_dfid(df, syn_id):
    """Gets df MultiIndex from synapse id (first index is pre_gid and it's only stored for better readability)"""
    return df[df.index.get_level_values(1) == syn_id].index


def store_params(df, conn_params):
    """Adds (generated and calculated) connection parameters to MultiIndex DataFrame"""
    for syn_id, syn_params in conn_params.items():
        df_id = _synid_to_dfid(df, syn_id)
        for param, val in syn_params.items():
            df.loc[df_id, param] = val


class ThresholdFinder(Config):
    """Class (to store info about and) to run single cell simulations in bluecellulab"""

    def run(self, post_gid):
        """Finds c_pre and c_post (see `plastyfire.simulator`) for all afferents of `post_gid`,
        calculates thresholds used in GluSynapse, stores them in a MultiIndex DataFrame, and saves them to csv."""
        from plastyfire.epg import ParamsGenerator
        from plastyfire.simulator import spike_threshold_finder, c_pre_finder, c_post_finder

        # get afferent gids (of `post_gid` within the given target)
        c = Circuit(self.circuit_config)
        df = c.nodes[self.node_pop].get(self.target, "synapse_class")
        gids = df.loc[df == "EXC"].index.to_numpy()  # just to make sure
        edges = c.edges[self.edge_pop]
        pre_gids = np.intersect1d(edges.afferent_nodes(post_gid), gids)
        df = init_df(edges, pre_gids, post_gid)
        # init extra parameter generator or if recipe is missing read params from edge file
        if self.use_extra_recipe:
            pgen = ParamsGenerator(c, self.node_pop, self.edge_pop, EXTRA_RECIPE_PATH)
        else:
            syn_df = read_sonata_params(edges, pre_gids, post_gid)
        # first test if gid can be stimulated to elicit a single spike
        L.info("Finding stimulus for gid %i (%s)" % (post_gid, c.nodes[self.node_pop].get(post_gid, "mtype")))
        t1 = time.time()
        for pulse_width in [1.5, 3, 5]:
            simres = spike_threshold_finder(self.sim_config, post_gid, 1, 0.1, pulse_width, 1000., 0.05, 5., 100,
                                            self.node_pop, False)
            if simres is not None:
                break
        # if gid can be stimulated to elicit a single spike find c_pre and c_post and calculate thersholds
        if simres is not None:
            stimulus = {"nspikes": 1, "freq": 0.1, "width": simres["width"],
                        "offset": simres["offset"], "amp": simres["amp"]}
            L.info("%.2f nA (%.1f s long) stimulus found in %s" % (stimulus["amp"], stimulus["width"],
                   time.strftime("%M:%S", time.gmtime(time.time() - t1))))
            for i, pre_gid in enumerate(pre_gids):
                L.info("Finding c_pre and c_post for %i -> %i (%i/%i)" % (pre_gid, post_gid, i+1, len(pre_gids)))
                if self.use_extra_recipe:
                    conn_params = pgen.generate_params(pre_gid, post_gid)
                else:
                    # in case these are read from the SONATA file it's not necessary to pass these parameters
                    # to the two functions below... but since we're saving them and therefore have to get them why not...
                    conn_params = syn_df.loc[syn_df["@source_node"] == pre_gid].drop(columns=["@source_node"]).to_dict(orient="index")
                c_pre = c_pre_finder(self.sim_config, self.fit_params, conn_params, pre_gid, post_gid,
                                     self.node_pop, self.edge_pop, False)
                c_post = c_post_finder(self.sim_config, self.fit_params, conn_params, pre_gid, post_gid, stimulus,
                                       self.node_pop, self.edge_pop, False)
                if c_post is not None:
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
                gc.collect()
        else:  # if not, keep negative threshols as initialized in the DataFrame (no plasticity)
            L.info("Stimulus couldn't be calibrated, skipping simulations, setting negative thresholds.")
            if self.use_extra_recipe:
                for pre_gid in pre_gids:
                    conn_params = pgen.generate_params(pre_gid, post_gid)
                    store_params(df, conn_params)
            else:
                syn_df.index = pd.MultiIndex.from_arrays([syn_df["@source_node"].to_numpy(), syn_df.index.to_numpy()])
                df.loc[:, PARAMS[:-2]] = syn_df[PARAMS[:-2]]
        # save results to csv
        L.info("Saving results to out/%i.csv" % post_gid)
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

