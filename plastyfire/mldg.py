# -*- coding: utf-8 -*-
"""
Machine learning data generator
last modified: András Ecker 04.2021
"""

import os
import yaml
from tqdm import tqdm
from cached_property import cached_property
import numpy as np
import pandas as pd
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell


# parameters to use for machine learning (the rest is either correlated with these, or not important)
usecols = ["pre_gid", "syn_id", "gmax0_AMPA", "gmax_p_AMPA", "gmax_NMDA", "volume_CR",
           "loc", "dist", "inp_imp", "theta_d", "theta_p"]


class MLDataGenerator(object):
    """Class to generate dataset for training ML algorithms to predict plasticity thresholds"""

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
    def sims_dir(self):
        return self.config["sims_dir"]

    @property
    def out_dir(self):
        return self.config["out_dir"]

    @property
    def fit_params(self):
        return self.config["fit_params"]

    @property
    def bc(self):
        return os.path.join(self.sims_dir, "BlueConfig")

    @property
    def out_fname(self):
        return os.path.join(self.out_dir, "mldata.pkl")

    @cached_property
    def inv_params(self):
        """Inverted fit params to get c_pre and c_post from theta_d and theta_p"""
        basal = np.array([[self.fit_params["a00"], self.fit_params["a01"]],
                          [self.fit_params["a10"], self.fit_params["a11"]]])
        inv_basal = np.linalg.inv(basal)
        apical = np.array([[self.fit_params["a20"], self.fit_params["a21"]],
                           [self.fit_params["a30"], self.fit_params["a31"]]])
        inv_apical = np.linalg.inv(apical)
        return {"a00": inv_basal[0, 0], "a01": inv_basal[0, 1], "a10": inv_basal[1, 0], "a11": inv_basal[1, 1],
                "a20": inv_apical[0, 0], "a21": inv_apical[0, 1], "a30": inv_apical[1, 0], "a31": inv_apical[1, 1]}

    def merge_csvs(self, save):
        """Loads in saved results from all (finished) sims and after some preprocessing
        concatenates them to a big DataFrame to be used for machine learning"""
        c = Circuit(self.bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        mtypes = c.cells.get(gids, Cell.MTYPE).to_numpy()
        ss_gids = gids[mtypes == "L4_SSC"]
        n = 0
        dfs = []
        for gid, mtype in tqdm(zip(gids, mtypes), total=len(gids),
                               desc="Loading saved results", miniters=len(gids)/100):
            f_name = os.path.join(self.sims_dir, "out", "%i.csv" % gid)
            if os.path.isfile(f_name):
                n += 1
                df = pd.read_csv(f_name, usecols=usecols, index_col=[0, 1])
                df["post_mtype"] = mtype
                # filter out SS to SS synapses (those won't be plastic - see Chindemi et al. 2020, bioRxiv)
                if gid in ss_gids:
                    pre_gids = df.index.get_level_values("pre_gid").to_numpy()
                    df = df.drop(np.intersect1d(pre_gids, ss_gids), level="pre_gid")
                # only add it to the training data if thresholds were found (default value if not found is -1)
                if df["theta_d"].values[0] != -1.:
                    dfs.append(df.droplevel("pre_gid"))
        print("Loaded data from %i/%i (%.2f%% of) cells." % (n, len(gids), (100 * n) / len(gids)))
        df = pd.concat(dfs)
        # some basic feature engineering and cleaning
        df["gsynSRSF"] = df["gmax_NMDA"] / df["gmax0_AMPA"]  # add NMDA/AMPA ratio
        df = df.drop("gmax0_AMPA", axis=1)  # drop baseline AMPA cond. as only the pot. is used in the c_pre sims.
        # drop rows where depression th. is higher then potentiation th.
        df = df.drop(df.query("theta_d >= theta_p").index)
        # add c_pre and c_post
        cond = [(df["loc"] == "apical"), (df["loc"] == "basal")]
        c_pres = [(self.inv_params["a20"] * df["theta_d"] + self.inv_params["a21"] * df["theta_p"]),
                  (self.inv_params["a00"] * df["theta_d"] + self.inv_params["a01"] * df["theta_p"])]
        c_posts = [(self.inv_params["a30"] * df["theta_d"] + self.inv_params["a31"] * df["theta_p"]),
                   (self.inv_params["a10"] * df["theta_d"] + self.inv_params["a11"] * df["theta_p"])]
        df["c_pre"] = np.select(cond, c_pres)
        df["c_post"] = np.select(cond, c_posts)
        if save:
            df.to_pickle(self.out_fname)
            print("Dataset of %.2f million samples saved to: %s" % (len(df)/1e6, self.out_fname))
        else:
            print("Dataset size: %.2f million samples" % (len(df)/1e6))
        return df


if __name__ == "__main__":

    gen = MLDataGenerator("../configs/hexO1_v7.yaml")
    data = gen.merge_csvs(save=True)

