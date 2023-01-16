"""
Machine Learning Data Generator
last modified: András Ecker 01.2023
"""

import os
from tqdm import tqdm
from cached_property import cached_property
import numpy as np
import pandas as pd
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell, Synapse

from plastyfire.config import Config

# parameters to use for machine learning (the rest is either correlated with these, or not important)
USECOLS = ["syn_id", "gmax_p_AMPA", "gmax_NMDA", "volume_CR", "loc", "theta_d", "theta_p"]
DTYPES = {"syn_id": np.int64, "gmax_p_AMPA": np.float32, "gmax_NMDA": np.float32, "volume_CR": np.float32,
          "loc": str, "theta_d": np.float32, "theta_p": np.float32}


def _load_csvs(gids, mtypes, sims_dir):
    """Loads in saved results from all single cell simulations"""
    dfs = []
    for gid, mtype in tqdm(zip(gids, mtypes), total=len(gids),
                           desc="Loading saved results", miniters=len(gids) / 100):
        f_name = os.path.join(sims_dir, "out", "%i.csv" % gid)
        df = pd.read_csv(f_name, usecols=USECOLS, index_col=0, dtype=DTYPES)
        df["post_mtype"] = mtype
        dfs.append(df)
    return pd.concat(dfs)


class MLDataGenerator(Config):
    """Class to generate dataset for training ML algorithms to predict plasticity thresholds"""

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

    def merge_csvs(self):
        """Loads in saved results and after some preprocessing
        concatenates them to a big DataFrame to be used for machine learning"""
        c = Circuit(self.bc_path)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        mtypes = c.cells.get(gids, Cell.MTYPE).to_numpy()
        df = _load_csvs(gids, mtypes, self.sims_dir)
        # filter out SS to SS synapses (those won't be plastic - see Chindemi et al. 2020, bioRxiv)
        ss_gids = gids[mtypes == "L4_SSC"]
        ss_syn_idx = c.connectome.pathway_synapses(ss_gids, ss_gids)
        df.drop(ss_syn_idx, inplace=True)
        # drop rows where depression th. is higher then potentiation th. (and where both are -1)
        df.drop(df.query("theta_d >= theta_p").index, inplace=True)
        # add c_pre and c_post
        cond = [(df["loc"] == "apical"), (df["loc"] == "basal")]
        c_pres = [(self.inv_params["a20"] * df["theta_d"] + self.inv_params["a21"] * df["theta_p"]),
                  (self.inv_params["a00"] * df["theta_d"] + self.inv_params["a01"] * df["theta_p"])]
        c_posts = [(self.inv_params["a30"] * df["theta_d"] + self.inv_params["a31"] * df["theta_p"]),
                   (self.inv_params["a10"] * df["theta_d"] + self.inv_params["a11"] * df["theta_p"])]
        df["c_pre"] = np.select(cond, c_pres)
        df["c_post"] = np.select(cond, c_posts)
        df.to_pickle(self.out_fname)
        print("Dataset of %.2f million samples saved to: %s" % (len(df)/1e6, self.out_fname))


if __name__ == "__main__":
    gen = MLDataGenerator("/gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/configs/hexO1_v7.yaml")
    gen.merge_csvs()


