# -*- coding: utf-8 -*-
"""
Updates sonata edge file with plasticity related parameters
last modified: András Ecker 05.2021
"""

import os
import gc
from tqdm import tqdm
import yaml
import shutil
import h5py
from cached_property import cached_property
import numpy as np
import pandas as pd
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell


# GluSynapse extra parameters
extra_params = ["volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA", "gmax_p_AMPA", "theta_d", "theta_p"]
# Note: mapping sonata group names to Glusynapse params
# "Use0_TM": "u_syn", "Dep_TM": "depression_time", "Fac_TM": "facilitation_time", "Nrrp_TM": "n_rrp_vesicles",
# "gmax0_AMPA": "conductance", "gmax_NMDA": "conductance * conductance_scale_factor"
usecols = ["syn_id", "Use0_TM", "Dep_TM", "Fac_TM", "Nrrp_TM", "gmax0_AMPA", "volume_CR", "rho0_GB",
           "Use_d_TM", "Use_p_TM", "gmax_d_AMPA", "gmax_p_AMPA", "gmax_NMDA", "theta_d", "theta_p"]
dtypes = {col: np.float32 if col not in ["syn_id", "Nrrp_TM", "rho0_GB"] else np.int64 for col in usecols}


def _get_population(h5f_name):
    """Gets population from sonata edge file"""
    with h5py.File(h5f_name, "r") as h5f:
        populations = list(h5f["edges"])
        if len(populations) > 1:
            raise RuntimeError("Multiple populations in the file")
    return populations[0]


def population_size(h5f_name):
    """Gets size of sonata population"""
    population = _get_population(h5f_name)
    with h5py.File(h5f_name, "r") as h5f:
        h5f_group = h5f["edges/%s/0" % population]
        return h5f_group["%s" % list(h5f_group)[0]].size


def update_population_properties(h5f_name, edge_properties, force=False):
    """Update sonata population with new properties"""
    assert isinstance(edge_properties, dict)
    population = _get_population(h5f_name)
    with h5py.File(h5f_name, "r+") as h5f:
        h5f_group = h5f["edges/%s/0/" % population]
        size = h5f_group["%s" % list(h5f_group)[0]].size
        exists = set(h5f_group) & set(edge_properties.keys())
        if not force and exists:
            raise RuntimeError("Some properties already exist: %s." % exists)
        # add edge properties
        for name, values in edge_properties.items():
            assert len(values) == size
            if force and name in h5f_group:
                del h5f_group[name]
            h5f_group.create_dataset(name, data=values)


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
    def target(self):
        return self.config["circuit"]["target"]

    @property
    def circuit_path(self):
        return self.config["circuit"]["path"]

    @property
    def sims_dir(self):
        return self.config["sims_dir"]

    @property
    def bc(self):
        return os.path.join(self.sims_dir, "BlueConfig")

    @property
    def out_dir(self):
        return self.config["out_dir"]

    @cached_property
    def inp_sonata_fname(self):
        return Circuit(self.circuit_path).config["connectome"]

    @property
    def out_pkl_fname(self):
        return os.path.join(self.out_dir, "edges.pkl")

    @property
    def out_sonata_fname(self):
        return os.path.join(self.out_dir, "edges.sonata")

    def init_sonata(self):
        """Copies base circuit's sonata connectome and adds extra fields"""
        shutil.copyfile(self.inp_sonata_fname, self.out_sonata_fname)
        size = population_size(self.out_sonata_fname)
        # add extra params one-by-one (otherwise there won't be enough memory)
        for extra_param in extra_params:
            dtype = np.float32 if extra_param != "rho0_GB" else np.int64
            fill_value = 0 if extra_param not in ["theta_d", "theta_p"] else -1
            edge_properties = {extra_param: np.full((size,), fill_value=fill_value, dtype=dtype)}
            update_population_properties(self.out_sonata_fname, edge_properties, force=True)

    def merge_csvs(self, save=True):
        """Loads in saved results from all sims and after some preprocessing
        concatenates them to a big DataFrame to be used in the sonata edge file"""
        c = Circuit(self.bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        dfs = []
        for gid in tqdm(gids, desc="Loading saved results", miniters=len(gids)/100):
            f_name = os.path.join(self.sims_dir, "out", "%i.csv" % gid)
            dfs.append(pd.read_csv(f_name, usecols=usecols, index_col=0, dtype=dtypes))
        df = pd.concat(dfs)
        del dfs
        gc.collect()
        # set SS to SS thresholds to -1 (those won't be plastic - see Chindemi et al. 2020, bioRxiv)
        ss_gids = c.cells.ids({"$target": self.target, Cell.MTYPE: "L4_SSC"})
        ss_syn_idx = c.connectome.pathway_synapses(ss_gids, ss_gids)
        df.loc[ss_syn_idx, "theta_d"] = -1
        df.loc[ss_syn_idx, "theta_p"] = -1
        # where depression th. is higher then potentiation th. set both to -1
        bad_syn_idx = df.query("theta_d >= theta_p").index
        df.loc[bad_syn_idx, "theta_d"] = -1
        df.loc[bad_syn_idx, "theta_p"] = -1
        if save:
            df.to_pickle(self.out_pkl_fname)
            print("Dataset of %.2f million samples saved to: %s" % (len(df) / 1e6, self.out_pkl_fname))
        return df


if __name__ == "__main__":

    writer = SonataWriter("../configs/hexO1_v7.yaml")
    # writer.init_sonata()
    df = writer.merge_csvs()


