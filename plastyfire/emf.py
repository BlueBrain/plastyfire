# -*- coding: utf-8 -*-
"""
Extra Morphology Features (for ML)
classify apical neurites into: trunk, tuft and obliques (basal dendrites are straightforward in NeuroM)
adds path distance, branch order and dendritic diameter (of the whole section, not the specific segment)
authors: András Ecker, Alexis Arnaudon, Sirio Bolaños-Puchet; last modified: 06.2021
"""

import os
import yaml
from tqdm import tqdm
import warnings
import numpy as np
import pandas as pd
import neurom as nm
from morphio import Morphology
from morph_tool.apical_point import apical_point_section_segment
import libsonata
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell, Synapse
from bluepyparallel import evaluate

pd.set_option("display.max_colwidth", 10000)  # solve truncated strings: see https://github.com/pandas-dev/pandas/issues/9784
warnings.filterwarnings("ignore", category=UserWarning)  # to disable ascii morph warning ...

# enums for ndarray columns and dendrite types
POST_GID, SEC_ID, LOC, DIST, BR_ORD, DIAM = 0, 1, 2, 3, 4, 5
BASAL, OBLIQUE, TUFT, TRUNK, APICAL = 0, 1, 2, 3, 4


def _isin_fast(whom, where, in_parallel=False):
    """Sirio's in line np.isin using joblib as parallel backend"""
    if in_parallel:
        from joblib import Parallel, delayed
        nproc = os.cpu_count()
        with Parallel(n_jobs=nproc, prefer="threads") as parallel:
            flt = parallel(delayed(np.isin)(chunk, where) for chunk in np.array_split(whom, nproc))
        return np.concatenate(flt)
    else:
        return np.isin(whom, where)


def get_edge_ids_fast(edge_file, nodes_src, nodes_tgt, population=None, parallel=False):
    """Sirio's inline `connecting_edges` (~4x faster when using parallelization)"""
    synfile = libsonata.EdgeStorage(edge_file)
    synpop = synfile.open_population(population if population is not None else list(synfile.population_names)[0])
    # get nodes_src -> nodes_tgt edge IDs
    ids_aff = synpop.afferent_edges(nodes_tgt)
    src_aff = synpop.source_nodes(ids_aff)
    flt = _isin_fast(src_aff, nodes_src, in_parallel=parallel)
    ids = ids_aff.flatten()[flt]
    # Used by bluepy (bluepysnap) [slower]
    # ids = synpop.connecting_edges(nodes_src, nodes_tgt).flatten()
    return ids


def _get_morph_paths(df, morh_dir):
    """Gets unique morphology path"""
    df = df.drop_duplicates().reset_index(drop=True)
    df["path"] = morh_dir + "/" + df["morphology"].astype("str") + ".asc"
    return df


def _compute_apical(row):
    """Dummy evaluator for bluepyparallel"""
    neuron = Morphology(row["path"])  # morph-tool needs the MorphIO format (while the rest of the code the NeuroM ones)
    return {"apical_point": apical_point_section_segment(neuron)[0]}


def compute_apical(df):
    """Compute apical points for all unique morphologies in parallel"""
    df = evaluate(df, _compute_apical, new_columns=[["apical_point", None]], parallel_factory="multiprocessing")
    return df


def get_morph_features(neuron, apical_section_id):
    """Further classifies apical dendrites to: trunk, tuft and oblique dendrites
    adds path distance and branch order and calculates mean section diameter"""
    neurite_features = {LOC: {}, DIST: {}, BR_ORD: {}, DIAM: {}}
    path_distances = nm.get("section_path_distances", neuron)
    branch_orders = nm.get("section_branch_orders", neuron)
    # basal dendrites are straightforward in NeuroM
    for section in nm.iter_sections(neuron):
        if section.type == nm.NeuriteType.basal_dendrite:
            neurite_features[LOC][section.id] = BASAL
            neurite_features[DIST][section.id] = path_distances[section.id-1]
            neurite_features[BR_ORD][section.id] = branch_orders[section.id-1]
            neurite_features[DIAM][section.id] = np.mean(section.points[:, 3])
    if apical_section_id is not None:
        apical_section = neuron.sections[apical_section_id]
        # in the beginning initialize all apicals as oblique dendrites (tuft and trunk will be overwritten...)
        for section in nm.iter_sections(neuron):
            if section.type == nm.NeuriteType.apical_dendrite:
                neurite_features[LOC][section.id] = OBLIQUE
                neurite_features[DIST][section.id] = path_distances[section.id-1]
                neurite_features[BR_ORD][section.id] = branch_orders[section.id-1]
                neurite_features[DIAM][section.id] = np.mean(section.points[:, 3])
        # above apical point: tuft dendrites
        for section in apical_section.ipreorder():
            neurite_features[LOC][section.id] = TUFT
        # upstream of apical point: trunk
        for section in apical_section.iupstream():
            neurite_features[LOC][section.id] = TRUNK
    else:  # if apicals cannot be further divided still return something meaningfull
        for section in nm.iter_sections(neuron):
            if section.type == nm.NeuriteType.apical_dendrite:
                neurite_features[LOC][section.id] = APICAL
                neurite_features[DIST][section.id] = path_distances[section.id-1]
                neurite_features[BR_ORD][section.id] = branch_orders[section.id-1]
                neurite_features[DIAM][section.id] = np.mean(section.points[:, 3])
    return neurite_features


def add_morph_features(morph_features, post_gid, neurite_features):
    """Adds fine-grained apical dendrite names and dendritic diam (of the whole section) to synapse df"""
    post_gid_idx = np.where(morph_features[:, POST_GID] == post_gid)[0]
    for sec_id in np.unique(morph_features[post_gid_idx, SEC_ID]):
        row_idx = post_gid_idx[np.where(morph_features[post_gid_idx, SEC_ID] == sec_id)[0]]
        morph_features[row_idx, LOC] = neurite_features[LOC][sec_id]
        morph_features[row_idx, DIST] = neurite_features[DIST][sec_id]
        morph_features[row_idx, BR_ORD] = neurite_features[BR_ORD][sec_id]
        morph_features[row_idx, DIAM] = neurite_features[DIAM][sec_id]


class MorphFeatures(object):
    """Class (to store info about and) to get extra morphological features for ML"""

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
    def bc(self):
        return os.path.join(self.sims_dir, "BlueConfig")

    @property
    def out_fname(self):
        return os.path.join(self.sims_dir, "out_mld", "extra_morph_features.pkl")

    def run(self):
        """main function that gets the extra morph features (first as an ndarray then saves it as a nice DataFrame)"""
        c = Circuit(self.bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        # pre-build lookups for morphologies
        morph_df = c.cells.get(gids, properties=["morphology"])
        apical_df = compute_apical(_get_morph_paths(morph_df, c.config["morphologies"]))
        # pre-calculate all synapse IDs
        syn_idx = get_edge_ids_fast(c.config["connectome"], gids-1, gids-1, None, True)
        # get section IDs and prepare data structure
        syn_locs = c.connectome.synapse_properties(syn_idx, [Synapse.POST_GID, Synapse.POST_SECTION_ID])
        morph_features = np.zeros((len(syn_idx), 6), dtype=np.float32)
        morph_features[:, POST_GID:LOC] = syn_locs.to_numpy()
        neurite_features = {}
        for post_gid in tqdm(gids, desc="Getting morphology features", miniters=len(gids) / 100):
            morph_name = morph_df.loc[post_gid, "morphology"]
            if morph_name not in neurite_features:
                morph = nm.load_neuron(apical_df.loc[apical_df["morphology"] == morph_name,
                                                     "path"].to_string(index=False))
                try:
                    apical_point = int(apical_df.loc[apical_df["morphology"] == morph_name, "apical_point"])
                except:
                    apical_point = None
                nf = get_morph_features(morph, apical_point)
                neurite_features[morph_name] = nf
            else:
                nf = neurite_features[morph_name]
            add_morph_features(morph_features, post_gid, nf)
        # convert ndarray to pandas DataFrame and save it
        df = pd.DataFrame(data=morph_features[:, LOC:], index=syn_idx, columns=["loc", "dist", "br_ord", "diam"])
        df.replace({"loc": {BASAL: "basal", OBLIQUE: "oblique", TUFT: "tuft",
                            TRUNK: "trunk", APICAL: "apical"}}, inplace=True)
        df = df.astype({"dist": np.float32, "br_ord": np.uint8, "diam": np.float32})
        df.index.name = "syn_id"
        df.to_pickle(self.out_fname)


if __name__ == "__main__":

    mf = MorphFeatures("../configs/hexO1_v7.yaml")
    mf.run()
