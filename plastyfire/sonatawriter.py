# -*- coding: utf-8 -*-
"""
Updates sonata edge file with plasticity related parameters
last modified: András Ecker 03.2021
"""

import os
import logging
import yaml
import pathlib
import shutil
import h5py
from cached_property import cached_property
import numpy as np
from bluepy.v2 import Circuit


# GluSynapse extra parameters
extra_params = ["volume_CR", "rho0_GB", "Use_d_TM", "Use_p_TM", "gmax_d_AMPA", "gmax_p_AMPA", "theta_d", "theta_p"]
# Note: mapping sonata group names to Glusynapse params
# "Use0_TM": "u_syn", "Dep_TM": "depression_time", "Fac_TM": "facilitation_time", "Nrrp_TM": "n_rrp_vesicles",
# "gmax0_AMPA": "conductance", "gmax_NMDA": "conductance * conductance_scale_factor"
L = logging.getLogger(__name__)


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
            L.info("Added edge Property: %s", name)


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
    def circuit_path(self):
        return self.config["circuit_path"]

    @property
    def target(self):
        return self.config["target"]

    @property
    def extra_recipe_path(self):
        return self.config["extra_recipe_path"]

    @property
    def out_sonata_fname(self):
        return self.config["out_sonata_fname"]

    @property
    def base_dir(self):
        return self.config["base_dir"]

    @property
    def fit_params(self):
        return self.config["fit_params"]

    @cached_property
    def circuit(self):
        return Circuit(self.circuit_path)

    @cached_property
    def inp_sonata_fname(self):
        return self.circuit.config["connectome"]

    @cached_property
    def valid_gids(self):
        from bluepy.v2.enums import Cell
        return self.circuit.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})

    def init_sonata(self):
        """Copies base circuit's sonata connectome and adds extra fields"""
        shutil.copyfile(self.inp_sonata_fname, self.out_sonata_fname)
        size = population_size(self.out_sonata_fname)
        # add extra params one-by-one (otherwise there won't be enough memory)
        for extra_param in extra_params:
            fill_value = 0.0 if extra_param not in ["theta_d", "theta_p"] else -1.0
            edge_properties = {extra_param: np.full((size,), fill_value=fill_value, dtype=np.float32)}
            update_population_properties(self.out_sonata_fname, edge_properties, force=True)

    def write_sim_files(self):
        """Writes simple BlueConfig used by BGLibPy"""
        from plastyfire.bcwriter import BCWriter

        pathlib.Path(self.base_dir).mkdir(exist_ok=True)
        target_fname = os.path.join(self.base_dir, "user.target")
        # Write empty user.target (just because there has to be one)
        with open(target_fname, "w") as f:
            f.write("")
        # Write BlueConfig
        bcw = BCWriter(self.circuit_path, duration=2000, target=self.target, target_file=target_fname, base_seed=1909)
        bcw.write(self.base_dir)

    def debug(self, pre_gid=147984, post_gid=147748):
        """Stupid function for debugging purpose"""
        from plastyfire.epg import ParamsGenerator
        from plastyfire.simulator import c_pre_finder, c_post_finder

        self.write_sim_files()
        pgen = ParamsGenerator(self.circuit_path, self.extra_recipe_path)
        default_params = pgen.generate_params(pre_gid, post_gid)

        try:
            cpre = c_pre_finder(self.base_dir, self.fit_params, default_params, pre_gid, post_gid,
                                fixhp=True, invivo=False)
            _, cpost = c_post_finder(self.base_dir, self.fit_params, default_params, pre_gid, post_gid,
                                     stimulus=None, fixhp=True, invivo=False)
        except RuntimeError:
            # Something went wrong, set negative threshols (no plasticity)
            for synapse_id, synapse_params in default_params.items():
                synapse_params["theta_d"] = -1.0
                synapse_params["theta_p"] = -1.0
        else:
            # Compute thresholds (this shouldn"t be here, but for now ok)
            for synapse_id, synapse_params in default_params.items():
                if synapse_params["loc"] == "basal":
                    synapse_params["theta_d"] = self.fit_params["a00"] * cpre[synapse_id] +\
                                                self.fit_params["a01"] * cpost[synapse_id]
                    synapse_params["theta_p"] = self.fit_params["a10"] * cpre[synapse_id] +\
                                                self.fit_params["a11"] * cpost[synapse_id]
                elif synapse_params["loc"] == "apical":
                    synapse_params["theta_d"] = self.fit_params["a20"] * cpre[synapse_id] +\
                                                self.fit_params["a21"] * cpost[synapse_id]
                    synapse_params["theta_p"] = self.fit_params["a30"] * cpre[synapse_id] +\
                                                self.fit_params["a31"] * cpost[synapse_id]
                else:
                    raise ValueError("Unknown location")


if __name__ == "__main__":

    writer = SonataWriter("../configs/hexO1_v7.yaml")
    writer.init_sonata()
    #writer.debug()


