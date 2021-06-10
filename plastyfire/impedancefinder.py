# -*- coding: utf-8 -*-
"""
Runscript for single cells that calculates features used for ML (distance to soma and impedance for all synapses)
last modified: András Ecker 05.2021
"""

import os
import time
import yaml
import logging
import argparse
import numpy as np
from bluepy.v2 import Circuit
from bluepy.v2.enums import Cell, Synapse

logging.basicConfig(level=logging.INFO)
L = logging.getLogger("impedancefinder")

FREQS = [1, 5, 10, 20, 50, 100, 200, 1000]


class ImpedanceFinder(object):
    """Class (to store info about and) to run single cell 'simulations' in BGLibPy"""

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

    def run(self, post_gid):
        from plastyfire.simulator import imp_finder
        c = Circuit(self.bc)
        gids = c.cells.ids({"$target": self.target, Cell.SYNAPSE_CLASS: "EXC"})
        pre_gids = np.intersect1d(c.connectome.afferent_gids(post_gid), gids).astype(np.int)
        syn_locs = c.connectome.pathway_synapses(pre_gids, post_gid, [Synapse.POST_SECTION_ID, "afferent_section_pos"])
        syn_locs = syn_locs.rename(columns={Synapse.POST_SECTION_ID: "sec_id", "afferent_section_pos": "pos"})
        L.info("Calculating impedance for %i synapses of gid %i (%s)" % (len(syn_locs), post_gid,
                                                                         c.cells.get(post_gid, Cell.MTYPE)))
        inp_imps = imp_finder(self.bc, post_gid, syn_locs, FREQS, True)
        inp_imps.index.name = "syn_id"
        L.info("Saving results to out_mld/%i.csv" % post_gid)
        inp_imps.astype(np.float32).to_pickle(os.path.join(self.sims_dir, "out_mld", "%i.pkl" % post_gid))


if __name__ == "__main__":

    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", type=str, help="path to YAML config file")
    parser.add_argument("post_gid", type=int, help="post_gid to simulate")
    args = parser.parse_args()

    sim = ImpedanceFinder(args.config_path)
    start_time = time.time()
    sim.run(args.post_gid)
    L.info("Elapsed time: %s" % time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)))

