"""
Custom BluePyOpt evaluator for the Graupner & Brunel model
authors: Giuseppe Chindemi (12.2020) + minor modifications by András Ecker (06.2024)
"""

import os
import pickle
import logging
import hashlib
import zmq
import time
import numpy as np
import pandas as pd
import bluepyopt as bpop
from ipyparallel import Client
from itertools import product

from plastyfire.config import OptConfig
from plastifire.pyslurm import submitjob, canceljob

MIN2MS = 60 * 1000.
FITTED_TAU = 278.3177658387  # previously optimized time constant of Ca*
logger = logging.getLogger(__name__)
DEBUG = False


def compute_epsp_ratio(args):
    """Runs sim. of connected pairs and gets EPSP ratio"""
    from plastyfire.ephysutils import Experiment
    from plastyfire.simulator import runconnectedpair

    param_dict, sim_dict = args  # Unpack arguments
    workdir = os.path.dirname(sim_dict["simpath"])
    param_dict["tau_effca_GB_GluSynapse"] = FITTED_TAU
    # Simulate experiment and get EPSP ratio
    raw_results = runconnectedpair(workdir, param_dict, [], sim_dict["fastforward"])
    exp_handler = Experiment(raw_results, c01duration=sim_dict["c01duration"],
                             c02duration=sim_dict["c02duration"], period=sim_dict["period"])
    epsp_ratio = exp_handler.compute_epsp_ratio(sim_dict["nepsp"])
    return sim_dict["protocol_id"], epsp_ratio


class Evaluator(bpop.evaluators.Evaluator):
    """Graupner & Brunel plasticity model evaluator"""
    def __init__(self, invitro_db, seed, sample_size, ipp_id):
        super(Evaluator, self).__init__()
        self.invitro_db = invitro_db
        self.ipp_id = ipp_id
        # Find all simulations
        np.random.seed(seed)
        self.allsims = []
        self.objectives = []
        for elem in self.invitro_db.itertuples():  # TODO
            # Load simulation config and extract simulation global parameters
            config = OptConfig("%s_%s.yaml" % (elem.pre_mtype, elem.post_mtype))
            np.testing.assert_almost_equal(config.T / 1000., elem.period_sweep)
            fastforward = config.fastforward
            if fastforward is None:
                fastforward = config.C01_duration * MIN2MS + config.nreps * config.T
            nepsp = int(config.C01_duration * MIN2MS / config.T)
            # Load simulation index
            sim_idx = pd.read_csv("index_%s_%s.csv" % (elem.pre_mtype, elem.post_mtype))
            sim_idx.set_index(["frequency", "dt"], inplace=True)
            sim_idx.sort_index(inplace=True)
            # Add target simulations
            paths = sim_idx.loc[elem.frequency_train, elem.dt_train]["path"]  # TODO
            if DEBUG:
                paths = paths.sample(3)
            else:
                paths = paths.sample(sample_size, random_state=np.random.randint(9999999))
            self.allsims.extend([{"protocol_id": elem.protocol_id, "period": elem.period_sweep,
                                  "c01duration": config.C01_duration, "c02duration": config.C02_duration,
                                  "fastforward": fastforward, "nepsp": nepsp, "simpath": path} for path in paths])
            # Add objective
            self.objectives.append(bpop.objectives.Objective(elem.protocol_id))
        logger.debug("Available sims:")
        for sim in self.allsims:
            logger.debug(sim)
        # Graupner-Brunel model parameters and boundaries,
        self.graup_params = [  # ("tau_effca_GB_GluSynapse", 150., 350.),
                             ("gamma_d_GB_GluSynapse", 1., 300.),
                             ("gamma_p_GB_GluSynapse", 1., 300.),
                             ("a00", 1., 5.),
                             ("a01", 1., 5.),
                             ("a10", 1., 5.),
                             ("a11", 1., 5.),
                             ("a20", 1., 15.),
                             ("a21", 1., 5.),
                             ("a30", 1., 15.),
                             ("a31", 1., 5.)]
        self.params = [bpop.parameters.Parameter(param_name, bounds=(min_bound, max_bound))
                       for param_name, min_bound, max_bound in self.graup_params]
        self.param_names = [param.name for param in self.params]

    def evaluate_with_lists(self, param_values):
        """Evaluate individual"""
        logger.debug("Evaluating individual: %s", param_values)
        # Check cache for a match
        hashsalt = str(param_values).encode()
        cachekey = hashlib.md5(hashsalt).hexdigest()
        if os.path.isfile(os.path.join(".cache", cachekey)):
            cache_data = pickle.load(open(os.path.join(".cache", cachekey), "rb"))  # load cached data
            np.testing.assert_array_equal(param_values, cache_data["individual"])  # verify no collision (OMG)
            logger.debug("Returning results from cache")
            return cache_data["error"]  # Return cache match
        # Set ipyparallel
        if self.ipp_id is None:
            ipp_id = submitjob()
            time.sleep(300)
        else:
            ipp_id = self.ipp_id
        rc = Client(profile_dir=".ipython/profile_ipyparallel.%d" % ipp_id,
                    context=zmq.Context(), timeout=100)
        lview = rc.load_balanced_view()
        param_dict = self.get_param_dict(param_values)  # convert individual to parameter dict
        # Compute EPSP ratio for all connections
        tasks = list(product([param_dict], self.allsims))
        logger.debug("Tasks:")
        for task in tasks:
            logger.debug(task)
        logger.debug("Running simulations...")
        results = lview.map(compute_epsp_ratio, tasks)
        # Assemble results
        res_db = pd.DataFrame([r for r in results], columns=["protocol_id", "epsp"])
        logger.debug("Simulations completed, results:")
        logger.debug(res_db)
        insilico_db = res_db.groupby("protocol_id")["epsp"].agg(["mean", "sem"]).add_suffix("_epsp").reset_index()
        logger.debug("Aggregating results")
        logger.debug(insilico_db)
        # Compute error, ensuring order
        merged_db = pd.merge(self.invitro_db, insilico_db, on="protocol_id", suffixes=("_invitro", "_insilico"))
        logger.debug("Joining in vitro and in silico results")
        logger.debug(merged_db)
        merged_db["error"] = np.abs((merged_db["mean_epsp_invitro"] - merged_db["mean_epsp_insilico"]) / merged_db["sem_epsp_invitro"])
        error = [float(merged_db.loc[merged_db["protocol_id"] == obj.name, "error"]) for obj in self.objectives]
        logger.debug("Sorting errors")
        logger.debug(error)
        outcome = [float(merged_db.loc[merged_db["protocol_id"] == obj.name, "mean_epsp_insilico"])
                   for obj in self.objectives]
        # Store results in cache and clean up
        with open(os.path.join(".cache", cachekey), "wb") as f:
            pickle.dump({"error": error, "outcome": outcome, "individual": list(param_values), "resdb": res_db}, f, -1)
        logger.debug("Cleaning up")
        rc.close()
        if self.ipp_id is None:
            canceljob(ipp_id)
        return error

    def get_param_dict(self, param_values):
        """Build dictionary of parameters for the Graupner & Brunel model
        from an ordered list of values (i.e. an individual)"""
        gbp = dict(zip(self.param_names, param_values))
        return gbp

