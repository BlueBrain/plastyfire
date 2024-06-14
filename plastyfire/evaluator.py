"""
Custom BluePyOpt evaluator for the Graupner & Brunel model
authors: Giuseppe Chindemi (12.2020) + minor modifications by András Ecker (06.2024)
"""

import os
import sys
import pickle
import logging
import traceback
import hashlib
import numpy as np
import pandas as pd
from bluepyopt.evaluators import Evaluator
from bluepyopt.ephys.simulators import NrnSimulator
from bluepyopt.objectives import Objective
from bluepyopt.parameters import Parameter
from ipyparallel import Client
from itertools import product

from plastyfire.config import OptConfig

MIN2MS = 60 * 1000.
FITTED_TAU = 278.3177658387  # previously optimized time constant of Ca*
# could use `SingletonWeightObjective`s, but it's easier to just multiply the (ordered) errors with the values below...
WEIGHT_REDUCE = np.array([1 / 8] * 2 + [1 / 4] * 3)  # weights of each protocol (lower for the first two)
CONFIGS_DIR = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/configs"
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


class Evaluator(Evaluator):
    """Graupner & Brunel plasticity model evaluator"""
    def __init__(self, fit_params, invitro_db, seed, sample_size, ipp_id):
        super(Evaluator, self).__init__()
        self.params = [Parameter(param_name, bounds=(min_bound, max_bound))
                       for param_name, min_bound, max_bound in fit_params]
        self.param_names = [param.name for param in self.params]
        self.invitro_db = invitro_db
        self.seed = seed
        self.sample_size = sample_size
        self.ipp_id = ipp_id
        self.sim = NrnSimulator()
        # Find all simulations
        self.all_sims, self.objectives = [], []
        for elem in self.invitro_db.itertuples():
            # Add objective
            self.objectives.append(Objective(elem.protocol_id))
            # Load simulation config and extract simulation global parameters
            config = OptConfig(os.path.join(CONFIGS_DIR, "%s_%s.yaml" % (elem.pre_mtype, elem.post_mtype)))
            np.testing.assert_almost_equal(config.T / 1000., elem.period_sweep)
            fastforward = config.fastforward
            if fastforward is None:
                fastforward = config.C01_duration * MIN2MS + config.nreps * config.T
            nepsp = int(config.C01_duration * MIN2MS / config.T)
            # Load simulation index (witten by `simwriter.py`)
            sim_idx = pd.read_csv(os.path.join(os.path.split(os.path.split(config.out_dir)[0])[0],
                                               "index_%s_%s.csv" % (elem.pre_mtype, elem.post_mtype)))
            sim_idx.set_index(["frequency", "dt"], inplace=True)
            sim_idx.sort_index(inplace=True)
            # Add target simulations
            paths = sim_idx.loc[elem.frequency_train, elem.dt_train]["path"]
            if DEBUG:
                paths = paths.sample(3)
            else:
                np.random.seed(self.seed)
                paths = paths.sample(self.sample_size, random_state=np.random.randint(9999999))
            self.all_sims.extend([{"protocol_id": elem.protocol_id, "period": elem.period_sweep,
                                   "c01duration": config.C01_duration, "c02duration": config.C02_duration,
                                   "fastforward": fastforward, "nepsp": nepsp, "simpath": path} for path in paths])
        logger.debug("Available sims:")
        for sim in self.all_sims:
            logger.debug(sim)

    def get_param_dict(self, param_values):
        """Build dictionary of parameters for the Graupner & Brunel model
        from an ordered list of values (i.e. an individual)"""
        return dict(zip(self.param_names, param_values))

    def evaluate_with_lists(self, param_values):
        """Evaluate individual"""
        try:
            logger.debug("Evaluating individual: %s", param_values)
            # Check cache for a match
            cachekey = hashlib.md5(str(param_values).encode()).hexdigest()
            pklf_name = os.path.join(".cache", "%s.pkl" % cachekey)
            if os.path.isfile(pklf_name):
                with open(pklf_name, "rb") as f:
                    cache_data = pickle.load(f)  # load cached data
                np.testing.assert_array_equal(param_values, cache_data["individual"])  # verify no collision (OMG)
                logger.debug("Returning results from cache")
                return cache_data["error"]  # Return cache match
            # Set ipyparallel
            rc = Client(profile_dir=".ipython/profile_ipyparallel.%d" % self.ipp_id, timeout=100)
            lview = rc.load_balanced_view()
            # Compute EPSP ratio for all connections
            param_dict = self.get_param_dict(param_values)  # convert individual to parameter dict
            tasks = list(product([param_dict], self.all_sims))
            logger.debug("Tasks:")
            for task in tasks:
                logger.debug(task)
            logger.debug("Running simulations...")
            results = lview.map(compute_epsp_ratio, tasks)
            # Assemble results
            res_db = pd.DataFrame([r for r in results], columns=["protocol_id", "epsp_ratio"])
            logger.debug("Simulations completed, results:")
            logger.debug(res_db)
            insilico_db = res_db.groupby("protocol_id")["epsp_ratio"].agg(["mean", "sem"]).add_suffix("_epsp_ratio").reset_index()
            logger.debug("Aggregating results")
            logger.debug(insilico_db)
            # Compute error, ensuring order
            merged_db = pd.merge(self.invitro_db, insilico_db, on="protocol_id", suffixes=("_invitro", "_insilico"))
            logger.debug("Joining in vitro and in silico results")
            logger.debug(merged_db)
            merged_db["error"] = np.abs((merged_db["mean_epsp_ratio_invitro"] - merged_db["mean_epsp_ratio_insilico"])
                                        / merged_db["sem_epsp_ratio_invitro"])
            error = [float(merged_db.loc[merged_db["protocol_id"] == obj.name, "error"].iloc[0])
                     for obj in self.objectives]
            error = (error * WEIGHT_REDUCE).tolist()  # weight protocols
            logger.debug("Sorting errors")
            logger.debug(error)
            outcome = [float(merged_db.loc[merged_db["protocol_id"] == obj.name, "mean_epsp_ratio_insilico"].iloc[0])
                       for obj in self.objectives]
            # Store results in cache and clean up
            with open(pklf_name, "wb") as f:
                pickle.dump({"error": error, "outcome": outcome, "individual": list(param_values), "resdb": res_db},
                            f, -1)
            logger.debug("Cleaning up")
            rc.close()
            return error
        except Exception:
            # Make sure exception and backtrace are thrown back to parent process
            raise Exception("".join(traceback.format_exception(*sys.exc_info())))

    def init_simulator_and_evaluate_with_lists(self, param_values):
        """
        Set NEURON variables and run evaluation with lists.
        Setting the NEURON variables is necessary when using `ipyparallel`,
        since the new subprocesses have pristine NEURON.
        """
        self.sim.initialize()
        return self.evaluate_with_lists(param_values)
