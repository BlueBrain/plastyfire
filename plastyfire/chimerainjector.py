"""
Inject a chimera solution into `bluepyopt` checkpoint, i.e.,
trains an XGBoost model on existing solutions saved in cache (see `xgbevaluator.py`)
uses it to finish the optimization, takes the best individual, and overwrites the last individual in cache with
this "chimera" solution, after which the optimization should be continued as before
authors: Giuseppe Chindemi (12.2020) + modifications by András Ecker (07.2024)
"""

import os
import sys
import shutil
import pickle
import logging
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from deap.tools import ParetoFront
from bluepyopt.deapext.optimisations import IBEADEAPOptimisation
from bluepyopt.deapext.optimisations import WSListIndividual

from plastyfire.xgbevaluator import XGBEvaluator

# print info into console
logging.basicConfig(stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
CSVF_NAME = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/biodata/paired_recordings.csv"
PROTOCOL_IDX = ["mrk97_01", "mrk97_02", "mrk97_07", "mrk97_08", "sjh06_02"]  # protocols to use for optimization
# Graupner & Brunel model parameters to be optimized (and their boundaries)
FIT_PARAMS = [  # ("tau_effca_GB_GluSynapse", 150., 350.),
              ("gamma_d_GB_GluSynapse", 50., 200.),
              ("gamma_p_GB_GluSynapse", 150., 300.),
              ("a00", 1., 5.),
              ("a01", 1., 5.),
              ("a10", 1., 5.),
              ("a11", 1., 5.),
              ("a20", 1., 10.),
              ("a21", 1., 5.),
              ("a30", 1., 10.),
              ("a31", 1., 5.)]
POOL_SIZE = 16  # larger pools don't seem to work due to OpenBLAS threading (you're wellcome to debug...)


if __name__ == '__main__':
    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1234, help="RNG master seed")
    parser.add_argument("-s", "--pop_size", type=int, default=128, help="Population size")
    parser.add_argument("-g", "--gen", type=int, default=30, help="Number of generations")
    parser.add_argument("-e", "--eta", type=float, default=20., help="Eta parameter")
    parser.add_argument("-m", "--mutpb", type=float, default=.7, help="Mutation probability")
    parser.add_argument("-c", "--cxpb", type=float, default=.3, help="Crossover probability")
    args = parser.parse_args()

    # Load in vitro results
    invitro_db = pd.read_csv(CSVF_NAME)
    invitro_db = invitro_db.loc[invitro_db["protocol_id"].isin(PROTOCOL_IDX)]
    # Create `bluepyopt` evaluator (will take some time as it does some XGBoost meta parameter tuning)
    ev = XGBEvaluator(FIT_PARAMS, invitro_db, os.getcwd())
    # Set map function (XGBoost doesn't like either `fork`, nor big pool sizes)
    pool = multiprocessing.get_context("spawn").Pool(POOL_SIZE)
    # Create `bluepyopt` optimization
    np.random.seed(args.seed + 1)
    opt = IBEADEAPOptimisation(ev, offspring_size=args.pop_size, eta=args.eta, mutpb=args.mutpb, cxpb=args.cxpb,
                               map_function=pool.map, hof=ParetoFront(), seed=np.random.randint(9999999))
    # Run optimization (w/o overwriting the checkpoint)
    cpf_name = "checkpoint.pkl"
    continue_cp = os.path.isfile(cpf_name)
    pop, hof, log, history = opt.run(max_ngen=args.gen, continue_cp=continue_cp, cp_filename=cpf_name, cp_frequency=0)
    # Get best solution
    best = hof[np.argmin([np.linalg.norm(np.array(ind.fitness.values)) for ind in hof])]
    # Load checkpoint and overwrite last individual with best solution
    with open(cpf_name, "rb") as f:
        cp = pickle.load(f)
    new_ind = WSListIndividual(best, obj_size=len(cp["population"][0].fitness.wvalues))
    cp["population"][-1] = new_ind
    cp["parents"][-1] = new_ind
    cpf_name_tmp = cpf_name + ".tmp"
    with open(cpf_name_tmp, "wb") as f:
        pickle.dump(cp, f, -1)
    if os.path.isfile(cpf_name_tmp):
        shutil.copy(cpf_name_tmp, cpf_name)



