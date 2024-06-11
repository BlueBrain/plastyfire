"""
Main run script for parameter optimization that parses command line arguments, runs optimization, and saves results
authors: Giuseppe Chindemi (12.2020) + minor modifications by András Ecker (06.2024)
"""

import os
import pickle
import logging
import argparse
import multiprocessing
import numpy as np
import pandas as pd
from deap.tools import ParetoFront
from bluepyopt.deapext.optimisations import IBEADEAPOptimisation

import plastyfire.evaluator as eval

logger = logging.getLogger("modelfitter")
CSVF_NAME = "/gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/biodata/paired_recordings.csv"
PROTOCOL_IDX = ["mrk97_01", "mrk97_02", "mrk97_07", "mrk97_08", "sjh06_02"]  # protocols to use for optimization
# model parameters to be optimized (and their boundaries)
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


if __name__ == '__main__':
    # Parse command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_size", type=int, default=100, help="Number of in silico connections per protocol")
    parser.add_argument("--seed", type=int, default=1234, help="RNG master seed")
    parser.add_argument("-s", "--pop_size", type=int, default=128, help="Population size")
    parser.add_argument("-g", "--gen", type=int, default=30, help="Number of generations")
    parser.add_argument("-e", "--eta", type=float, default=20., help="Eta parameter")
    parser.add_argument("-m", "--mutpb", type=float, default=.7, help="Mutation probability")
    parser.add_argument("-c", "--cxpb", type=float, default=.3, help="Crossover probability")
    parser.add_argument("--ipp_id", type=int, help="IPython Parallel ID")
    parser.add_argument("-v", "--verbose", default=False, action="store_true", help="Verbose messages")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    # Configure logger
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    # Enable debugging mode
    if args.debug:
        eval.DEBUG = True
        args.pop_size = 4
        args.gen = 3
    # Set support directories
    if not os.path.exists(".cache"):
        os.makedirs(".cache")

    # Load in vitro results
    invitro_db = pd.read_csv(CSVF_NAME)
    invitro_db = invitro_db.loc[invitro_db["protocol_id"].isin(PROTOCOL_IDX)]
    # Create `bluepyopt` evaluator
    np.random.seed(args.seed)
    ev = eval.Evaluator(FIT_PARAMS, invitro_db, np.random.randint(9999999), args.sample_size, args.ipp_id)
    # Set map function
    pool = multiprocessing.Pool(args.pop_size)
    # Create `bluepyopt` optimization
    logger.info("Optimization parameters\nEta = %f Mut = %f Cx = %f" % (args.eta, args.mutpb, args.cxpb))
    np.random.seed(args.seed + 1)
    opt = IBEADEAPOptimisation(ev, offspring_size=args.pop_size, eta=args.eta, mutpb=args.mutpb, cxpb=args.cxpb,
                               map_function=pool.map, hof=ParetoFront(), seed=np.random.randint(9999999))
    # Run optimization
    cpf_name = "checkpoint.pkl"
    continue_cp = os.path.isfile(cpf_name)
    logger.info("Resuming optimization" if continue_cp else "Starting a new optimization")
    pop, hof, log, history = opt.run(max_ngen=args.gen, continue_cp=continue_cp,
                                     cp_filename=cpf_name, cp_frequency=1)
    # Gather and store best solution
    best = hof[np.argmin([np.linalg.norm(np.array(ind.fitness.values)) for ind in hof])]
    with open("bestsol.pkl", "wb") as f:
        pickle.dump(ev.get_param_dict(best), f, -1)
    logger.info("Optimization concluded")


