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
from functools import partial
from deap.tools import ParetoFront
from bluepyopt.deapext.IBEADEAPoptimisations import IBEADEAPOptimisation

import plastyfire.evaluator as eval

logger = logging.getLogger("modelfitter")
PROTOCOL_IDX = ["mrk97_01", "mrk97_02", "mrk97_07", "mrk97_08", "sjh06_02"]  # protocols to use for optimization
WEIGHT_REDUCE = np.array([1 / 8] * 2 + [1 / 4] * 3)  # weights of each protocols (lower for the first two)


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
    invitro_db = pd.read_csv("/gpfs/bbp.cscs.ch/project/proj32/glusynapse_20190926_release/invitro/paired_recording.csv")
    invitro_db = invitro_db.loc[invitro_db["protocol_id"].isin(PROTOCOL_IDX)]
    # Create `bluepyopt` evaluator
    np.random.seed(args.seed)
    ev = eval.Evaluator(invitro_db, seed=np.random.randint(9999999),
                        sample_size=args.sample_size, ipp_id=args.ipp_id)
    # Set map function
    pool = multiprocessing.Pool(args.pop_size)
    # Create `bluepyopt` objects
    logger.info("Optimization parameters\nEta = %f Mut = %f Cx = %f" % (args.eta, args.mutpb, args.cxpb))
    opt = IBEADEAPOptimisation(ev, offspring_size=args.pop_size, eta=args.eta, mutpb=args.mutpb, cxpb=args.cxpb,
                               map_function=pool.map, hof=ParetoFront(),
                               fitness_reduce=partial(np.average, weights=WEIGHT_REDUCE),
                               seed=np.random.randint(9999999))
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


