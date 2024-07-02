#!/bin/sh
#SBATCH --job-name=chimera
#SBATCH --account=proj96
#SBATCH --partition=prod
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --chdir=/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/bcl-plastyfire/fitting/n100/seed19091997/
#SBATCH --output=logs/chimera.log

source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/setupenv.sh
cp /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/plastyfire/chimerainjector.py .
python chimerainjector.py --gen=100 --seed=19091997

