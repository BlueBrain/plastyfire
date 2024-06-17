#!/bin/sh
#SBATCH --job-name=plast_opt
#SBATCH --account=proj96
#SBATCH --partition=prod
#SBATCH --qos=longjob
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --chdir=/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/bcl-plastyfire/fitting/n100/seed19091997/
#SBATCH --output=opt-%j.log

# Set environment
source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/setupenv.sh
krenew -b -K 10
set -x
set -e

# Set next job
cp /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/modelfitter.sh .
sbatch --dependency=afterany:${SLURM_JOBID} modelfitter.sh

# Run
cp /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/ipp.sh .
python /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/plastyfire/modelfitter.py --gen=100 --sample_size=100 --seed=19091997 -v

