#!/bin/sh
#SBATCH --job-name=eval_IBEA
#SBATCH --account=proj96
#SBATCH --partition=prod
#SBATCH --time=6:00:00
#SBATCH --nodes=3
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --chdir=/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/bcl-plastyfire/fitting/n100/seed19091997/
#SBATCH --output=logs/eval-%j.log

# Set environment
source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/setupenv.sh
set -x
set -e
echo "TMPDIR:" $TMPDIR
export IPYTHONDIR="`pwd`/.ipython"
export IPYTHON_PROFILE=ipyparallel.${SLURM_JOBID}

echo "Launching controller"
ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 1m

echo "Launching engines"
srun ipengine --timeout=500 --profile=${IPYTHON_PROFILE}

