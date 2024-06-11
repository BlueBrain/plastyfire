#!/bin/sh
#SBATCH --job-name=plast_opt
#SBATCH --account=proj96
#SBATCH --partition=prod
#SBATCH --time=24:00:00
#SBATCH --nodes=16
#SBATCH --constraint=cpu
#SBATCH --cpus-per-task=2
#SBATCH --no-requeue
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --chdir=/gpfs/bbp.cscs.ch/project/proj96/scratch/home/ecker/bcl-plastyfire/fitting/n100/seed19091997/
#SBATCH --output=opt.log

# Set environment
source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/setupenv.sh
krenew -b -K 10

# Set environment
echo "TMPDIR:" $TMPDIR
export IPYTHONDIR="`pwd`/.ipython"
export IPYTHON_PROFILE=ipyparallel.${SLURM_JOBID}

echo "Launching controller"
ipcontroller --init --ip='*' --sqlitedb --ping=30000 --profile=${IPYTHON_PROFILE} &
sleep 1m

echo "Launching engines"
srun ipengine --timeout=500 --profile=${IPYTHON_PROFILE} &
sleep 5m

# Set next job
sbatch --dependency=afterany:${SLURM_JOBID} modelfitter.sh

# Run
python /gpfs/bbp.cscs.ch/project/proj96/home/ecker/plastyfire/plastyfire/modelfitter.py --gen=200 --sample_size=100 --seed=19091997 --ipp_id=${SLURM_JOBID} -v
