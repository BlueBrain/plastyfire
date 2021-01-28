#!/bin/bash

module purge
module load archive/2020-12
module load neurodamus-neocortex/1.1-3.0.2

source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/dev-plastyfire/bin/activate

#unset PMI_RANK
export BGLIBPY_RNG_MODE=Random123
#export PYTHONWARNINGS="ignore"
