#!/bin/bash

module purge
module load archive/2020-12
module load neurodamus-neocortex/1.2-3.1.0

source /gpfs/bbp.cscs.ch/project/proj96/home/ecker/dev-plastyfire/bin/activate

CUSTOM_ND_ROOT=/gpfs/bbp.cscs.ch/project/proj96/home/ecker/gerrit
export BGLIBPY_MOD_LIBRARY_PATH=$CUSTOM_ND_ROOT/x86_64/libnrnmech.so
export BGLIBPY_RNG_MODE=Random123
