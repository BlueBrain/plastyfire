#!/bin/bash

# get mod files
rm -rf mod_fix
module purge
module load archive/2020-12
module load neurodamus-neocortex/1.2-3.1.0
cp -r $NEURODAMUS_NEOCORTEX_ROOT/lib/mod mod_fix
# remove HDF5 reading and writing mod files
rm mod_fix/BinReports.mod
rm mod_fix/BinReportHelper.mod
rm mod_fix/HDF5reader.mod
rm mod_fix/HDF5record.mod
rm mod_fix/SonataReports.mod
rm mod_fix/SonataReportHelper.mod
# copy Giuseppe's mod file
cp /gpfs/bbp.cscs.ch/project/proj32/glusynapse_20190926_release/src/neocortex/common/mod/GluSynapse.mod mod_fix

# compile mod files
rm -rf /x86_64
module purge
module load archive/2020-12
module load neuron/7.9.0b
nrnivmodl mod_fix
