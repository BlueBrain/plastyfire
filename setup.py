#!/usr/bin/env python

import imp
import sys

from setuptools import setup, find_packages

if sys.version_info < (3, 6):
    sys.exit("Sorry, Python < 3.6 is not supported")

VERSION = imp.load_source("", "plastyfire/version.py").__version__

setup(
    name="plastyfire",
    author=["Giuseppe Chindemi", "Andras Ecker"],
    author_email="andras.ecker@epfl.ch",
    version=VERSION,
    description="plasticity @BBP",
    long_description="plastyfire: generalize plastic synapses @BBP",
    url="http://bluebrain.epfl.ch",
    license="LGPL-3.0",
    install_requires=["pyyaml>=5.3.1",
                      "cached-property>=1.5.2",
                      "tqdm>=4.52.0",
                      "h5py>=2.7,<3",  # voxcell and thus bluepy 0.16.0 require h5py<3
                      "numpy==1.17.3",
                      "scipy>=1.6.0",
                      "pandas>=1.2.1",
                      "neurom==1.5.4",  # morph-tool<2.5 only works with NeuroM v1 and NeuroM v1 version higher than this require h5py>3...
                      "morphio==2.6.2",  # morph-tool<2.5 only works with MorphIO v2 and MorphIO v2 version higher than this require h5py>3...
                      "morph-tool==2.2.21",  # bluepy 0.16 is only constistent with morph-tool<2.5 and this is the highest version of morph-tool, that works with h5py<3
                      "libsonata>=0.1.6",
                      "xgboost>=1.4.0",
                      "bayesian-optimization>=1.2.0"
                      ] + [
                      "bluepy[all]==0.16.0",
                      "bglibpy<4.1",  # 4.1 is the first version that doesn't keep global syn_idx
                      "bluepyparallel>=0.0.5"
                      ],
    packages=find_packages(),
    python_requires=">=3.6",
    extras_require={"docs": ["sphinx", "sphinx-bluebrain-theme"]},
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
