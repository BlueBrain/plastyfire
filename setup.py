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
                      "h5py>=2.7,<3",  # can't use h5py>3 with bluepy 0.16.0
                      "numpy>=1.19.4",
                      "scipy>=1.6.0",
                      "pandas>=1.2.1",
                      "libsonata>=0.1.6",
                      "xgboost>=1.4.0",
                      "bayesian-optimization>=1.2.0"
                      ] + [
                      "bluepy[all]==0.16.0",
                      "bglibpy<4.1"  # 4.1 is the first version that doesn't keep global syn_idx
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
