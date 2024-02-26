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
    install_requires=["cached-property>=1.5.2",
                      "tqdm>=4.66.2",
                      "h5py>=3.10.0",
                      "numpy>=1.26.4",
                      "scipy>=1.12.0",
                      "pyarrow>=15.0.0",  # to keep pandas happy
                      "pandas>=2.2.0",
                      "neurom==3.2.8",
                      "libsonata>=0.1.25",
                      "bluepysnap>=3.0.1",
                      "Connectome-Utilities>=0.4.6",
                      "bluecellulab>=2.2.5"],
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
