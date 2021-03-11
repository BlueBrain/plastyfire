plastyfire
============

optimize and generalize plastic synapses @BBP
The package heavily relies on BGLibPy - the single cell simulator of the BBP (built on top of PyNEURON) so if you're not in the BBP probably you can't run anything


Installation
------------

.. code-block::

  module purge
  module load archive/2020-12
  module load python/3.7.4
  python -m venv dev-plastyfire
  source dev-plastyfire/bin/activate
  git clone https://github.com/andrisecker/plastyfire.git
  cd plastyfire
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .
  cd plastyfire
  ./compile_mods.sh


EXC-EXC pathway-specific recipe (xml->DataFrame)
--------

.. code-block::

  # the first part is a bit hacky as the xml reader is not packaged yet
  module purge
  module load unstable
  module load spykfunc/0.15.9
  export PYTHONPATH=$PYTHONPATH:/gpfs/bbp.cscs.ch/home/matwolf/work/funcz-183/linux-rhel7-x86_64/gcc-9.3.0/spykfunc-develop-2gfrwu/lib/python3.8/site-packages
  python xmlrecipe.py
  module purge
  source ../../dev-plastyfire/bin/activate
  python glusynapserecipe.py


Run single cell sims to get C_pre and C_post
--------

  source setupenv.sh  # it has hardcoded virtualenv path which you might want to change
