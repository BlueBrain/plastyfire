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
  git clone --recursive https://github.com/andrisecker/plastyfire.git
  cd plastyfire/plastyfire
  pip install -i https://bbpteam.epfl.ch/repository/devpi/simple -e .


Examples
--------

.. code-block::

  cd plastyfire
  source setupenv.sh  # it has hardcoded virtualenv path which you might want to change
  plastyfire TODO...
