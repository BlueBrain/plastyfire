# Plastyfire

Plastic synapses @BBP

The package relies on [bluecellulab](https://github.com/BlueBrain/BlueCelluLab) to run single cell simulation and record calcium peaks at synapses that are used to generate plasticity thresholds as described in [Chindemi et al. 2022, Nature Comms.](https://www.nature.com/articles/s41467-022-30214-w).


## Examples

Simplest use case: find C_pre and C_post between a connected pair of neurons

```
from plastyfire.simulator import spike_threshold_finder, c_pre_finder, c_post_finder

sim_config = "<PATH-TO-SONATA-SIMULATION-CONFIG>"  # see example below in which `simwriter.py` generates one
pre_gid, post_gid = 470, 458

# deliver a spike from the presynaptic cell (not simulated) and measure [Ca^{2+}]_i peaks at synapses on the postsynaptic cell (AKA. C_pre values)
c_pre = c_pre_finder(sim_config, None, None, pre_gid, post_gid)
print(c_pre)
# find parameters of current injection that make the postsynaptic cell fire a single action potential
nspikes, freq = 1, 0.1
for pulse_width in [1.5, 3, 5]:
    sim_results = spike_threshold_finder(sim_config, post_gid, nspikes, freq, pulse_width, 1000., 0.05, 5., 100)
    if sim_results is not None:
        break
stimulus = {"nspikes": nspikes, "freq": freq, "width": sim_results["width"], "offset": sim_results["offset"], "amp": sim_results["amp"]}
# use that stimulus to elicit a single postsynaptic spike and measure bAP-induced [Ca^{2+}]_i peaks at synapses made by the presynaptic cell (AKA. C_post values)
c_post = c_post_finder(sim_config, None, None, pre_gid, post_gid, stimulus)
print(c_post)

```

A more systematic approach: do the above for all connections and write the results to SONATA edge file (that can be used for network simulations)

```
python simwriter.py  # write `simulation_config.json` based on the YAML config file and batch scripts for `thresholdfinder.py` below
python thresholdfinder.py {config_path} {post_gid}  # find all C_pre and C_post values of a given `post_gid` and saves them into a CSV
python sonatawriter.py  # concatenates saved CSV files and writes a new SONATA edge file
```

An other (more complicated) use case is the optimization of the parameters of the plasticity model

```
python simwriter.py  # find suitable pairs and write simulation files based on the YAML config file(s)
python modelfitter.py  # optimize parameters using a genetic algorithm (by running hundreds of thousands of simulations)
```

The fitting part takes several days (the setup described in [Chindemi et al. 2022, Nature Comms.](https://www.nature.com/articles/s41467-022-30214-w) uses 100 pairs, for 5 STDP protocols and 128 individuals in a single generation, which leads to 64k simulations to run for minutes of biological time) and therefore is managed by batch sripts (instead of the simple `python modelfitter.py` above). Two main modes are implemented for running it:

- `modelfitter.sh` lanches a single, single node job, which will fire up small jobs per individual parallelized across nodes by IPyparallel (see also `ipp.sh`)
- `single_alloc_modelfitter.sh` also launches a single job, but it instantiates a single big IPyparallel cluster across many nodes that manages the whole optimization process

To try to speed up the convergence, run `chimerainjector.sh` which will train an XGBoost model based on the individuals evaluated so far, finish the optimization with the trained XGBoost model predicting the outcomes, and then injecting the best solution back to the optimization checkpoint (to be used in further generations).


## Installation
Simply run `pip install .`

All dependencies are declared in the `setup.py` and are available from [pypi](https://pypi.org/)


## Citations
If you use this software, kindly use the following BibTeX entries for citation:

```
@article{Chindemi2022,
author = {Chindemi, Giuseppe and Abdellah, Marwan and Amsalem, Oren and Benavides-Piccione, Ruth and Delattre, Vincent and Doron, Michael and Ecker, Andras and Jaquier, Aurelien T.. and King, James and Kumbhar, Pramod and Monney, Caitlin and Perin, Rodrigo and R{\"{o}}ssert, Christian and Tuncel, M. Anil and van Geit, Werner and DeFelipe, Javier and Graupner, Michael and Segev, Idan and Markram, Henry and Muller, Eilif B},
doi = {10.1038/s41467-022-30214-w},
journal = {Nature Communications},
number = {3038},
title = {{A calcium-based plasticity model predicts long-term potentiation and depression in the neocortex}},
volume = {13},
year = {2022}
}

@article{Ecker2023,
author = {Ecker, Andr{\'{a}}s and Santander, Daniela Egas and Abdellah, Marwan and Alonso, Jorge Blanco and Bola{\~{n}}os-Puchet, Sirio and Chindemi, Giuseppe and Isbister, James B and King, James Gonzalo and Kumbhar, Pramod and Magkanaris, Ioannis and Muller, Eilif B and Reimann, Michael W},
doi = {https://doi.org/10.1101/2023.08.07.552264},
journal = {bioRxiv},
title = {{Long-term plasticity induces sparse and specific synaptic changes in a biophysically detailed cortical model}},
year = {2023}
}
```


## Acknowledgements & Funding
The development of this software was supported by funding to the Blue Brain Project, a research center of the École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal Institutes of Technology.

Copyright (c) 2024 Blue Brain Project / EPFL.