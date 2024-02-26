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


## Installation
Simply run `pip install .`

All dependencies are declared in the `setup.py` and are available from [pypi](https://pypi.org/)


## Citation
If you use this software, kindly use the following BibTeX entry for citation:

```
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

Copyright (c) 2023 Blue Brain Project / EPFL.