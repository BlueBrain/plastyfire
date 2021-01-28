"""
BlueConfig writer (since BGLibPy doesn"t work w/o a BlueConfig)
authors: Giuseppe Chindemi (09.2020)
+minor modifications for new bluepy version by András Ecker (01.2021)
"""

import os
from bluepy.v2 import Circuit

run_default_template = """\
Run Default
{{
    
    CircuitPath {circuit_path}
    nrnPath {nrn_path}
    MorphologyPath {morphology_path}
    METypePath {metype_path}
    MEComboInfoFile {mecombo_file}
    CellLibraryFile circuit.mvd3
    
    CurrentDir {output_path}
    OutputRoot {output_path}
    TargetFile {target_file}
    
    CircuitTarget {target}
    RunMode LoadBalance
    RNGMode Random123
    BaseSeed {base_seed}
    Dt 0.025
    Duration {duration}
    
}}\n
"""

pulsestimulus_template = """\
Stimulus {stim_name}
{{
    Mode Current
    Pattern Pulse
    AmpStart {amp}
    AmpEnd {amp}
    Frequency {freq}
    Width {width}
    Delay {delay}
    Duration {duration}
}}\n
"""

inject_template = """\
StimulusInject {inj_name}
{{
    Stimulus {stim_name}
    Target {target}
}}\n
"""


class BCWriter(object):
    """Small class to write BlueConfigs for BGLibPy"""

    def __init__(self, circuit_config, duration=1000, target="Mosaic", target_file=None, base_seed=1909):
        """Constructor with basic circuit info"""
        circuit = Circuit(circuit_config)
        self.duration = duration
        self.target = target
        self.base_seed = base_seed
        self.stimuli = []
        self.injections = []
        # Get circuit configuration
        self.morphology_path = circuit.config["morphologies"]
        self.metype_path = circuit.config["emodels"]
        self.mecombo_file = circuit.config["mecombo_info"]
        self.circuit_path = circuit.config["segment_index"]  # this is a bit hacky...
        self.nrn_path = circuit.config["connectome"]
        self.target_file = "" if target_file is None else target_file

    def add_pulsestimulus(self, stim_name, amp, freq, width, delay, duration):
        """Adds Stimulus block with given parameters to the BlueConfig"""
        self.stimuli.append(pulsestimulus_template.format(stim_name=stim_name,
                                                          amp=amp, freq=freq,
                                                          width=width, delay=delay,
                                                          duration=duration))

    def add_injection(self, inj_name, stim_name, target):
        """Adds StimulusInject block to BlueConfig"""
        self.injections.append(inject_template.format(inj_name=inj_name, stim_name=stim_name,
                                                      target=target))

    def write(self, output_path):
        """Write BlueConfig"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        run_default = self._generate_run_default(output_path)
        bc_path = os.path.join(output_path, "BlueConfig")
        with open(bc_path, "w") as f:
            f.write(run_default)
            f.write("".join(self.stimuli))
            f.write("".join(self.injections))
        return bc_path

    def _generate_run_default(self, output_path):
        """Fills in template with basic circuit info"""
        return run_default_template.format(morphology_path=self.morphology_path,
                                           metype_path=self.metype_path,
                                           mecombo_file=self.mecombo_file,
                                           circuit_path=self.circuit_path,
                                           nrn_path=self.nrn_path,
                                           base_seed=self.base_seed,
                                           output_path=output_path,
                                           target_file=self.target_file,
                                           target=self.target,
                                           duration=self.duration)

