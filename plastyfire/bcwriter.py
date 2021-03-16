"""
BlueConfig writer (since BGLibPy doesn't work w/o a BlueConfig)
authors: Giuseppe Chindemi (09.2020)
+minor modifications for new bluepy version by András Ecker (01.2021)
"""

import os
from bluepy.v2 import Circuit


class BCWriter(object):
    """Small class to write BlueConfigs for BGLibPy"""

    def __init__(self, circuit_config, duration, target, target_file, base_seed):
        """Constructor with basic circuit info"""
        c = Circuit(circuit_config)
        self.duration = duration
        self.target = target
        self.target_file = "" if target_file is None else target_file
        self.base_seed = base_seed
        # Get circuit configuration
        self.morphology_path = c.config["morphologies"]
        self.metype_path = c.config["emodels"]
        self.mecombo_file = c.config["mecombo_info"]
        self.circuit_path = c.config["segment_index"]  # this is a bit hacky...
        self.nrn_path = c.config["connectome"]

    def _generate_run_default(self, output_path):
        """Fills in template with basic circuit info"""
        with open("templates/BlueConfig.tmpl", "r") as f:
            templ = f.read()
        return templ.format(morphology_path=self.morphology_path,
                            metype_path=self.metype_path,
                            mecombo_file=self.mecombo_file,
                            circuit_path=self.circuit_path,
                            nrn_path=self.nrn_path,
                            base_seed=self.base_seed,
                            output_path=output_path,
                            target_file=self.target_file,
                            target=self.target,
                            duration=self.duration)

    def write(self, output_path):
        """Write BlueConfig"""
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        run_default = self._generate_run_default(output_path)
        bc = os.path.join(output_path, "BlueConfig")
        with open(bc, "w") as f:
            f.write(run_default)
        return bc



