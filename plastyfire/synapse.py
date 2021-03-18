"""
Class that represents a synapse in BGLibPy
original code by Werner van Geit and Anil Tuncel
+ GluSynapse compatibility by Giuseppe Chindemi, Alberto Antonetti and András Ecker
@remarks Copyright (c) BBP/EPFL 2012; All rights reserved.
         Do not distribute without further notice.
"""

import numpy as np
import bglibpy
from bglibpy.tools import printv


class Synapse(object):
    """ Class that represents a synapse in BGLibPy """

    def __init__(self, cell, location, syn_id, syn_description, connection_parameters,
                 base_seed, popids=None, extracellular_calcium=None):
        """
        Constructor
        Parameters
        ----------
        cell: Cell - Cell that contains the synapse
        location: float in [0, 1] - Location on the section this synapse is placed
        syn_id: (string,integer) - Synapse identifier
        syn_description: list of floats - Parameters of the synapse
        connection_parameters: list of floats - Parameters of the connection
        base_seed: float - Base seed of the simulation, the seeds for this synapse will be derived from this
        popids: 2 ints - population IDs of source and target populations (if None set to 0, 0)
        extracellular_calcium: float - extracellular Ca++ concentration to scale release prob.
        """
        self.persistent = []
        self.cell = cell
        self.post_gid = cell.gid
        self.syn_id = syn_id
        # self.projection, self.syn_id = self.syn_id  # only in bglibpy>=4.1
        self.projection = ''
        self.syn_description = syn_description
        self.connection_parameters = connection_parameters
        self.hsynapse = None
        self.extracellular_calcium = extracellular_calcium
        if popids is not None:
            self.source_popid, self.target_popid = popids
        else:
            self.source_popid, self.target_popid = 0, 0  # Default in Neurodamus
        # pylint: disable = C0103
        self.pre_gid = int(syn_description[0])
        # self.delay = syn_description[1]
        post_sec_id = syn_description[2]
        self.isec = int(post_sec_id)
        post_seg_id = syn_description[3]
        self.ipt = post_seg_id
        post_seg_distance = syn_description[4]
        self.syn_offset = post_seg_distance
        # weight = syn_description[8]
        self.syn_D = syn_description[10]
        self.syn_F = syn_description[11]
        self.syn_DTC = syn_description[12]
        self.syn_type = int(syn_description[13])
        if cell.rng_settings is None:
            self.rng_setting = bglibpy.RNGSettings(mode="Compatibility", base_seed=base_seed)
        else:
            if base_seed is not None:
                raise Exception("Synapse: base_seed and RNGSettings cant be used together")
            self.rng_settings = cell.rng_settings
        if len(syn_description) >= 18:
            if syn_description[17] <= 0:
                raise ValueError("Value smaller than 0.0 found for Nrrp: %s at synapse %d in gid %d" %
                                 (syn_description[17], self.syn_id, self.cell.gid))
            if syn_description[17] != int(syn_description[17]):
                raise ValueError("Non-integer value for Nrrp found: %s at synapse %d in gid %d" %
                                 (syn_description[17], self.syn_id, self.cell.gid))
            self.Nrrp = int(syn_description[17])
            if len(syn_description) == 20:
                self.u_hill_coefficient = syn_description[18]
                self.conductance_ratio = syn_description[19]
            else:
                self.u_hill_coefficient = None
                self.conductance_ratio = None
        else:
            self.Nrrp = None
            self.u_hill_coefficient = None
            self.conductance_ratio = None
        self.u_scale_factor = self.calc_u_scale_factor(self.u_hill_coefficient, self.extracellular_calcium)
        self.syn_U = self.u_scale_factor * syn_description[9]
        self.post_segx = location
        # pylint: enable = C0103

        if self.is_inhibitory():
            raise NotImplementedError()  # to be used only with GluSynapse
        elif self.is_excitatory():
            self.mech_name = "GluSynapse"
            self.hsynapse = bglibpy.neuron.h.GluSynapse(self.post_segx, sec=self.cell.get_hsection(post_sec_id))
            self.hsynapse.tau_d_AMPA = self.syn_DTC
            self.hsynapse.Use0_TM = abs(self.syn_U)
            self.hsynapse.Dep_TM = abs(self.syn_D)
            self.hsynapse.Fac_TM = abs(self.syn_F)
            if self.Nrrp is not None:
                self.hsynapse.Nrrp_TM = self.Nrrp
        else:
            raise Exception("Synapse not inhibitory or excitatory")

        if self.rng_settings.mode == "Random123":
            self.randseed1 = self.cell.gid + 250
            self.randseed2 = self.syn_id + 100
            self.randseed3 = self.source_popid * 65536 + self.target_popid + self.rng_settings.synapse_seed + 300
            self.hsynapse.setRNG(self.randseed1, self.randseed2, self.randseed3)
        else:
            rndd = bglibpy.neuron.h.Random()
            self.randseed1 = syn_id * 100000 + 100
            if self.rng_settings.mode == "Compatibility":
                self.randseed2 = self.cell.gid + 250 + self.rng_settings.base_seed
            elif self.rng_settings.mode == "UpdatedMCell":
                self.randseed2 = self.source_popid * 16777216 + self.cell.gid + \
                                 250 + self.rng_settings.base_seed + self.rng_settings.synapse_seed
            else:
                raise ValueError(
                    "Synapse: unknown RNG mode: %s" % self.rng_settings.mode)
            self.randseed3 = None  # Not used in this case
            rndd.MCellRan4(self.randseed1, self.randseed2)
            rndd.uniform(0, 1)
            self.hsynapse.setRNG(rndd)
            self.persistent.append(rndd)

        self.hsynapse.synapseID = self.syn_id

        self.synapseconfigure_cmds = []
        # hoc exec synapse configure blocks
        if "SynapseConfigure" in self.connection_parameters:
            for cmd in self.connection_parameters["SynapseConfigure"]:
                self.synapseconfigure_cmds.append(cmd)
                cmd = cmd.replace("%s", "\n%(syn)s")
                hoc_cmd = cmd % {"syn": self.hsynapse.hname()}
                hoc_cmd = "{%s}" % hoc_cmd
                bglibpy.neuron.h(hoc_cmd)

    def is_inhibitory(self):
        """Returns True if synapse is inhibitory"""
        return self.syn_type < 100

    def is_excitatory(self):
        """Returns True if synapse is excitatory"""
        return self.syn_type >= 100

    def delete(self):
        """Delete the connection"""
        if hasattr(self, "persistent"):
            for persistent_object in self.persistent:
                del persistent_object

    @staticmethod
    def calc_u_scale_factor(u_hill_coefficient, extracellular_calcium):
        """Calculates scale factor of release probability based on Hill function"""
        if extracellular_calcium is None or u_hill_coefficient is None:
            return 1.0

        def hill(extracellular_calcium, y, K_half):
            return y * extracellular_calcium ** 4 / (K_half ** 4 + extracellular_calcium ** 4)

        def constrained_hill(K_half):
            y_max = (K_half ** 4 + 16) / 16
            return lambda x: hill(x, y_max, K_half)

        def f_scale(x, y):
            return constrained_hill(x)(y)

        u_scale_factor = np.vectorize(f_scale)(u_hill_coefficient, extracellular_calcium)
        printv("Scaling synapse Use with u_hill_coeffient %f, extra_cellular_calcium %f with a factor of %f" %
               (u_hill_coefficient, extracellular_calcium, u_scale_factor), 50)

        return u_scale_factor

    @property
    def info_dict(self):
        """Convert the synapse info to a dict from which it can be reconstructed"""
        synapse_dict = dict()
        synapse_dict["synapse_id"] = self.syn_id
        synapse_dict["pre_cell_id"] = self.pre_gid
        synapse_dict["post_cell_id"] = self.post_gid
        synapse_dict["post_sec_id"] = self.isec
        synapse_dict["post_sec_name"] = bglibpy.neuron.h.secname(sec=self.cell.get_hsection(self.isec)).split(".")[1]
        synapse_dict["post_segx"] = self.post_segx
        synapse_dict["mech_name"] = self.mech_name
        synapse_dict["syn_type"] = self.syn_type
        synapse_dict["randseed1"] = self.randseed1
        synapse_dict["randseed2"] = self.randseed2
        synapse_dict["randseed3"] = self.randseed3
        synapse_dict["synapseconfigure_cmds"] = self.synapseconfigure_cmds
        # Parameters of the mod mechanism
        synapse_dict["synapse_parameters"] = {}
        synapse_dict["synapse_parameters"]["Use"] = self.hsynapse.Use0_TM
        synapse_dict["synapse_parameters"]["Dep"] = self.hsynapse.Dep_TM
        synapse_dict["synapse_parameters"]["Fac"] = self.hsynapse.Fac_TM
        if synapse_dict["mech_name"] == "GluSynapse":
            synapse_dict["synapse_parameters"]["tau_d_AMPA"] = self.hsynapse.tau_d_AMPA
        else:
            raise Exception("info_dict only works with Glusynapse!")
        # newly added parameters
        synapse_dict["synapse_parameters"]["conductance_ratio"] = self.conductance_ratio
        synapse_dict["synapse_parameters"]["u_scale_factor"] = self.u_scale_factor
        synapse_dict["synapse_parameters"]["extracellular_calcium"] = self.extracellular_calcium
        return synapse_dict

    def __del__(self):
        """Destructor"""
        self.delete()



