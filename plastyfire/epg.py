"""
Extra Parameter Generator
authors: Giuseppe Chindemi (12.2020)
+minor modifications and docs by András Ecker (01.2021)
"""

import numpy as np
import pandas as pd
from scipy import stats
from bluepy.v2.enums import Cell, Synapse
from neurom import NeuriteType


def _get_covariance_matrix(pathway_recipe):
    """
    Covariance matrix of synaptic parameters (see eq. (28) in Chindemi et al. 2020, bioRxiv)
    Note: D,F parameters of the TM model are not correlated with the rest
    """
    cov = np.eye(6)
    cov[0][3] = cov[3][0] = float(pathway_recipe["u_gsyn_r"]) * float(pathway_recipe["gsyn_nrrp_r"])
    cov[0][4] = cov[4][0] = float(pathway_recipe["u_gsyn_r"])
    cov[0][5] = cov[5][0] = float(pathway_recipe["u_gsyn_r"]) * float(pathway_recipe["spinevol_gsyn_r"])
    cov[3][4] = cov[4][3] = float(pathway_recipe["gsyn_nrrp_r"])
    cov[3][5] = cov[5][3] = float(pathway_recipe["spinevol_nrrp_r"])
    cov[4][5] = cov[5][4] = float(pathway_recipe["spinevol_gsyn_r"])
    return cov


def _get_distributions(distname, mu, sigma):
    """Takes mu and sigma (stored as mean and std in the xml recipe) and returns distributions used by Spykfunc"""
    if distname == "beta":
        a = -(mu*(mu**2 - mu + sigma**2)) / sigma**2
        b = ((mu - 1)*(mu**2 - mu + sigma**2)) / sigma**2
        dist = stats.beta(a, b)
    elif distname == "gamma":
        a = mu**2/sigma**2
        scale = sigma**2/mu
        dist = stats.gamma(a, loc=0, scale=scale)
    elif distname == "poisson":
        dist = stats.poisson(mu - 1, loc=1)
    else:
        raise NotImplementedError()
    return dist


def _normtodist(dist, value):
    """Converts random normal sample to a random sample from a given distribution (see `_get_distributions()`)"""
    unif_value = stats.norm.cdf(value)  # Convert to uniform
    dist_value = dist.ppf(unif_value)  # Convert to dist
    return dist_value


def _get_ltpltd_params(u, gsyn, k_u, k_gsyn):
    """Gets LTP/LTD parameters (randomly based on the release probability `u`)"""
    rho0 = stats.binom.rvs(1, u)
    if rho0 > 0.5:  # Potentiated synapse (see equations (8-9) and (17-18) in Chindemi et al. 2020, bioRxiv)
        u_d = np.power(u, 1/k_u)
        u_p = u
        gsyn_d = (1/k_gsyn)*gsyn
        gsyn_p = gsyn
    else:  # Depressed synapse (see equations (8-9) and (17-18) in Chindemi et al. 2020, bioRxiv)
        u_d = u
        u_p = np.power(u, k_u)
        gsyn_d = gsyn
        gsyn_p = k_gsyn*gsyn
    params = {"rho0_GB": rho0, "Use_d_TM": u_d, "Use_p_TM": u_p,
              "gmax_d_AMPA": gsyn_d, "gmax_p_AMPA": gsyn_p}
    return params


class ParamsGenerator(object):
    """Small class for generating synapse parameters with correlations"""

    def __init__(self, circuit, extra_recipe_path, k_u=0.2, k_gsyn=2):
        """Constructor that loads circuit and extra recipe parameters from csv"""
        self.k_u = k_u
        self.k_gsyn = k_gsyn
        self.circuit = circuit
        self.extra_recipe = pd.read_csv(extra_recipe_path, index_col=[0, 1])
        # Set ordered list of parameter names
        self.namelst = ["u", "d", "f", "nrrp", "gsyn", "spinevol"]
        self.paramlst = ["Use0_TM", "Dep_TM", "Fac_TM", "Nrrp_TM", "gmax0_AMPA", "volume_CR"]

    def generate_params(self, pre_gid, post_gid):
        """
        Generate parameters (both general U,D,F etc. and plasticity related) for synapses between `pre_gid` and `post_gid`
        using the same distributions, means and stds as in the original recipe but with correlation between parameters.
        Compared to Spykfunc this function generates different parameters to all synapses mediating the given connection
        in other words: inter-connection variability is *not* assumed to be zero
        """
        # Find pathway recipe
        pre_mtype = self.circuit.cells.get(pre_gid, Cell.MTYPE)
        post_mtype = self.circuit.cells.get(post_gid, Cell.MTYPE)
        pathway_recipe = self.extra_recipe.loc[pre_mtype, post_mtype]
        # Assemble synapse parameter distribution list
        distlst = [_get_distributions(pathway_recipe["%sDist" % name],
                   pathway_recipe["%s" % name], pathway_recipe["%sSD" % name]) for name in self.namelst]
        # Generate multivariate normal sample with prescribed correlations
        cov = _get_covariance_matrix(pathway_recipe)
        np.random.seed((pre_gid, post_gid))
        sample_normal = stats.multivariate_normal.rvs(cov=cov)

        # Generate parameters for each synapse (TODO: vectorize)
        syns = self.circuit.connectome.pair_synapses(pre_gid, post_gid, Synapse.POST_BRANCH_TYPE)
        syn_params = dict()
        for syn_id, branch_type in syns.items():
            # Convert random normal samples to desired distributions
            sample_params = map(_normtodist, distlst, sample_normal)
            params = dict(zip(self.paramlst, sample_params))
            # Add LTP / LTD params
            params.update(_get_ltpltd_params(params["Use0_TM"], params["gmax0_AMPA"], self.k_u, self.k_gsyn))
            # Add NMDA conductance
            params["gmax_NMDA"] = params["gmax0_AMPA"] * pathway_recipe["gsynSRSF"]
            # Add branch type
            if branch_type == NeuriteType.basal_dendrite:
                params["loc"] = "basal"
            elif branch_type == NeuriteType.apical_dendrite:
                params["loc"] = "apical"
            else:
                raise ValueError("Unknown neurite type")
            syn_params[syn_id] = params
        return syn_params

