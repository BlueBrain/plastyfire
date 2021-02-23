"""
Adds plasticity releated parameters to the preloaded recipe DataFrame
author: András Ecker, last update: 02.2021
based on Giuseppe Chindemi's scripts up until 09.2020
"""

import pandas as pd
from scipy import stats

# L5 TPC NMDA/AMPA ratio (different from the xml recipe)
l5_nmda = 0.55  # calibrated by Giuseppe based on Markram et al. 1997
# parameters related to GluSynapse specific spine volume
spinevol_dist = stats.lognorm
spinevol_params = (0.865, 0, 0.061)  # Christian's fit on Ruth's data
gbasal_dist = stats.lognorm
gbasal_params = (0.886, 0, 1.106)  # Giuseppe's fit on L5 basal (v6)
spinevol_sf = spinevol_dist.mean(*spinevol_params) / gbasal_dist.mean(*gbasal_params)  # spinevol scaling factor
# correlations between U_SE, NRRP, gsyn, spine volume (see Chindemi et al. 2020, bioRxiv eqs: (23)-(26))
u_gsyn_r = 0.9  # assumption by Giuseppe
gsyn_nrrp_r = 0.9   # from Harris and Stevens 1989 (not really NRRP, but total N)
spinevol_nrrp_r = 0.92  # from Harris and Stevens 1989 (not really NRRP, but total N)
spinevol_gsyn_r = 0.88  # from Arellano et al. 2007


def _update_L5_NMDA_ratio(df):
    """Updates NMDA/AMPA ratio for L5 TPCs"""
    df_tmp = df[df.index.get_level_values(0).str.contains("L5_TPC:.*")]
    idx = df_tmp[df_tmp.index.get_level_values(1).str.contains("L5_TPC:.*")].index
    df.loc[idx, "gsynSRSF"] = l5_nmda


def _add_spinevol(df):
    """Adds GluSynapse specific spine volume to the df (based on pathway specific gsyn and NMDA ratio)"""
    df["spinevol"] = spinevol_sf * (1 + df["gsynSRSF"]) * df["gsyn"]
    df["spinevolSD"] = spinevol_sf * (1 + df["gsynSRSF"]) * df["gsynSD"]


def _add_rs(df):
    """Adds parameter generation (see `plastyfire/epg.py`) specific correlations"""
    df["u_gsyn_r"] = u_gsyn_r
    df["gsyn_nrrp_r"] = gsyn_nrrp_r
    df["spinevol_nrrp_r"] = spinevol_nrrp_r
    df["spinevol_gsyn_r"] = spinevol_gsyn_r


def _add_dists(df):
    """Adds distributions of the various parameters to the df (used by `plastyfire/epg.py`)"""
    df["gsynDist"] = "gamma"
    df["nrrpDist"] = "poisson"
    df["uDist"] = "beta"
    df["dDist"] = "gamma"
    df["fDist"] = "gamma"
    df["spinevolDist"] = "gamma"


def update_df(df):
    """Updates preloaded (with xml recipe) DataFrame (see `plastyfire/xmlrecipe.py`)
    to have all `plastyfire/epg.py` and GluSynapse specific parameters"""
    _update_L5_NMDA_ratio(df)
    _add_spinevol(df)
    _add_rs(df)
    _add_dists(df)


if __name__ == "__main__":

    recipe_in = "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/base_xml_recipe.csv"
    recipe_out = "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/recipe.csv"

    df = pd.read_csv(recipe_in, index_col=[0, 1])
    update_df(df)
    df.to_csv(recipe_out)


