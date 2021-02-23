"""
Reads (EXC-EXC) pathway specific synapse parameters from xml recipe to a MultiIndex pandas DataFrame
(using spykfunc's python API to read the xml)
author: András Ecker, last update: 02.2021
"""

import pandas as pd
from recipe import Recipe

# v7 EXC mtypes (once recipe is pip installable replace this with a bluepy call ...)
mtypes = ["L2_IPC", "L2_TPC:A", "L2_TPC:B", "L3_TPC:A", "L3_TPC:C",
          "L4_SSC", "L4_TPC", "L4_UPC",
          "L5_TPC:A", "L5_TPC:B", "L5_TPC:C", "L5_UPC",
          "L6_BPC", "L6_HPC", "L6_IPC", "L6_TPC:A", "L6_TPC:C", "L6_UPC"]
# synapse properties stored in (v7 circuit style) xml recipe
properties = ["gsyn", "gsynSD", "nrrp", "dtc", "dtcSD",
              "u", "uSD", "d", "dSD", "f", "fSD",
              "gsynSRSF", "uHillCoefficient"]


def init_df():
    """Initializes an empty MultiIndex DataFrame (to be filled with values read from the xml recipe)"""
    mi = pd.MultiIndex.from_product([mtypes, mtypes], names=["pre_mtype", "post_mtype"])
    df_tmp = mi.to_frame()
    df = df_tmp.drop(columns=["pre_mtype", "post_mtype"])  # stupid pandas ...
    for prop in properties:
        df[prop] = 0.0
    df["synapse_type"] = ""
    return df


def _fix_regexp(s):
    """Updates regexps used in the xml recipe to work with python's `re`"""
    return s.replace('*', '.*')


def rule_to_idx(rule, df):
    """Gets df MultiIndex from recipe rule (only EXC-EXC)"""
    if rule.fromSClass == "EXC" and rule.toSClass == "EXC":
        return df.index
    elif rule.fromMType != "*" and rule.toMType != "*":
        pre_mtype = _fix_regexp(rule.fromMType)
        post_mtype = _fix_regexp(rule.toMType)
        df_tmp = df[df.index.get_level_values(0).str.contains(pre_mtype)]
        return df_tmp[df_tmp.index.get_level_values(1).str.contains(post_mtype)].index
    else:
        return []


def synapse_type_to_idx(synapse_type, df):
    """Gets df MultiIndex from synapse_type"""
    return df[df["synapse_type"] == synapse_type].index


def recipe_to_df(recipe, df):
    """Reads in (EXC-EXC) pathway specific values to `pandas.DataFrame`"""

    # get synapse types for all pathways
    for rule in recipe.synapse_properties.rules:
        idx = rule_to_idx(rule, df)
        if len(idx):
            df.loc[idx, "synapse_type"] = rule.type
    # based on the synapse types fill in values
    for synapse_class in recipe.synapse_properties.classes:
        idx = synapse_type_to_idx(synapse_class.id, df)
        if len(idx):
            for prop in properties:
                df.loc[idx, prop] = getattr(synapse_class, prop)


if __name__ == "__main__":

    recipe_in = "/gpfs/bbp.cscs.ch/project/proj83/home/ecker/CircuitBuildRecipe/inputs/4_synapse_generation/ALL/builderRecipeAllPathways.xml"
    recipe_out = "/gpfs/bbp.cscs.ch/project/proj96/circuits/plastic_v1/base_xml_recipe.csv"
    recipe = Recipe(recipe_in)

    df = init_df()
    recipe_to_df(recipe, df)
    df.to_csv(recipe_out)

