"""
reaction_coord.py
"""

import os
from pydantic import BaseModel
from typing import List, Any

from .output_parser import read_embed_output, read_matrop_matrix


class EvenHandedInfo(BaseModel):
    lmo_mat: Any = None
    s_mat: Any = None
    nocc: int = None
    nao: int = None
    nocc_a: int = None
    nocc_b: int = None
    sub_a_indices: set = None
    sub_b_indices: set = None
    overlap_metric: List[int] = None
    root: str = None


# Returns a list of EvenHandedInfo objects
def create_reaction_coord(start_dir, name_of_output_file):
    """
    This compiles information about each molecule and stores it in a list.

    The order of the molecules in the returned list is considered to be the order of the reaction coordinate by this
    program. This order is determined by sorting the subdirectories of the provided top-level directory (start_dir).
    The sort is performed by the sort() function.

    Parameters
    ----------
    start_dir : str
        The top-level directory that contains all molecules included in the reaction coordinate as subdirectories.

    name_of_output_file : str
        Name of the Molpro output file for each molecule. This only works if the output file name is the same for each
        molecule.

    Returns
    -------
    reaction_coord : list
        List of EvenHandedInfo objects. Each EvenHandedInfo object corresponds to one molecule.
    """

    # Compile all necessary information from each geometry along the reaction coordinate.
    reaction_coord = []
    for root, dirs, files in os.walk(os.path.abspath(start_dir)):
        # Sort dirs in place so directories are recursed in sorted order
        dirs.sort()
        for filename in files:
            if filename == name_of_output_file:
                embed_output_dict = read_embed_output(os.path.join(root,filename))
                nao = embed_output_dict['nao']
                lmo_mat = read_matrop_matrix(os.path.join(root,'orbs_mat.txt'), nao)
                s_mat = read_matrop_matrix(os.path.join(root,'s_mat.txt'), nao)
                reac_dict = {
                    **embed_output_dict,
                    'lmo_mat': lmo_mat,
                    's_mat': s_mat,
                    'overlap_metric': None,
                    'root': root
                     }
                reaction_coord.append(EvenHandedInfo(**reac_dict))
    return reaction_coord
