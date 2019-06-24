"""
even_handed.py
Implementation of the even-handed subsystem selection for projection-based embedding.

Handles the primary functions
"""

import os
import numpy as np
import argparse
import warnings

from reaction_coord import *

def calculate_overlap_metric(mol1, mol2, sub_a_even_handed):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    if mol1.nocc != mol2.nocc:
        warnings.warn('Number of occupied orbitals differ between coordinates.')

    if mol1.nao != mol2.nao:
        warnings.warn('Number of atomic orbitals differ between coordinates.')

    # Calculate S1^1/2
    s_eig1, s_vec1 = np.linalg.eig(mol1.s_mat)
    tmp_s_sqrt_mat1 = np.zeros([mol1.nao,mol1.nao])
    for i in range(mol1.nao):
        tmp_s_sqrt_mat1[:,i] = s_vec1[:,i] * np.sqrt(s_eig1[i])
    s_sqrt_mat1 = np.matmul(tmp_s_sqrt_mat1, s_vec1.T)

    # Calculate S2^1/2
    s_eig2, s_vec2 = np.linalg.eig(mol2.s_mat)
    tmp_s_sqrt_mat2 = np.zeros([mol2.nao,mol2.nao])
    for i in range(mol2.nao):
        tmp_s_sqrt_mat2[:,i] = s_vec2[:,i] * np.sqrt(s_eig2[i])
    s_sqrt_mat2 = np.matmul(tmp_s_sqrt_mat2, s_vec2.T)

    # Calculate S^k,k+1 matrix
    if mol1.nao == mol2.nao:
        # Original way (from J. Chem. Phys. 2018, 149 (14), 144101.)
        s12 = np.matmul(s_sqrt_mat1, s_sqrt_mat2)
    elif mol1.nao < mol2.nao:
        # s12 is square with dimensions mol1.nao x mol1.nao
        s12 = np.matmul(s_sqrt_mat1, s_sqrt_mat2[mol1.nao:,mol1.nao:])
    elif mol1.nao > mol2.nao:
        # s12 is square with dimensions mol2.nao x mol2.nao
        s12 = np.matmul(s_sqrt_mat1[mol2.nao:,mol2.nao:], s_sqrt_mat2)

    # Calculate L^k * S^k,k+1 * L^k+1 matrix
    if mol1.nao == mol2.nao:
        # Original way (from J. Chem. Phys. 2018, 149 (14), 144101.)
        # s12, mol1.lmo_mat and mol2.lmo_mat are square
        # lmo1_s12_lmo2 is square nocc1 x nocc2 b/c nocc1 == nocc2
        s12_lmo2 = np.dot(s12, mol2.lmo_mat)
        lmo1_s12_lmo2 = np.dot(mol1.lmo_mat.T, s12_lmo2)
    elif mol1.nao < mol2.nao:
        # s12, and mol1.lmo_mat is square
        # mol2.lmo_mat is rectangular (AO dimension is reduced to mol1.nao)
        # lmo1_s12_lmo2 is rectangular nocc1 x nocc2
        s12_lmo2 = np.dot(s12, mol2.lmo_mat[mol1.nao:,:])
        lmo1_s12_lmo2 = np.dot(mol1.lmo_mat.T, s12_lmo2)
    elif mol1.nao > mol2.nao:
        # s12, and mol2.lmo_mat is square
        # mol1.lmo_mat is rectangular (AO dimension is reduced to mol2.nao)
        # lmo1_s12_lmo2 is rectangular nocc1 x nocc2
        s12_lmo2 = np.dot(s12, mol2.lmo_mat)
        lmo1_s12_lmo2 = np.dot(mol1.lmo_mat.T, s12_lmo2)

    # Square each element in the matrix
    tmp_overlap_metric = np.square(lmo1_s12_lmo2)

    if sub_a_even_handed:
        # Original way (from J. Chem. Phys. 2018, 149 (14), 144101.)
        # Even handedly select subsystem A
        # Zero out rows that correspond to orbs in sub B in geom 1
        for i in range(1,mol1.nocc+1):
            if i not in mol1.sub_a_indices:
                tmp_overlap_metric[i-1,:] = 0.0
    else:
        # Even handedly select subsystem B
        # Zero out rows that correspond to orbs in sub A in geom 1
        for i in range(1,mol1.nocc+1):
            if i not in mol1.sub_b_indices:
                tmp_overlap_metric[i-1,:] = 0.0

    # Compress down to 1-dimension where rows are compressed and added
    overlap_metric = np.sum(tmp_overlap_metric, axis = 0)
    return overlap_metric[:mol2.nocc]


# Perform even-handed algorithm
def even_handed(reaction_coord, sub_a_even_handed=True):
    """
    Placeholder function to show example docstring (NumPy format)

    Replace this function and doc string for your own project

    Parameters
    ----------
    with_attribution : bool, Optional, default: True
        Set whether or not to display who the quote is from

    Returns
    -------
    quote : str
        Compiled string including quote and optional attribution
    """

    print('Before even-handed embedding')
    print('Subsystem A orbitals')
    for item in reaction_coord:
        print(str(item.root.split('/')[-1]) + ': ' + str(len(item.sub_a_indices)) + ' orbitals' )

    print('Subsystem B orbitals')
    for item in reaction_coord:
        print(str(item.root.split('/')[-1]) + ': ' + str(len(item.sub_b_indices)) + ' orbitals' )

    if len(reaction_coord) == 1:
        raise RuntimeError('Need more than one coordinate to perform even-handed embedding.')

    # First pass through geometries
    for i in range(len(reaction_coord)-1):
        # Use np.argsort to return array of indices that index data in a sorted order
        # Sorted in ascending (small values first). Only care about the last nocc_a1 values.
        overlap_metric2 = calculate_overlap_metric(reaction_coord[i], reaction_coord[i+1], sub_a_even_handed)
        sorted_indices2 = np.argsort(overlap_metric2)
        nocc2 = reaction_coord[i+1].nocc

        # Added 1 to all indices so they properly match the orbital indices
        add1_sorted_indices2 = [x+1 for x in sorted_indices2]

        if sub_a_even_handed:
            # Select the nocc_a1 largest overlap_metric2 values for even_handed_set2
            # And unionize the two sets.
            nocc_a1 = reaction_coord[i].nocc_a
            even_handed_set2 = set(add1_sorted_indices2[-1*nocc_a1:])
            sub_a_indices2 = reaction_coord[i+1].sub_a_indices
            updated_sub_a_indices2 = sub_a_indices2.union(even_handed_set2)
            updated_sub_b_indices2 = set([x for x in range(1,nocc2+1) if x not in updated_sub_a_indices2])
            #updated_sub_b_indices2 = reaction_coord[i+1].sub_b_indices - updated_sub_a_indices2
        else:
            # Select the nocc_b1 largest overlap_metric2 values for even_handed_set2
            nocc_b1 = reaction_coord[i].nocc_b
            even_handed_set2 = set(add1_sorted_indices2[-1*nocc_b1:])
            sub_b_indices2 = reaction_coord[i+1].sub_b_indices
            updated_sub_b_indices2 = sub_b_indices2.intersection(even_handed_set2)
            updated_sub_a_indices2 = set([x for x in range(1,nocc2+1) if x not in updated_sub_b_indices2])
            #updated_sub_a_indices2 = reaction_coord[i+1].sub_a_indices - updated_sub_b_indices2

        # Update reaction_coord info for i+1
        reaction_coord[i+1].nocc_a = len(updated_sub_a_indices2)
        reaction_coord[i+1].sub_a_indices = updated_sub_a_indices2
        reaction_coord[i+1].nocc_b = len(updated_sub_b_indices2)
        reaction_coord[i+1].sub_b_indices = updated_sub_b_indices2

        # Add overlap_metric data to the reaction_coord dictionary
        reaction_coord[i+1].overlap_metric = overlap_metric2

    # Second pass through geometries in reverse
    for i in reversed(range(len(reaction_coord)-1)):
        # Use np.argsort to return array of indices that index data in a sorted order
        # Sorted in ascending (small values first). Only care about the last nocc_a2 values.
        overlap_metric1 = calculate_overlap_metric(reaction_coord[i+1], reaction_coord[i], sub_a_even_handed)
        sorted_indices1 = np.argsort(overlap_metric1)
        nocc1 = reaction_coord[i].nocc

        # Added 1 to all indices so they properly match the orbital indices
        add1_sorted_indices1 = [x+1 for x in sorted_indices1]

        if sub_a_even_handed:
            # Select the nocc_a2 largest overlap_metric1 values for even_handed_set1
            # and unionize the two sets.
            nocc_a2 = reaction_coord[i+1].nocc_a
            even_handed_set1 = set(add1_sorted_indices1[-1*nocc_a2:])
            sub_a_indices1 = reaction_coord[i].sub_a_indices
            updated_sub_a_indices1 = sub_a_indices1.union(even_handed_set1)
            updated_sub_b_indices1 = set([x for x in range(1,nocc1+1) if x not in updated_sub_a_indices1])
            #updated_sub_b_indices1 = reaction_coord[i].sub_b_indices - updated_sub_a_indices1
        else:
            # Select the nocc_b2 largest overlap_metric1 values for even_handed_set1
            nocc_b2 = reaction_coord[i+1].nocc_b
            even_handed_set1 = set(add1_sorted_indices1[-1*nocc_b2:])
            sub_b_indices1 = reaction_coord[i].sub_b_indices
            updated_sub_b_indices1 = sub_b_indices1.intersection(even_handed_set1)
            updated_sub_a_indices1 = set([x for x in range(1,nocc1+1) if x not in updated_sub_b_indices1])
            #updated_sub_a_indices1 = reaction_coord[i].sub_a_indices - updated_sub_b_indices1

        # Update reaction_coord info for i
        reaction_coord[i].nocc_a = len(updated_sub_a_indices1)
        reaction_coord[i].sub_a_indices = updated_sub_a_indices1
        reaction_coord[i].nocc_b = len(updated_sub_b_indices1)
        reaction_coord[i].sub_b_indices = updated_sub_b_indices1

        # Add overlp_metric data to the reaction_coord dictionary
        reaction_coord[i].overlap_metric = overlap_metric1

    print('')
    print('After even-handed embedding')
    print('Subsystem A orbitals')
    for item in reaction_coord:
        print(str(item.root.split('/')[-1]) + ': ' + str(len(item.sub_a_indices)) + ' orbitals' )

    print('Subsystem B orbitals')
    for item in reaction_coord:
        print(str(item.root.split('/')[-1]) + ': ' + str(len(item.sub_b_indices)) + ' orbitals' )

    print('')
    print('Even-handed Subsystem A Orbital Indices')
    print('Subsystem A indices')
    for item in reaction_coord:
        print(item.root.split('/')[-1] + ': ' +str(list(item.sub_a_indices)))

    print('Subsystem B indices')
    for item in reaction_coord:
        print(item.root.split('/')[-1] + ': ' +str(list(item.sub_b_indices)))

        #orb_fname = os.path.join(root,str(root.split('/')[-1]) + '_sub_a.txt')
        #with open(orb_fname,'w') as file1:
        #    file1.write('Even-handed Subsystem A Orbital Indices\n')
        #    file1.write(str(list(orbital_indices)))

    print('')
    print('Overlap Metric for each Molecular Orbital')
    for item in reaction_coord:
        print(item.root.split('/')[-1] + ': ' +str(list(item.overlap_metric)))

        #overlap_fname = os.path.join(root,str(root.split('/')[-1]) + '_overlap_metric.txt')
        #with open(overlap_fname,'w') as file1:
        #    file1.write('Overlap Metric\n')
        #    for i in range(len(overlap_metric[0])):
        #        line = "{:3d} {:8f}\n".format(overlap_metric[0][i], overlap_metric[1][i])
        #        file1.write(line)

    # Do some sanity checks to make sure the even handed procedure worked.
    if sub_a_even_handed:
        for i in range(len(reaction_coord)-1):
            assert(reaction_coord[i].nocc_a == reaction_coord[i+1].nocc_a), 'Number of orbitals in subsystem A is not consistent between geometries.'
    else:
        for i in range(len(reaction_coord)-1):
            assert(reaction_coord[i].nocc_b == reaction_coord[i+1].nocc_b), 'Number of orbitals in subsystem B is not consistent between geometries.'

    return reaction_coord


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', type=str, help='Top level directory that contains output files to be used for even-handed.', required=True)
    parser.add_argument('-n', type=str, help='Name of output file. This relies on output files having the same name.', required=True)
    parser.add_argument('--b', dest='sub_a_even_handed', action='store_false', help='If this option is given subsystem B is made even-handed.')
    parser.set_defaults(sub_a_even_handed=True)

    args = parser.parse_args()
    start_dir = os.path.join(os.getcwd(), args.d)
    name_of_output_file = args.n

    # Assemble reaction coordinate
    reaction_coord = create_reaction_coord(start_dir, name_of_output_file)

    # Perform even-handed embedding
    even_handed_reac_coord = even_handed(reaction_coord, args.sub_a_even_handed)
