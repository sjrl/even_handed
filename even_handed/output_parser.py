"""
output_parser.py
Functions to read molpro output files for embedding info and matrop matrix files.
"""

import numpy as np


def read_embed_output(molpro_output):
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

    nocc = None
    nocc_a = None
    nao = None
    sub_a_indices = None
    keep_reading = False

    with open(molpro_output, 'r') as file1:
        for lc, lines in enumerate(file1):
            fields = lines.split()

            if len(fields) == 0:
                continue

            # Read nocc
            if fields[:-1] == ['Total', 'number', 'of', 'electrons:']:
                # NOTE: Hard coded in closed shell by dividing by 2
                nocc = int(float(fields[-1])/2)

            # Read nocc_a
            if fields[:-1] == ['Number', 'of', 'active', 'electrons:']:
                # NOTE: Hard coded in closed shell by dividing by 2
                nocc_a = int(float(fields[-1])/2)

            # Read sub_a_indices
            if fields[:4] == ['MOs', 'in', 'active', 'region:']:
                keep_reading = True
                lc_start = lc
                sub_a_indices = set([int(x.split('.')[0]) for x in fields[4:]])
            elif keep_reading and lc > lc_start:
                if fields[:-1] == ['Total', 'number', 'of', 'electrons:'] or \
                        fields[:4] == ['MOs', 'in', 'frozen', 'region:']:
                    keep_reading = False
                    continue
                more_indices = set([int(x.split('.')[0]) for x in fields])
                sub_a_indices.update(more_indices)

            # Read nao
            if fields[:3] == ['NUMBER', 'OF', 'CONTRACTIONS:']:
                nao = int(fields[3])

    assert (nocc_a == len(sub_a_indices)), 'Number of A MOs does not match length of A MO indices.'

    # Determine sub_b_indices from sub_a_indices and total number of MOs
    sub_b_indices = set([x for x in range(1, nocc+1) if x not in sub_a_indices])
    nocc_b = len(sub_b_indices)

    assert(nocc_a + nocc_b == nocc), 'Number of A + B MOs does not match total number of occupied MOs.'

    embed_output_dict = {
        'nocc': nocc,
        'nao': nao,
        'nocc_a': nocc_a,
        'nocc_b': nocc_b,
        'sub_a_indices': sub_a_indices,
        'sub_b_indices': sub_b_indices,
         }

    return embed_output_dict


def read_matrop_matrix(matrop_output, nao):
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

    # Read matrop matrix
    matrix_list = []
    reading = False
    with open(matrop_output,'r') as file1:
        for lc, lines in enumerate(file1):
            fields = lines.split(',')
            # print(fields)

            # Determine when start and stop reading ORB matrix
            if fields[0] == 'BEGIN_DATA':
                reading = True
                continue
            elif fields[0] == 'END_DATA':
                reading = False

            if reading:
                if fields[0].split()[0] == '#':
                    continue
                else:
                    tmp_list = [float(x) for x in fields[:-1]]
                    matrix_list.extend(tmp_list)

    tmp_matrix = np.copy(matrix_list)
    matrix = np.reshape(tmp_matrix, [nao,nao])
    return matrix
