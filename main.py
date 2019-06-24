
import os
import argparse

from even_handed.even_handed import even_handed
from even_handed.reaction_coord import create_reaction_coord


def main():
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


if __name__ == "__main__":
    main()
