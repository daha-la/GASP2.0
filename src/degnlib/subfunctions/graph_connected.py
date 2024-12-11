#!/usr/bin/env python3
import argparse
from Bio.File import as_handle
from degnutil import argument_parsing as argue, pandas_util as panda


def get_parser():
    parser = argparse.ArgumentParser(
        description="Get groups of connected nodes.")
    argue.inoutfile(parser, help="Table where first and second column are node IDs.")
    argue.delimiter(parser)
    argue.header(parser)

    return parser


def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))


def connected(nodeAs, nodeBs):
    groups = []
    for nodeA, nodeB in zip(nodeAs, nodeBs):
        edge = {nodeA, nodeB}
        # add edge nodes to a group if they intersect group nodes, 
        # otherwise make a new group
        match = False
        for i, group in enumerate(groups):
            if nodeA in group or nodeB in group:
                groups[i] |= edge
                match = True
        if not match: groups.append(edge)

    # combine all groups that intersect
    while len(groups) > 1:
        inter = _intersect(groups)
        if inter is None: return groups
        groups[inter[0]] |= groups[inter[1]]
        del groups[inter[1]]
        # start over every time there is intersections found


def _intersect(groups):
    """
    Smaller helper function that finds the first pair of sets with intersection.
    :param groups: list of sets
    :return: two element tuple with indices of the intersecting sets.
    """
    for i in range(len(groups) - 1):
        for j in range(i + 1, len(groups)):
            if len(groups[i] & groups[j]) > 0: return i, j


def main(args):
    args = parse_args(args)
    table = panda.read_pandas(args.infile, args.delimiter, args.header)
    groups = connected(table.iloc[:,0], table.iloc[:,1])
    
    outdelim = args.delimiter if args.delimiter is not None else '\t'
    with as_handle(args.outfile, 'w') as fh:
        for group in groups: fh.write(outdelim.join(group) + '\n')


if __name__ == '__main__':
    main(get_parser().parse_args())
