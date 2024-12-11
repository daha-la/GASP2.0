#!/usr/bin/env python3
import argparse
from degnutil.input_output import log, open_
from degnutil import argument_parsing as argue

def get_parser():
    parser = argparse.ArgumentParser(description="Show or fix row lengths, which is the number of cells in a row.")
    argue.inoutfile(parser)
    argue.delimiter_tab(parser)
    parser.add_argument("-F", "--fix", action="store_true", help="Correct row lengths, i.e. add empty cells to match the first line or remove cells. Default is printing length of each row.")
    parser.add_argument("-r", "--right", action="store_true", help="Right align, i.e. add empty cells at start of lines. Default is appending to the end of lines.")
    return parser

def parse_args(args):
    return argue.parse_inoutfile(argue.parse_delimiter(args))

def main(args):
    args = parse_args(args)

    with open_(args.infile) as infile, open_(args.outfile, 'w') as outfile:

        if not args.fix:
            for line in infile:
                outfile.write(str(len(line.split(args.delimiter)))+'\n')
            return

        # logging of lines and cells modified
        n_removed, n_cells_removed, n_empty_cells_removed, n_added, n_cells_added = 0, 0, 0, 0, 0
        first_line = next(infile)
        outfile.write(first_line)
        length = len(first_line.strip('\n').split(args.delimiter))
        log.info(f"length of first line = {length}")
        for line in infile:
            l = len(line.split(args.delimiter))
            if l == length: outfile.write(line)
            elif l < length:
                # add empty cells
                n_added += 1
                add = args.delimiter * (length - l)
                n_cells_added += length - l
                if args.right: # right-align so modify start
                    outfile.write(add + line)
                else:
                    outfile.write(line.strip('\n') + add + '\n')

            elif l > length:
                # remove cells that might be empty or might not
                n_removed += 1
                rm = l - length
                n_cells_removed += rm
                line = line.strip('\n').split(args.delimiter)
                if args.right: # right-align so modify start
                    line, remove = line[rm:], line[:rm]
                else:
                    line, remove = line[:-rm], line[-rm:]
                n_empty_cells_removed += sum(c == "" for c in remove)
                outfile.write(args.delimiter.join(line) + '\n')

        if n_added > 0:
            log.info(f"Added {n_cells_added} empty cells to the {'start' if args.right else 'end'} of {n_added} lines.")
        if n_removed > 0:
            log.info(f"Removed {n_cells_removed} cells from the {'start' if args.right else 'end'} of {n_removed} lines.")
            if n_empty_cells_removed < n_cells_removed:
                log.warning(f"{n_cells_removed - n_empty_cells_removed} removed cells were NOT empty.")




if __name__ == '__main__':
    main(get_parser().parse_args())

