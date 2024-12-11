#!/usr/bin/env python3
import argparse
from pathlib import Path
from Bio.File import as_handle
from degnutil import argument_parsing as argue
from degnutil.string_util import has_match
from degnutil.input_output import log
from degnutil.path_util import isfile

def get_parser():
    parser = argparse.ArgumentParser(description="Advanced grep. Match from and to patterns to extract range(s) of lines, and patterns from a file.")
    argue.inoutfile(parser)
    parser.add_argument("-m", "--match", nargs="+", default=None, help="Pattern(s) to match. Provide filename to read args from each line in the file.")
    parser.add_argument("-f", "--from", dest="From", nargs="+", default=None, help="Start printing at line where pattern(s) match. Provide filename to read args from each line in the file.")
    parser.add_argument("-H", "--header", default=False, nargs="?", const=True,
                        help="Indicate a header is present. "
                             "A value can be provided if the header is not a single line. "
                             "It can be a character present as the first character in header lines (e.g. #). "
                             "It can also be a character first present at the end of the header lines, "
                             "e.g. '' to mean an empty line is present after the header lines. "
                             "Can also be an integer to indicate number of header lines.")
    parser.add_argument("-t", "--to", nargs="+", help="Printing until there is a match. Inclusive version of -s\--stop.")
    parser.add_argument("-s", "--stop", nargs="+", help="Stop printing BEFORE line that matches this. (e.g. '' for grepping until empty line).")
    return parser


def parse_args(args):
    args.match = parse_pattern(args.match)
    args.From = parse_pattern(args.From)
    args.to = parse_pattern(args.to)
    args.stop = parse_pattern(args.stop)

    try: args.header = int(args.header)
    except ValueError:
        if args.header in [r'\n', '']: args.header = "\n"

    return argue.parse_inoutfile(args)


def parse_pattern(patterns):
    if patterns is None: return None
    if len(patterns) == 1:
        if patterns[0] in [r'\n', '']: return "^$"
        if isfile(patterns[0]):
            patterns = Path(patterns[0]).read_text().split('\n')
    # make regex and filter empty match
    patterns = '|'.join([p for p in patterns if p != ''])
    return patterns.replace(r'\t', '\t')

def _has_match(pattern, line):
    if pattern is None: return False
    return has_match(pattern, line.strip())

def main(args):
    args = parse_args(args)
    
    printing = False
    
    with as_handle(args.infile) as infile, as_handle(args.outfile, 'w') as outfile:
        
        if args.header is not False:
            # reading header
            if args.header is True: outfile.write(next(infile))
            elif isinstance(args.header, int):
                for _ in range(args.header):
                    outfile.write(next(infile))
            else:  
                header_found = False
                i_header = 0  # cannot use enumerate here
                for line in infile:
                    if line.startswith(args.header):
                        header_found = True
                        outfile.write(line)
                        i_header += 1
                    elif not header_found:
                        outfile.write(line)
                        i_header += 1
                    else:
                        # we are no longer looking at a header line
                        if _has_match(args.From, line):
                            outfile.write(line)
                            printing = True
                log.info("Header read for {} lines.".format(i_header))

        # printing variable now controls if we are in a printing region.
        # printing lines between args.From and args.to/stop. If args.match is given we have to also filter for that.
        if args.From is None:
            if args.to is None and args.stop is None:
                for line in infile:
                    if _has_match(args.match, line):
                        outfile.write(line)
                return

            # getting here means args.to is given but not args.From, so the range of printing starts at once.
            printing = True

        for line in infile:
            if not printing: printing = _has_match(args.From, line)
            if printing:  # We are at or after a args.From match. cannot be replaced with 'else' statement.
                # look for args.to/stop. If it is at the same line as args.From then we keep printing.
                if _has_match(args.From, line): outfile.write(line)
                elif _has_match(args.stop, line): printing = False
                elif _has_match(args.to, line):
                    outfile.write(line)
                    printing = False
                else: outfile.write(line)

        

if __name__ == '__main__':
    main(get_parser().parse_args())

