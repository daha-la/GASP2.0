#!/usr/bin/env python3
# coding=utf-8
import sys
import argparse
from shlex import split as shell_split
import logging
import degnutil.input_output as io

"""
This script is for functions related to argument parsing, 
e.g. convenience functions for arguments often used.
"""


def verbose(parser):
    """
    Add a verbose setting to set the logging levels.
    :param parser: parser to modify
    :return: None
    """
    # print logging.info level
    parser.add_argument("-v", "--verbose", action=LoggingAction, const=logging.DEBUG, help="set flag to print verbosely.")


def quiet(parser):
    """
    Add a quiet setting.
    :param parser: parser to modify
    :return: None
    """
    # print logging.error level
    parser.add_argument("-q", "--quiet", action=LoggingAction, const=logging.ERROR, 
                        help="set flag to run quietly, i.e. reduce printing.")


def force(parser, help=None):
    """
    Add a force setting. 
    Used e.g. to overwrite or delete files.
    :param parser: argparse.ArgumentParser
    :param help: string help text, ignore for default text
    :return: 
    """
    if not help: help = "set flag to force actions."
    if not has_argument(parser, "force"):
        parser.add_argument("-f", "--force", action="store_true", help=help)


def yes(parser, help=None):
    """
    Add a yes setting. 
    Used e.g. to automatically answer yes.
    :param parser: argparse.ArgumentParser
    :param help: string help text, ignore for default text
    :return: 
    """
    if not help: help = "set flag to answer yes to question."
    if not has_argument(parser, "yes"):
        parser.add_argument("-y", "--yes", action="store_true", help=help)


class LoggingAction(argparse._StoreConstAction):
    def __init__(self, option_strings, dest, const, default=False, required=False, help=None):
        super().__init__(option_strings=option_strings, dest=dest, const=const, default=default, required=required, help=help)
        
    def __call__(self, parser, namespace, values, option_string=None):
        # use const as the logging level, default value is False like in store_true
        if self.const is not False:
            io.log.setLevel(self.const)
            # besides setting the logging level we also save a boolean variable like in store_true
            self.const = True
        super().__call__(parser, namespace, values, option_string)


def parse_args(parser, command):
    """
    Parse arguments given in a string using a given parser 
    where the arguments are split as they would have been in the unix shell.
    :param parser: argparse.ArgumentParser
    :param command: string
    :return: parsed args Namespace
    """
    return parser.parse_args(shell_split(command))


def subparser_group(parser):
    return parser.add_subparsers(title="function", description="The function to perform",
                                 help="Write a function followed by -h/--help to get help for it")


def add_subparsers(addto, parsers):
    """
    Add parsers to a subparser group or main parser using default subparser group.
    :param addto: subparser group or main parser to add the parsers to.
    :param parsers: dict of the parsers to add. 
    The keys are used as the identifier when starting a subcommand for the parser
    :return: the subparser group for chaining
    """
    if not hasattr(addto, "add_parser"):
        # main parser given
        return add_subparsers(subparser_group(addto), parsers)
    
    for name, parser in parsers.items():
        addto.add_parser(name, parents=[parser], description=parser.description)
    return addto


def has_argument(parser, dest):
    """
    Return whether a parser has an optional argument
    :param parser: argparse.ArgumentParser
    :param dest: string dest name of the argument
    :return: bool
    """
    # it seems it works for both _positionals and _optionals
    return dest in [arg.dest for arg in parser._positionals._actions]


def delimiter(parser, help="Delimiter character to separate values in file. Default=whitespace."):
    parser.add_argument("-d", "--sep", default=None, dest="delimiter", help=help)


def parse_delimiter(args):
    if args.delimiter == r'\t': args.delimiter = "\t"
    return args


def delimiter_tab(parser):
    parser.add_argument("-d", "--sep", default='\t', dest="delimiter",
                        help="Delimiter character to separate values in file. Default=tab.")

def fieldsep(parser):
    parser.add_argument("--fieldsep", dest="fieldsep", default=' ', help="Sequence id field separator")
    
def kvsep(parser):
    parser.add_argument("--kvsep", default='=', dest="kvsep", help="Sequence id key/value pair separator")


def header(parser):
    """
    Add header.
    Can be read with:
    table = read_pandas(args.infile, sep=args.delimiter, header=args.header)
    :param parser: 
    :return: 
    """
    parser.add_argument("-H", "--header", action="store_true",
                        help="Set flag to indicate the table has a header.")


def nproc(parser, default=None):
    """
    Add argument for number of processes.
    If default is not set, it is detected to be the available amount on current running process.
    :param parser: argument parser to add this argument term to.
    :param default: set this to e.g. 1 to not find default value from current running process.
    :return:
    """
    help = "Number of processors to use."
    if default:
        help += " Default set to {}".format(default)
    else:
        default = io.get_nproc(1)
        if default < 100:
            # we only use the current number of procs if it is not excessive
            help += " Default detected. Currently {}.".format(default)
        else:
            default = 1
            help += " Default=1."
        
    parser.add_argument("-np", "--nproc", type=int, default=default, help=help)


def infile(parser, help="Input file, default is stdin."):
    parser.add_argument('infile', nargs='?', default=sys.stdin, help=help)

def infiles(parser, help="Input file(s), default is stdin."):
    parser.add_argument('infiles', nargs="*", default=[sys.stdin], help=help)

def outfile(parser, help="Output file, default is stdout."):
    parser.add_argument('-o', '--out', dest='outfile', default=sys.stdout, help=help)

def inoutfile(parser, help="An input file and an output file."):
    parser.add_argument('inoutfile', nargs='*', default=[sys.stdin, sys.stdout], help=help)

def parse_inoutfile(args):
    if len(args.inoutfile) > 2:
        raise argparse.ArgumentTypeError("Expected 2 or less in and out files.")
    if len(args.inoutfile) == 1:
        # in earlier versions we check pipe to decide if arg is in or oufile,
        # however this fails in some edge cases, e.g. if this subfunction is 
        # not given a stdin pipe, but it is a command in a shell script that 
        # receives stdin pipe. There was apparently also an issue with symlink. 
        # The solution is to realize that we never really wanna write an 
        # outfile name as a singular arg since the whole point of being able to 
        # write outfile arg here is to do things like FILENAME{before,after} 
        # which can't be done with a single outfile arg.
        args.inoutfile = [args.inoutfile[0], sys.stdout]
        
    # replace "-"
    if args.inoutfile[0] == "-": args.inoutfile[0] = sys.stdin
    if args.inoutfile[1] == "-": args.inoutfile[1] = sys.stdout
    
    args.infile, args.outfile = args.inoutfile
    
    return args

