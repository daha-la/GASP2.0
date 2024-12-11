#!/usr/bin/env python3
# coding=utf-8
import sys, os, stat
import time
import re
import subprocess
import shutil
import logging
from degnutil import string_util as st


"""
This script is for IO related functions that are not specifically reading or writing files. 
It is not meant to be run on its own, but rather support other scripts.
"""

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger()

def process(command, timeout=None, max_tries=10):
    """
    Run a subprocess and wait for it to end.
    :param command: unix or python command to run
    :param timeout: possibility of setting a timeout to use before restarting process
    :param max_tries: number of times to restart the process before throwing an exception
    :return: (stdout, stderr) of process
    """
    for i_try in range(max_tries):
        time_begin = time.time()
        # run without interactive, so .bashrc defined things are not available
        proc = subprocess.Popen(["/bin/bash", "-c", command])
        # wait for subprocess to end
        try:
            results = proc.communicate(timeout=timeout)
            time_end = time.time()
            logging.debug("Process succeeded in {:.1f}".format(time_end - time_begin))
            return results
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.communicate()
            time_end = time.time()
            logging.warning("Subprocess attempt {:d} timed out after {:.1f}".format(i_try, time_end - time_begin))
    logging.error("Max number of tries ({:d}) for subprocess exceeded".format(max_tries))
    raise subprocess.TimeoutExpired(command, timeout)


def input_number(prompt):
    """
    Prompt for input that has to be a number
    :param prompt: 
    :return: 
    """
    while True:
        try: return st.parse_number(input(prompt))
        except (ValueError, TypeError): pass


def confirm(prompt):
    """
    Prompt for y/n input for confirmation.
    :param prompt: string message to use as prompt
    :return: True if y is pressed, False otherwise
    """
    from degnutil.get_key import getch
    
    while True:
        print(prompt, end='', flush=True)
        inp = getch().lower()
        if inp in ['y', 'n']:
            print(inp)
            return inp == 'y'
        print()


def print_overwrite(*line, sep=" "):
    """
    Print by overwriting the last print line. No newline is included here.
    :param line: string line to print
    :param sep: string separating each element in param line
    :return: 
    """
    sys.stdout.write("\r" + st.join(line, sep))
    sys.stdout.flush()


def get_terminal_width():
    """
    Get the number of characters that makes up the width of the terminal.
    Defaults to 80.
    :return: int terminal width
    """
    return shutil.get_terminal_size((80, 20))[0]


def get_terminal_height():
    """
    Get the number of characters that makes up the width of the terminal.
    Defaults to 80.
    :return: int terminal width
    """
    return shutil.get_terminal_size((80, 20))[1]


def get_memory_use():
    """
    Get memory use for this process.
    :return: memory use in GB.
    """
    import psutil
    memory_in_bytes = psutil.Process(os.getpid()).memory_info().rss
    return round(memory_in_bytes * 1e-9, 3)


def get_nproc(fallback=None):
    """
    Get number of processors available by checking for it in various ways. 
    :param fallback: if it cannot be detected fallback to this value
    :return: number of processors available
    """
    try: return int(os.environ['NPROC'])
    except KeyError:
        nproc = get_nproc_from_nodefile()
        if nproc: return nproc
        nproc = get_CPUs_allowed()
        if nproc: return nproc
        return fallback


def get_CPUs_allowed():
    """
    Try to retrieve the number of available CPU cores by reading a status file for the current process.
    If it fails to find this info, the value None is returned.
    :return: 
    """
    try:
        m = re.search(r'(?m)^Cpus_allowed:\s*(.*)$', open('/proc/self/status').read())
        if m:
            res = bin(int(m.group(1).replace(',', ''), 16)).count('1')
            if res > 0: return res
    except IOError: return None


def get_nproc_from_nodefile():
    """
    Try to get number of avaible processors from PBS_NODEFILE. 
    This variable will be unset if we are in the login node, so this will return None in that case.
    :return: 
    """
    bash_cmd = 'if [ ! -z $PBS_NODEFILE ]; then wc -l < $PBS_NODEFILE; fi'
    try: return int(os.popen(bash_cmd).read())
    except ValueError: return


def remove_stdout():
    """
    Redirect stdout to /dev/null
    :return: 
    """
    sys.stdout = open(os.devnull, 'w')


def save_stream(stream):
    """
    In cases where files are read multiples times you can run into issues when piping,
    since stdin is a stream that unlike normal files can only be read once,
    and is not read from the first line if closed and reopened but instead continues where it was left off.
    To manage this, this function can be used to check if infile name is stdin and in that case save it to /tmp memory
    and use that instead as the new infname.
    Use it when you have written a function that you would like to be able to handle stdin,
    but you are reading the infile multiple times.
    :return: string name of path, i.e. "/tmp/stream_<random number>"
    """
    import random
    
    memory_filename = "/tmp/stream_" + str(random.randint(0, 999999))
    with open(memory_filename, 'w') as memory_file:
        memory_file.write(stream.read())
    return memory_filename


def is_stdin(stream_or_path):
    """
    Return if stream_or_path is a stdin stream. 
    Works for sys.stdin and "/dev/stdin"
    :param stream_or_path: 
    :return: bool
    """
    return "stdin" in str(stream_or_path)


def save_if_stream(stream_or_path):
    if is_stdin(stream_or_path):
        return save_stream(stream_or_path)
    return stream_or_path


def is_reading_from_pipe():
    """
    Find out if we are piping into this script, i.e. should we be looking at stdin?
    This method is more robust that sys.stdin.isatty() and similar isatty calls, since they fail for e.g.
    cat <(myscript.py file) | ...
    A small consideration is FIFO is a named pipe, so it might mean that we would not detect an anonymous pipe.
    I don't think it is a problem on a modern system.
    :return: bool
    """
    return stat.S_ISFIFO(os.stat(0).st_mode)


def open_(filename, mode='r'):
    """
    Try/except version so if stdin or stdout is given, they will already be open and are simply returned.
    :param filename: 
    :param mode: 
    :return: 
    """
    try: return open(filename, mode)
    except TypeError: return filename


def script_path():
    return os.path.realpath(sys.modules['__main__'].__file__)

def script_dir():
    return os.path.dirname(script_path())


def dynamic_parser(pattern, description=None):
    """
    A dynamic parser that only loads method named "get_parser" from each submodule.
    :param pattern: relative path to files that should be added. Should have an asterisk to match multiple scripts. 
    Should probably be relative to package root (degnlib), haven't tested anything else.
    :param description: Text description for parser to print in help messages.
    :return: ArgumentParser
    """
    from degnutil.argument_parsing import subparser_group
    from argparse import ArgumentParser
    from pathlib import Path
    from glob import iglob
    
    assert '*' in pattern
    
    parser = ArgumentParser(description=description)
    subparsers = subparser_group(parser)

    package = script_dir()
    for module in iglob(package + "/" + pattern):
        module = str(Path(module).relative_to(package).with_suffix('')).replace('/', '.')
        module_parser = getattr(__import__(module, fromlist=["get_parser"]), "get_parser")
        # name is based on what is unique for each submodule so we trim away the common parts.
        name = module[pattern.find('*'):].replace('_', '-')
        subparsers.add_parser(name, parents=[module_parser()], add_help=False).set_defaults(function=module)

    return parser


def dynamic_main(parser):
    """
    A dynamic call to a main function.
    Using subparsers the parser can sometimes lack the function attribute in which case the help is printed. 
    :param parser: a parser with either a attribute function referring to a script with attribute main
    :return: None
    """
    parsed_args = parser.parse_args()
    if hasattr(parsed_args, "function"):
        getattr(__import__(parsed_args.function, fromlist=["main"]), "main")(parsed_args)
    else: parser.print_help()
