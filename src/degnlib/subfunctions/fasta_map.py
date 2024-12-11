#!/usr/bin/env python3
import argparse
from Bio import SeqIO
from Bio.File import as_handle
from degnutil import argument_parsing as argue, pandas_util as panda, bio_util as bio
from degnutil.input_output import log


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add a field to seq descriptions from a table that maps from a currently present field to the new field.")
    argue.inoutfile(parser)
    parser.add_argument("-t", "--table", required=True, help="")
    parser.add_argument("-1", "--old", default=0,
                        help="Name of old description field. Default is first word of sequence description and first column in table.")
    parser.add_argument("-2", "--new", default=1,
                        help="Name or column index for new description field. Default is second column in table. "
                             "If table has no header, the entire seq description is replaced.")
    argue.delimiter(parser)
    argue.fieldsep(parser)
    argue.kvsep(parser)
    argue.header(parser)
    parser.add_argument("-i", "--case-insensitive", action="store_true", 
                        help="Case insensitive comparison of the values between record descriptions and the values found in the table.")
    
    return parser
    

def parse_args(args):
    return argue.parse_inoutfile(args)


def main(args):
    args = parse_args(args)
    table = panda.read_pandas(args.table, sep=args.delimiter, header=args.header)
    if args.case_insensitive:
        old2new = dict(zip(panda.col(table, args.old).str.upper(), panda.col(table, args.new).str.upper()))
    else:
        old2new = dict(zip(panda.col(table, args.old), panda.col(table, args.new)))
    
    
    # get a string name for the new field
    try: new = table.columns[int(args.new)]
    except ValueError: new = args.new

    def _write_sequence(outfile, record, oldvalue):
        """
        
        :param outfile: 
        :param record: 
        :param oldvalue: 
        :return: whether the field was successfully added
        """
        if args.case_insensitive: oldvalue = oldvalue.upper()
        try: newvalue = old2new[oldvalue]
        except KeyError:
            bio.write_record(record, outfile)
            return False
        # if new is int it means the table did not have a header. In that case we fail and replace the description with the new name.
        try: record.description += "{}{:s}{}{}".format(args.fieldsep, new, args.kvsep, newvalue)
        except ValueError: record.description = newvalue
        bio.write_record(record, outfile)
        return True

    n_missed = 0

    with as_handle(args.outfile, 'w') as outfile:
        try: old = int(args.old)
        except ValueError:
            # str old index
            n_written = 0
            for i, record in enumerate(SeqIO.parse(args.infile, "fasta")):
                description = record.description.split(args.fieldsep)
                for field in description:
                    kv = field.split(args.kvsep)
                    if len(kv) == 2 and kv[0] == args.old:
                        n_missed += not _write_sequence(outfile, record, kv[1])
                        n_written += 1
                        break
            if n_written < i+1: # i+1 = number of records in infile
                log.error("Old field in descriptions were found for {} out of {} records.".format(n_written, i+1))
                
        else:
            # int old index
            for record in SeqIO.parse(args.infile, "fasta"):
                description = record.description.split(args.fieldsep)
                n_missed += not _write_sequence(outfile, record, description[old])
    
    if n_missed > 0:
        log.warning("{} records did not have a new field added.".format(n_missed))
           

if __name__ == '__main__':
    main(get_parser().parse_args())

