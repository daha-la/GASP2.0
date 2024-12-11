#!/usr/bin/env python3
import argparse
import logging

from degnutil import string_util as st
import pandas as pd
from xlsxwriter.table import Table
from xlsxwriter.worksheet import Worksheet
from warnings import warn
from pathlib import Path
import sys
import re

# modify Table to allow for filtering

def filter_column(self, criteria):
    tokens = self._extract_filter_tokens(criteria)
    if not (len(tokens) == 3 or len(tokens) == 7):
        warn("Incorrect number of tokens in criteria '%s'" % criteria)
    return self._parse_filter_expression(criteria, tokens)

def _write_auto_filter(self):
    """Extend Table to show active filters by copying code from Worksheet."""
    autofilter = self.properties.get('autofilter', 0)
    col_id = self.properties.get('col_id', 0)
    criteria = self.properties.get('criteria', 0)
    if not autofilter: return

    if not criteria:
        self._xml_empty_tag('autoFilter', [('ref', autofilter)])
    else:
        tokens = self.filter_column(criteria)
        self._xml_start_tag('autoFilter', [('ref', autofilter)])
        self._xml_start_tag('filterColumn', [('colId', col_id)])
        self._write_custom_filters(tokens)
        self._xml_end_tag('filterColumn')
        self._xml_end_tag('autoFilter')

# add filtering to the table auto filter writer
Table._extract_filter_tokens = Worksheet._extract_filter_tokens
Table._parse_filter_expression = Worksheet._parse_filter_expression
Table._parse_filter_tokens = Worksheet._parse_filter_tokens
Table._write_custom_filter = Worksheet._write_custom_filter
Table._write_custom_filters = Worksheet._write_custom_filters
Table.filter_column = filter_column
Table._write_auto_filter = _write_auto_filter


def _write_url(self, row, col, url, cell_format=None, string=None, tip=None):
    # Check that row and col are valid and store max and min values
    if self._check_dimensions(row, col):
        return -1

    # Set the displayed string to the URL unless defined by the user.
    if string is None:
        string = url

    # Default to external link type such as 'http://' or 'external:'.
    link_type = 1

    # Remove the URI scheme from internal links.
    if url.startswith('internal:'):
        url = url.replace('internal:', '')
        string = string.replace('internal:', '')
        link_type = 2

    # Remove the URI scheme from external links and change the directory
    # separator from Unix to Dos.
    external = False
    if url.startswith('external:'):
        url = url.replace('external:', '')
        url = url.replace('/', '\\')
        string = string.replace('external:', '')
        string = string.replace('/', '\\')
        external = True

    # Strip the mailto header.
    string = string.replace('mailto:', '')

    # Check that the string is < 32767 chars
    str_error = 0
    if len(string) > self.xls_strmax:
        warn("Ignoring URL since it exceeds Excel's string limit of "
             "32767 characters")
        return -2

    # Copy string for use in hyperlink elements.
    url_str = string

    # External links to URLs and to other Excel workbooks have slightly
    # different characteristics that we have to account for.
    if link_type == 1:

        # Split url into the link and optional anchor/location.
        if '#' in url:
            url, url_str = url.split('#', 1)
        else:
            url_str = None

        url = self._escape_url(url)

        if url_str is not None and not external:
            url_str = self._escape_url(url_str)

        # Add the file:/// URI to the url for Windows style "C:/" link and
        # Network shares.
        if re.match(r'\w:', url) or re.match(r'\\', url):
            url = 'file:///' + url

        # Convert a .\dir\file.xlsx link to dir\file.xlsx.
        url = re.sub(r'^\.\\', '', url)

    # Excel limits the escaped URL and location/anchor to 255 characters.
    tmp_url_str = url_str or ''
    max_url = self.max_url_length
    if len(url) > max_url or len(tmp_url_str) > max_url:
        warn("Ignoring URL '%s' with link or location/anchor > %d "
             "characters since it exceeds Excel's limit for URLS" %
             (force_unicode(url), max_url))
        return -3

    # Check the limit of URLS per worksheet.
    self.hlink_count += 1

    # Write previous row if in in-line string constant_memory mode.
    if self.constant_memory and row > self.previous_row:
        self._write_single_row(row)

    # Add the default URL format.
    if cell_format is None:
        cell_format = self.default_url_format

    # Write the hyperlink string.
    self._write_string(row, col, string, cell_format)

    if self.hlink_count > 65530:
        # warn("Ignoring URL '%s' since it exceeds Excel's limit of "
        #      "65,530 URLS per worksheet." % force_unicode(url))
        return -4

    # Store the hyperlink data in a separate structure.
    self.hyperlinks[row][col] = {
        'link_type': link_type,
        'url': url,
        'str': url_str,
        'tip': tip}

    return str_error
# remove 65k limit warning that spams output and allow for urls, even though the ones after the 65k limit will not be clickable
Worksheet._write_url = _write_url



def get_parser():
    parser = argparse.ArgumentParser(description="Convert table file(s) into an excel file.")
    parser.add_argument('infile', nargs="*", default=[], help="Table(s) or text file(s). .tsv, .csv, .txt, .tsv.gz or .csv.gz.")
    parser.add_argument('-o', '--out', dest='outfile', required=True, help="Specify outfile. Using stdout fails.")
    parser.add_argument("-s", "--sheet", default=["Sheet1"], nargs="+",
                        help="Sheet name(s) for each infile. If sheet names are repeated then tables are concatenated in the same sheet.")
    parser.add_argument("-f", "--format", action="store_true", help="Format as table.")
    parser.add_argument("-c", "--criteria", nargs="+", help="Filtering criteria for each table, e.g. \"row <= 10\".")
    parser.add_argument("-p", "--prefix", nargs="+", help="File(s) or string(s) to write just before infile(s) without gap.")
    parser.add_argument("--gap", type=int, default=1, help="How much gap to insert between concatenated tables. Default=1.")
    parser.add_argument("-a", "--args", help="Provide args in a table file where header is the long name of the arg.")
    parser.add_argument("-w", "--autofit", action="store_true", help="Try to adjust widths of columns to the table contents.")
    parser.add_argument("--cond-cell", nargs="+", help="Highlight cells with conditional that satisfy a criteria, e.g. '== 42'.")
    parser.add_argument("--cond-row", nargs="+", help="Highlight rows with conditional where a column satisfy a criteria, e.g. 'col == 42'. TODO.")
    parser.add_argument("--format-cell", nargs="+", help="Highlight cell by changing its format where it satisfies a criteria, e.g. '== 42'. TODO.")
    parser.add_argument("--format-row", nargs="+", help="Highlight rows by changing their format where a column satisfy a criteria, e.g. 'col == 42'. TODO.")
    return parser


def parse_args(args):
    if args.args is not None:
        table = pd.read_table(args.args, sep="\t")
        for col in table:
            setattr(args, col.lstrip('-'), table[col])

    # we don't want to worry about accidentally forgetting a space at either end
    args.sheet = [s.strip() for s in args.sheet]
    if len(args.sheet) == 1:
        args.sheet = [args.sheet[0] for _ in args.infile]
    else:
        assert len(args.sheet) == len(args.infile), f"{len(args.sheet)} != {len(args.infile)}"

    if args.criteria is not None and not args.format:
        raise NotImplementedError("Use -f/--format when filtering.")

    if args.criteria is None:
        args.criteria = [None for _ in args.infile]
    elif len(args.criteria) == 1:
        args.criteria = [args.criteria[0] for _ in args.infile]
    else:
        assert len(args.criteria) == len(args.infile)

    if args.prefix is None:
        args.prefix = [None for _ in args.infile]
    elif len(args.prefix) == 1:
        args.prefix = [args.prefix[0] for _ in args.infile]
    else:
        assert len(args.prefix) == len(args.infile)


def read_file(fname):
    """
    Read table or text file based on file ending.
    :param fname:
    :return: pandas or list of strings.
    """
    if fname is None or fname == '' or pd.isna(fname): return None
    try:
        if fname.endswith('.tsv') or fname.endswith('.tsv.gz'): return pd.read_table(fname)
        if fname.endswith('.csv') or fname.endswith('.csv.gz'): return pd.read_csv(fname)
        if fname.endswith('.txt'): return Path(fname).read_text().splitlines()
        if fname == '-': return pd.read_table(sys.stdin, sep='\t')
    except:
        warn(f"Problem reading {fname}")
        raise
    logging.warning(f"Unknown file format of {fname}, assuming text.")
    return Path(fname).read_text().splitlines()


def read(string):
    """
    Flexible reader that allows for string being either a filename or just the string itself that is to be read.
    :param string:
    :return: pandas or list of strings
    """
    if pd.isna(string): return []
    if Path(string).exists(): return read_file(string)
    else: return [string]


def pd_parse_values(table):
    # in case of columns with both strings and numbers, the whole column will by default be string.
    # Let's instead have it contain numbers if the value can be parsed as one.
    if type(table) != pd.DataFrame: return table
    return table.applymap(st.parse_value)


def add_table(table, sheet_name, writer, startrow=0, startcol=0, format=False, criteria=None):
    table.to_excel(writer, sheet_name=sheet_name, index=False, startrow=startrow, startcol=startcol)
    sheet = writer.sheets[sheet_name]
    if format:
        t = format_as_table(table, sheet, startrow, startcol)
        if criteria:
            filter_table(table, t, sheet, criteria, startrow)


def add_text(lines, sheet_name, writer, startrow=0, startcol=0):
    """
    Add lines of text to a sheet. The text in each line is split into cells based on tab characters.
    :param lines: list of strings, potentially with tabs in them
    :param sheet_name:
    :param writer:
    :param startrow:
    :param startcol:
    :return:
    """
    if sheet_name in writer.sheets:
        sheet = writer.sheets[sheet_name]
    else:
        sheet = writer.book.add_worksheet(sheet_name)
        writer.sheets[sheet_name] = sheet

    for i, line in enumerate(lines):
        for j, cell in enumerate(line.split('\t')):
            sheet.write(startrow+i, startcol+j, cell)


def add(toadd, sheet_name, writer, startrow=0, startcol=0, format=False, criteria=None):
    if toadd is None: return 0
    if type(toadd) == pd.DataFrame:
        add_table(toadd, sheet_name, writer, startrow, startcol, format=format, criteria=criteria)
        return len(toadd) + 1  # +1 for header
    else:
        add_text(toadd, sheet_name, writer, startrow, startcol)
        return len(toadd)


def format_as_table(table, sheet, startrow=0, startcol=0, style='Table Style Light 8'):
    options = {'columns': [{'header': col} for col in table.columns], 'style': style}
    return sheet.add_table(startrow, startcol, startrow + table.shape[0], startcol + table.shape[1] - 1, options)


def filter_table(table, table_dict, sheet, criteria, startrow):
    table_dict['criteria'] = criteria
    criteria = criteria.split()
    col = criteria[0]
    try: table_dict['col_id'] = list(table).index(col)
    except ValueError:
        warn("{} not found among {}".format(col, list(table)))
        return
    # rename col to x in criteria
    criteria[0] = "x"
    criteria = " ".join(criteria)
    for i, x in enumerate(table[col]):
        if not eval(criteria):
            sheet.set_row(startrow + 1 + i, options={'hidden': True})


def autofit(sheet, tables, criteria=None, maximum=25, precision=3, extra=3):
    # filter
    if criteria is not None:
        for i, t in enumerate(tables):
            try:
                tables[i] = t.loc[t.eval(criteria), :]
            except pd.core.computation.ops.UndefinedVariableError:
                warn("{} not evaluated on {}".format(criteria, list(t)))

    floatfmt = "{:." + str(precision) + "f}"

    for i in range(max(t.shape[1] for t in tables)):
        width = 0
        for table in tables:
            if i < table.shape[1]:
                column = table.iloc[:, i]
                if column.dtype == float:
                    width = max(width, len(column.name), max(map(len, [floatfmt.format(v) for v in column])))
                else:
                    width = max(width, len(column.name), max(map(len, map(str, column))))

        sheet.set_column(i, i, min(width + extra, maximum))


def get_cell_highlight_conditionals(expressions, workbook):
    expressions = [hc.split(maxsplit=1) for hc in expressions]
    expressions = [[c, st.parse(v)] for c, v in expressions]
    # make sure string is wrapped in quotes
    for i, (c, v) in enumerate(expressions):
        if type(v) == str: expressions[i][1] = '"{}"'.format(v.strip("'\""))
    return [{'type': 'cell', 'criteria': c, 'value': v, 'format': workbook.add_format({'bg_color': 'yellow'})}
            for c, v in expressions]


def main(args):
    parse_args(args)

    infiles = [pd_parse_values(read_file(fname)) for fname in args.infile]
    prefixes = [pd_parse_values(read(fname)) for fname in args.prefix]

    sheets = {s: [] for s in args.sheet}
    for s, t, p, c in zip(args.sheet, infiles, prefixes, args.criteria):
        sheets[s].append([t, p, c])

    with pd.ExcelWriter(args.outfile, engine='xlsxwriter') as writer:
        if args.cond_cell:
            cell_conditionals = get_cell_highlight_conditionals(args.cond_cell, writer.book)

        for sheet_name, tpcs in sheets.items():
            startrow = 0
            for infile, prefix, criteria in tpcs:
                startrow += add(prefix, sheet_name, writer, startrow, format=args.format)
                startrow += add(infile, sheet_name, writer, startrow, format=args.format, criteria=criteria) + args.gap

                if type(infile) == pd.DataFrame:
                    if args.cond_cell:
                        # idx = eval("infile " + args.cond_cell)
                        for conditional in cell_conditionals:
                            writer.sheets[sheet_name].conditional_format(startrow - args.gap - len(infile), 0, startrow - args.gap, infile.shape[1]-1, conditional)

                    if args.cond_row:
                        idx = infile.eval(args.cond_row)
                        # TODO

            if args.autofit:
                tables = [tpc[0] for tpc in tpcs if type(tpc[0]) == pd.DataFrame]
                if len(tables) > 0:
                    autofit(writer.sheets[sheet_name], tables, criteria=criteria)


if __name__ == '__main__':
    main(get_parser().parse_args())


