#!/usr/bin/env julia

import Base

"""
Input and output utilities.
"""
module InputOutput

export open

"""
Open a file or return unchanged if given an already opened stream.
"""
open(filename::AbstractString, args...) = Base.open(filename, args...)
open(io, args...) = io


end;

