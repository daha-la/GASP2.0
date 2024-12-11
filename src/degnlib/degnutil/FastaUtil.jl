#!/usr/bin/env julia
isdefined(Main, :InputOutput) || include("InputOutput.jl")

module FastaUtil
using FASTX
import ..InputOutput

export read_fasta

"""
Read sequences and ids from a fasta file.
"""
function read_fasta(handleish)
    reader = InputOutput.open(handleish, "r") |> FASTA.Reader
    records = collect(reader)
    close(reader)
    String.(sequence.(records)), identifier.(records)
end



end;


