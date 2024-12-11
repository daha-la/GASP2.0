#!/usr/bin/env julia
isdefined(Main, :CLI) || include("../degnutil/CLI.jl")
isdefined(Main, :InputOutput) || include("../degnutil/InputOutput.jl")
isdefined(Main, :FastaUtil) || include("../degnutil/FastaUtil.jl")
isdefined(Main, :MotifUtil) || include("../degnutil/MotifUtil.jl")

"""
Get information content for groups of sequences. Currently grouped by identical seq id.
Output format is two tab separated columns: seq id, then IC values for each position, delimited by spaces. 
- group: false to measure IC among all sequences. true to group by sequence ids. Provide filename with id groups on each line to measure IC for custom groups.
"""

using .CLI, .InputOutput, .FastaUtil, .MotifUtil
using DelimitedFiles

"""
- group: whether to measure IC grouped by seq ids. Default is measuring across all seqs. 
"""
function write_IC(writer, seqs::Vector, ids::Vector, alphabet::String, group::Bool)
    if group
        for id in unique(ids)
            ICs = get_IC(String.(seqs[ids .== id]), alphabet)
            write(writer, id * "\t" * join(ICs, " ") * "\n")
        end
    else
        ICs = get_IC(String.(seqs), alphabet)
        write(writer, join(ICs, "\n") * "\n")
    end
end

"""
- group: vector where each element is a group of seq ids.
"""
function write_IC(writer, seqs::Vector, ids::Vector, alphabet::String, groups::Vector)
    for group in groups
        g_seqs = vcat((seqs[ids .== id] for id in group)...)
        ICs = get_IC(String.(g_seqs), alphabet)
        write(writer, join(g, " ") * "\t" * join(ICs, " ") * "\n")
    end
end


function get_alphabet(seqs)
    letters = seqs |> prod |> unique
    join(letters[isletter.(letters)])
end


function main(io, o; alphabet::String, group, delimiter=isspace, header::Bool=false)
    i, o = inout(io, o)
    seqs, ids = read_fasta(i)
    alphabet != "" || (alphabet = get_alphabet(seqs))
    if !(group isa Bool)
        header || error("Not implemented.")
        group = readdlm(group, delimiter)
    elseif !group && !all(length.(seqs) .== length(seqs[1]))
        error("Sequences are not all the same length so a frequency matrix cannot be made. Perhaps you wanted to group by header name (-g/--group)?")
    end
    
    writer = InputOutput.open(o, "w")
    write_IC(writer, seqs, ids, alphabet, group)
    close(writer)
end

