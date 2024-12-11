#!/usr/bin/env julia
isdefined(Main, :InputOutput) || include("InputOutput.jl")
isdefined(Main, :StringUtil) || include("StringUtil.jl")

"""
This module is for motif related functions. 
It is not meant to be run on its own, but rather support other scripts.
"""
module MotifUtil
using DataStructures, DataFrames
using Distributions: entropy
import ..InputOutput
using ..StringUtil

export read_meme, write_meme
export get_IC, get_logo_height


"""
Read minimal MEME format.
http://meme-suite.org/doc/meme-format.html
"""
function read_meme(handleish)
    alphabet, strands, bg_freq, motifs, n_samples = nothing, nothing, nothing, OrderedDict(), OrderedDict()
    name = nothing

    io = InputOutput.open(handleish)
    if !startswith(readline(io), "MEME version") error("File not in MEME minimal format.") end

    for line in eachline(io)
        line = strip(line)
        if line == "" continue end

        if startswith(line, "ALPHABET=")
            alphabet = line[length("ALPHABET=")+1:end] |> lstrip
            # minimal meme format allows for multi-line alphabets. Let's not worry about that if we don't have to.
            if !isalpha(alphabet) error("Multi-line alphabet descriptions not supported yet.") end
            alphabet = collect(alphabet)
        
        elseif startswith(line, "strands:")
            strands = line[length("strands:")+1:end] |> lstrip

        elseif startswith(line, "Background letter frequencies")
            bg_freq = []
            for bg_freq_line in eachline(io)
                bg_freq_line = strip(bg_freq_line)
                if bg_freq_line == "" break end
                if startswith(bg_freq_line, "MOTIF")  # reading past bg freq
                    name = split(bg_freq_line)[2]  # ignore alternative names (format is MOTIF name alt_name)
                    break
                end

                bg_freq_line = split(bg_freq_line)
                # handles multi-line
                append!(bg_freq, Pair.(bg_freq_line[1:2:end], bg_freq_line[2:2:end]))
            end
            bg_freq = OrderedDict(bg_freq)
        
        elseif startswith(line, "MOTIF")
            name = split(line)[2]  # ignore alternative names (format is MOTIF name alt_name)

        elseif startswith(line, "letter-probability matrix:")
            mat_info = line[length("letter-probability matrix:")+1:end] |> lstrip |> split
            mat_info = Pair.(rstrip.(mat_info[1:2:end], '='), parse_intfloat.(mat_info[2:2:end])) |> Dict
            if "alength" in keys(mat_info) && alphabet !== nothing
                @assert mat_info["alength"] == length(alphabet)
            end
            if "nsites" in keys(mat_info)
                n_samples[name] = mat_info["nsites"]
            end
            # read the matrix numbers
            motifs[name] = []
            for mat_line in eachline(io)
                mat_line = strip(mat_line)
                if mat_line == "" break end
                if startswith(mat_line, "MOTIF")
                    name = split(mat_line)[2]  # ignore alternative names
                    break
                end
                push!(motifs[name], split(mat_line))
                if "alength" in keys(mat_info)
                    @assert length(split(mat_line)) == mat_info["alength"]
                end
            end
        end
    end
    close(io)

    # parse list of list of string into matrix of int or float.
    # position along axis=1, and letter along axis=2
    for (k, v) in motifs motifs[k] = parse_intfloat(hcat(v...))' end
    # assign alphabet names to columns
    if alphabet !== nothing
        for (k, v) in motifs motifs[k] = DataFrame(v, string.(alphabet)) end
    end

    (alphabet=alphabet, strands=strands, bg_freq=bg_freq, motifs=motifs, n_samples=n_samples)
end


"""
Write MEME minimal version 4 format.
"""
function write_meme(handleish; alphabet=nothing, strands=nothing, bg_freq=nothing, motifs, n_samples=nothing)
    # get alphabet
    alphabet !== nothing || 
    (alphabet = get_alphabet(bg_freq); alphabet !== nothing) || 
    (alphabet = get_alphabet(motifs))
    if alphabet !== nothing
        alphabet = join(alphabet)
        # motifs should have the same alphabet
        for matrix in values(motifs)
            mat_alphabet = get_alphabet(matrix)
            @assert mat_alphabet === nothing || join(mat_alphabet) == alphabet "Motifs have different alphabets."
        end
    end

    bg_freq = get_bg_freq(bg_freq, alphabet)
    n_samples = get_n_samples(n_samples, motifs)

    io = InputOutput.open(handleish, "w")
    write(io, "MEME version 4\n\n")
    alphabet === nothing || write(io, "ALPHABET= $alphabet\n\n")
    strands === nothing || write(io, "strands: $(join(strands, " "))\n\n")
    if bg_freq !== nothing
        write(io, "Background letter frequencies\n")
        write(io, join((join(kv, " ") for kv in bg_freq), " ") * "\n\n")
    end

    for (name, matrix) in motifs
        w, alength = size(matrix)
        nsites = get(n_samples, name, nothing)
        write(io, "MOTIF $name\n")
        write(io, "letter-probability matrix: alength= $alength w= $w")
        if nsites === nothing write(io, "\n")
        else write(io, " nsites= $nsites\n")
        end
        for row in eachrow(matrix) write(io, " " * join(row, " ") * "\n") end
        write(io, "\n")
    end
    close(io)
end

get_alphabet(bg_freqs::OrderedDict{Float64}) = keys(bg_freqs)
function get_alphabet(motifs::OrderedDict)
    for matrix in values(motifs)
        alphabet = get_alphabet(matrix)
        alphabet === nothing || return alphabet
    end
end
get_alphabet(matrix::DataFrame) = names(matrix)
get_alphabet(x) = nothing

get_bg_freq(bg_freq::OrderedDict, ::Any) = bg_freq
get_bg_freq(bg_freq::Vector, ::Nothing) = error("Alphabet should be provided in order to write the background frequencies.")
get_bg_freq(bg_freq::Vector, alphabet) = OrderedDict(zip(alphabet, bg_freq))
get_bg_freq(::Nothing, ::Any) = nothing

get_n_samples(n_samples::OrderedDict, ::Any) = n_samples
get_n_samples(n_samples::Vector, motifs::OrderedDict) = OrderedDict(zip(keys(motifs), n_samples))
get_n_samples(n_samples::Int, motifs::OrderedDict) = OrderedDict((k, n_samples) for k in keys(motifs))
get_n_samples(::Nothing, ::Any) = OrderedDict{String,Int}()




"""
Get the empirical position frequency matrix for sequences.
- seqs: list of strings. Aligned sequences.
- alphabet: e.g. "ACGU"
"""
get_PFM(seqs::Vector{String}, alphabet::String) = get_PFM(seqs, collect(alphabet))
function get_PFM(seqs::Vector{String}, alphabet::Vector{Char})
    # matrix for letters with seq position along dim=1, and each sequence found along dim=2. 
    letters = hcat(collect.(seqs)...)
    hcat((sum(letters .== l; dims=2) for l in alphabet)...)
end
"""
Get empirical position probability matrix for sequences.
"""
function get_PPM(seqs, alphabet)
    pfm = get_PFM(seqs, alphabet)
    pfm ./ sum(pfm; dims=2)
end


"""
Get information content, i.e. the total height of a bar in a logo plot.
- f: frequency of each letter, e.g. 4 floats for RNA.
"""
get_IC(f::AbstractVector{Float64}) = log2(length(f)) - entropy(f, 2)
"""
Get information content for multiple positions.
- f: position along dim 1, letter along dim 2.
"""
get_IC(f::AbstractMatrix{Float64}) = get_IC.(eachrow(f))
"""
Get the information content for a 3D array where the last dim is along alphabet.
"""
get_IC(f::AbstractArray{Float64,3}) = [get_IC(f[i, j, :]) for i in 1:size(f,1), j in 1:size(f,2)]

"""
- n: number of samples.
"""
get_IC(f::AbstractArray, n::Int) = get_IC(f) .- small_sample_correction(n, size(f)[end])



get_IC(seqs::Vector{String}, alphabet) = get_IC(get_PPM(seqs, alphabet))



"""
The approximation of the small-sample correction.
- n: sample size
- s: alphabet size
"""
small_sample_correction(n::Int, s::Int) = 1/log(2) * (s-1)/(2n)


get_logo_height(f) = get_IC(f) .* f
get_logo_height(f, n::Int) = get_IC(f, n) .* f







end;

