#!/usr/bin/env julia
isdefined(Main, :MotifUtil) || include("../degnutil/motif_util.jl")

module MotifEcho
using ..MotifUtil


function main(infile, outfile;)
    memes = read_meme(infile)
    write_meme(outfile; memes...)
end

end;

