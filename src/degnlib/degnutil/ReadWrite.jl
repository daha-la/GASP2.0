#!/usr/bin/env julia
module ReadWrite

using DelimitedFiles

export loaddlm, loadmat, savedlm


function loaddlm(fname::String)
	ext = splitext(fname)[2]
	if ext in [".mat", ".txt", ".ssv"] readdlm(fname, ' ')
	elseif ext == ".csv" readdlm(fname, ',')
	elseif ext == ".tsv" readdlm(fname, '\t')
	else error("File format not recognized.") end
end
function loaddlm(fname::String, T::Type)
	ext = splitext(fname)[2]
	if ext in [".mat", ".txt", ".ssv"] readdlm(fname, ' ', T)
	elseif ext == ".csv" readdlm(fname, ',', T)
	elseif ext == ".tsv" readdlm(fname, '\t', T)
	else error("File format not recognized.") end
end
"Load dlm where we try to parse as int, and if that fails as float (which is does automatically)."
function loadmat(fname::String)
	try return loaddlm(fname, Int64)
	catch; return loaddlm(fname) end
end



function savedlm(fname::String, x::AbstractArray)
	ext = splitext(fname)[2]
	if ext in [".mat", ".txt", ".ssv"] writedlm(fname, x, ' ')
	elseif ext == ".csv" writedlm(fname, x, ',')
	elseif ext == ".tsv" writedlm(fname, x, '\t')
	else error("File format not recognized.") end
end
savedlm(o::Base.TTY, x::Matrix) = writedlm(o, x)


end;
