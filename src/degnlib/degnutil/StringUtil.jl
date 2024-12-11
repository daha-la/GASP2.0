#!/usr/bin/env julia


module StringUtil

export isalpha, parse_intfloat


isalpha(string::AbstractString) = all(isletter.(collect(string)))


"""
Parse to int if possible, otherwise float.
Can be broadcasted to parse each individual value in array.
"""
function parse_intfloat(str::AbstractString)
    out = tryparse(Int, str)
    if out === nothing parse(Float64, str)
    else out
    end
end
"""
Parse a matrix. 
So parse_int.(mat) will result in matrix that might have both ints and floats, 
while parse_int(mat) will give a matrix that is either all ints or all fallback to float. 
"""
function parse_intfloat(array::AbstractArray)
    out = tryparse.(Int, array)
    if any(out .=== nothing) parse.(Float64, array)
    else out
    end
end


end;

