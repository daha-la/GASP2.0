#!/usr/bin/env julia

module ArrayUtil

export cumrange, sumdrop, euclidean

"""
Get the ranges [1:vec[1], vec[1]+1:vec[2], ...]
Tested to be faster than iterating without cumsum.
"""
cumrange(vec) = (i-v+1:i for (v,i) in zip(vec,cumsum(vec)))

sumdrop(array; dims) = dropdims(sum(array; dims=dims); dims=dims)

"""
Euclidean distance between X and Y along the last dimension.
"""
euclidean(X::AbstractArray, Y::AbstractArray) = .√sumdrop((X .- Y).^2; dims=ndims(X))

"""
Kullback–Leibler divergence KLD(X || Y).
"""
relative_entropy(X::AbstractArray, Y::AbstractArray) = sumdrop(X .* log(X ./ Y); dims=ndims(X))


end;

