#!/usr/bin/env julia
isdefined(Main, :MotifUtil) || include("../degnutil/motif_util.jl")
isdefined(Main, :ArrayUtil) || include("../degnutil/array_util.jl")


module MotifAlign
using DataStructures, DataFrames
using Statistics
using Distributions: entropy
using Random: shuffle!
using LinearAlgebra
using Logging
using ..MotifUtil, ..ArrayUtil

"""
Euclidean distance score.
Reduce over the last axis alone,
which means if 1d you get a scalar score, if 2d you get a vector of scores and if 3d you get a matrix of scores.
- X: array of values, e.g. frequencies for each letter.
- Y: array of values, e.g. frequencies for each letter.
- return: -euclidean distance
"""
ED_score(X::AbstractArray, Y::AbstractArray) = - euclidean(X, Y)
"""
Scoring based on Kullback–Leibler divergence.
Average of KLD(X || Y) and KLD(Y || X).
"""
KLD_score(X::AbstractArray, Y::AbstractArray) = (relative_entropy(X, Y) .+ relative_entropy(Y, X)) ./ 2
"""
Another idea for a scoring as alternative to ED.
Rewards frequencies that are in agreement but is neutral for no agreement.
"""
prod_score(X::AbstractArray, Y::AbstractArray) = sumdrop(X .* Y; dims=ndims(X))
"""
Product of logo heights.
"""
logo_height_score(X::AbstractArray, Y::AbstractArray) = sumdrop(get_logo_height(X) .* get_logo_height(Y); dims=ndims(X))


"""
Get two 3D arrays with all unique pairs of comparison between each position in query and target.
- Q: dim 1 is position, dim 2 is letter
- T: dim 1 is position, dim 2 is letter
- return: X, Y
"""
function QT2XY(Q::Matrix, T::Matrix)
    # we cat along axis 3 since using axis 1 or 2 will not add a new dimension.
    # Then we quickly move that new axis to dim 2 and 1 so that position in X changes along dim 1, and in Y along dim 2,
    # while there are no changes observed along dim 2 and 1. Dim 3 will then be letter for both X and Y.
    X = permutedims(cat((Q for _ in 1:size(T,1))...; dims=3), [1,3,2])
    Y = permutedims(cat((T for _ in 1:size(Q,1))...; dims=3), [3,1,2])
    @assert size(X) == size(Y)
    X, Y
end

"""
Center medians in each row.
from shift_all_pairwise_scores in tomtom.c
"""
center_medians( Γ::Matrix) = Γ .-  median(Γ; dims=2)
center_medians!(Γ::Matrix) = Γ .-= median(Γ; dims=2)

"""
Reshape matrix Γ from Matrix to Matrix{Matrix}
with new indexing [query, target][query position, target position].
"""
function jagged_Γ(Γ::Matrix, ws::Vector)
    [Γ[is, js] for is in cumrange(ws), js in cumrange(ws)]
end

"""
Unjag matrix of matrices, where a matrix contains a matrix within each element.
Each matrix within are then hcatted side-by-side and vcatted above one another.
It is necessary that heights are the same for all array[i,:] matrices
and widths are the same for all array[:,j] matrices.
"""
unjag(array::Matrix) = hcat((vcat(array[:,j] ...) for j in 1:size(array,2))...)


"""
Get score for a specific pair each with a given offset.
- Γ_QT: score matrix with each position in a query along dim 1, and each position in a target along dim 2.
- offset_Q: How much query is shifted relative to the window.
- offset_T: How much target is shifted relative to the window.
"""
function get_pair_score(Γ_QT::Matrix, offset_Q::Int, offset_T::Int, window::Int)
    wQ, wT = size(Γ_QT)
    # indices within Q and T that fits inside the window
    Q_ind = (1:window) .- offset_Q
    T_ind = (1:window) .- offset_T
    # only use indices inside the query and target
    valid = (1 .<= Q_ind .<= wQ) .& (1 .<= T_ind .<= wT)
    Γ_QT[CartesianIndex.(Q_ind, T_ind)[valid]] |> sum
end
"""
Get a score for a specific pair each with a given offset. 
All positions in the window is scored, not just the positions where query and target overlaps.
- gap_scores_Q: same length as dim 1 in Γ_QT. Score between each position in query and a gap.
- gap_scores_T: same length as dim 2 in Γ_QT. Score between each position in target and a gap.
"""
function get_pair_score(Γ_QT::Matrix, offset_Q::Int, offset_T::Int, gap_scores_Q::Vector, gap_scores_T::Vector, window::Int)
    wQ, wT = size(Γ_QT)
    # indices within Q and T that fits inside the window
    Q_ind = (1:window) .- offset_Q
    T_ind = (1:window) .- offset_T
    # indices inside the query and target
    in_Q = 1 .<= Q_ind .<= wQ
    in_T = 1 .<= T_ind .<= wT
    overlap_score = Γ_QT[CartesianIndex.(Q_ind, T_ind)[in_Q .& in_T]] |> sum
    gap_Q_score   = sum(gap_scores_Q[Q_ind[in_Q .& .!in_T]])  # = 0 for logo_height_score
    gap_T_score   = sum(gap_scores_T[T_ind[in_T .& .!in_Q]])  # = 0 for logo_height_score
    gap_gap_score = -2 * sum(.!in_Q .& .!in_T)
    gap_score     = -2 * sum(.!in_Q .| .!in_T)
    # sum all scores
    overlap_score + gap_Q_score + gap_T_score + gap_gap_score
    # overlap_score + gap_score
    # overlap_score  # when using a positive scoring (prod or logo height) we don't need to punish gap since 0 will be a low score here.
end

"""
- offsets_Q: vector or range of ints
- offsets_T: vector or range of ints
"""
function get_pair_scores(Γ_QT::Matrix, offsets_Q, offsets_T, window::Int)
    [get_pair_score(Γ_QT, oq, ot, window) for oq in offsets_Q, ot in offsets_T]
end
function get_pair_scores(Γ_QT::Matrix, offsets_Q, offsets_T, gap_scores_Q, gap_scores_T, window::Int)
    [get_pair_score(Γ_QT, oq, ot, gap_scores_Q, gap_scores_T, window) for oq in offsets_Q, ot in offsets_T]
end

function get_pair_scores(Γ_jag::Matrix{Matrix{Float64}}, offsets::Vector, window::Int)
    [get_pair_scores(Γ_jag[q,t], offsets[q], offsets[t], window) for q in 1:size(Γ_jag,1), t in 1:size(Γ_jag,2)] |> unjag
end
function get_pair_scores(Γ_jag::Matrix{Matrix{Float64}}, offsets::Vector, gap_scores::Vector, window::Int)
    [get_pair_scores(Γ_jag[q,t], offsets[q], offsets[t], gap_scores[q], gap_scores[t], window) for q in 1:size(Γ_jag,1), t in 1:size(Γ_jag,2)] |> unjag
end

"""
- QT: matrix with dim #motifs × #letters
- bg_freqs: gap is treated as a position with background frequency of each letter.
"""
function get_gap_scores(QT::Matrix, offset_ranges, bg_freqs::Vector=ones(size(QT,2))/size(QT,2); score_func=ED)
    scores::Vector = score_func(QT, bg_freqs')
    [scores[r] for r in offset_ranges]
end

"""
Get overall score where each motif has a specific offset.
- Γ: matrix with indexing [query, target][query position, target position]
- offsets_Q: 2D column vector
- offsets_T: 2D row vector
- window: integer window length.
"""
function get_score(Γ_jag::Matrix{Matrix{Float64}}, offsets_Q::Matrix{Int}, offsets_T::Matrix{Int}, window::Int)
    sum(get_pair_score.(Γ_jag, offsets_Q, offsets_T, window))
end
"""
A hopefully more efficient method to get a global score.
"""
get_score(Π::Matrix, rowcols::Vector) = view(Π, rowcols, rowcols) |> sum


"""
Get range of potential offsets for a motif,
where it will overlap with the window by the at least the given minimum.
- w: width of motif.
- window: length of window.
- min_overlap: minimum overlap of motif and window.
"""
get_offsets(w::Int, window::Int; min_overlap::Int=1) = -w+min_overlap:window-min_overlap

"""
Get part of matrix (or dataframe) present within window.
- offset: how much the matrix is shifted relative to the window.
"""
get_matrix_in_window(matrix, offset::Int, window::Int) = matrix[max(0,-offset)+1 : min(end,window-offset), :]
"""
Get part of matrix (or dataframe) present within window where other parts of the window are zeros.
"""
function get_matrix_in_window_pad(matrix::Matrix, offset::Int, window::Int)
    ret = similar(matrix, window, size(matrix,2))
    ret .= 0
    ret[max(0,offset)+1 : min(end,size(matrix,1)+offset), :] = get_matrix_in_window(matrix, offset, window)
    ret
end
function get_matrix_in_window_pad(matrix::DataFrame, offset::Int, window::Int)
    ret = similar(matrix, window)
    ret .= 0
    ret[max(0,offset)+1 : min(end,size(matrix,1)+offset), :] = get_matrix_in_window(matrix, offset, window)
    ret
end

"""
Get total information content for each given offset for a matrix.
"""
get_ICs(matrix::Matrix, offsets, window::Int) = sum.(get_IC.(get_matrix_in_window.(Ref(matrix), offsets, window)))
get_ICs(matrix::Matrix, offsets, n_samples::Int, window::Int) = sum.(get_IC.(get_matrix_in_window.(Ref(matrix), offsets, window), n_samples))

"""
Filter potential offsets by information content.
- matrix: frequencies to calculate IC from.
- offsets: list or range of potential offsets for the matrix. 
- min_frac: an offset if IC > min_frac * max(ICs)
- window: length of window used for calculating IC.
"""
function IC_filter_offsets(matrix::Matrix, offsets, min_frac::Float64, window::Int)
    ICs = get_ICs(matrix, offsets, window)
    offsets[ICs .> min_frac * maximum(ICs)]
end

"""
Get worst possible score for an offset, where other offsets can assume any potential value.
- Π_col: column of Π, containing pairwise scores for a motif at a specific offset compared to all other motifs at all potential offsets.
- ranges: list of ranges containing each motif, in order to separate rows of the column into groups for each motif.
"""
worstcase(Π_col::AbstractVector, ranges) = sum(minimum(Π_col[r]) for r in ranges)
"Same as worstcase but max among scores instead of min."
bestcase( Π_col::AbstractVector, ranges) = sum(maximum(Π_col[r]) for r in ranges)

"""
Get logical indexing of the offsets that are in best case better than all worst case offsets for the particular motif.
"""
function better_than_worst(Π, ranges)
    worst = worstcase.(eachcol(Π), Ref(ranges))
    best  = bestcase.( eachcol(Π), Ref(ranges))

    vcat((best[r] .> maximum(worst[r]) for r in ranges)...)
end


"""
Find optimal offsets for all motifs with a simple method that iteratively finds the best solution.
This assumes the problem is convex and we don't end in a local maximum.
"""
solve_convex(Π::Matrix, ranges::Base.Generator) = solve_convex(Π, collect(ranges))
function solve_convex(Π::Matrix, ranges::Vector; max_iterations=100)
    scores = zeros(max_iterations)
    # first, find the best offset for each motif, measured on all potential offsets for the other motifs.
    solution, _ = _next_solution(Π, ranges)
    # 0-indexed ind of each motif. Converts solution ind within each range to index along a dim in Π matrix.
    sol2ind = [r[1]-1 for r in ranges]
    # iterate until convergence or max_iterations
    solution_last = similar(solution)
    indices = collect(1:length(ranges))
    for i in 1:max_iterations
        shuffle!(indices)
        solution[indices], scores[i] = _next_solution(Π, ranges[indices], solution + sol2ind)
        @info "iteration= $i\tscore= $(scores[i])"
        if all(solution .== solution_last) scores = scores[1:i]; break end
        solution_last .= solution
    end
    solution, scores
end

"""
Helper function for solve_convex. 
Finds the offset in each range that results in the largest score given all other offsets.
"""
_next_solution(Π::Matrix, ranges::Vector) = _next_solution(Π, ranges, :)
function _next_solution(Π::Matrix, ranges::Vector, rows)
    scores, solution = zip((findmax(dropdims(sum(Π[rows, r]; dims=1); dims=1)) for r in ranges)...)
    return collect(solution), sum(scores)  # collect, since it is an NTuple (unmutable, and has no similar method)
end



"""
Align all motifs to all others within a window of allowed positions.
"""
function align_motifs(motifs::OrderedDict, window::Int, min_overlap::Int, min_frac::Float64; score_func=logo_height_score)
    # assume all column names are the same
    matrices = Array.(values(motifs))
    ws = size.(matrices, 1)
    offsets = get_offsets.(ws, window; min_overlap=min_overlap)
    offsets = IC_filter_offsets.(matrices, offsets, min_frac, window)
    w_offsets = length.(offsets)
    @debug "Number of global scores to compare=" prod(BigInt.(w_offsets))
    
    QT = vcat(matrices...)
    Γ = score_func(QT2XY(QT, QT)...) #|> center_medians
    Π = get_pair_scores(jagged_Γ(Γ, ws), offsets, get_gap_scores(QT, cumrange(ws); score_func=score_func), window)
    @info "size(Π)=" size(Π)

    impossible = .!better_than_worst(Π, cumrange(w_offsets))
    any(impossible) && @info "There are some offsets that will never be the solution that can be removed."

    sol, sol_scores = solve_convex(Π, cumrange(w_offsets))
    sol_offsets = [o[s] for (o,s) in zip(offsets, sol)]
    @info sol_offsets

    for ((k, v), sol_offset) in zip(motifs, sol_offsets)
        motifs[k] = get_matrix_in_window_pad(v, sol_offset, window)
    end
    motifs
end

function main(infile, outfile, window::Int; min_overlap::Int=3, min_frac::Float64=.5)
    memes = read_meme(infile)
    motifs = align_motifs(memes.motifs, window, min_overlap, min_frac)
    # applying arg in expanding of memes is overridden by explicit call afterwards.
    write_meme(outfile; memes..., motifs=motifs)
end

end;
