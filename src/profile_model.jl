#######################################################################################
################################### ProfileModel{q} ###################################
#######################################################################################

"""
    ProfileModel{q} <: EvolutionModel{q}

```
P :: SVector{q, Float64}
μ :: Float64
alphabet :: Alphabet
```
An independent model without using the genetic code.
Ordering is irrelevant in this case, defaults to `1:L`.
"""
@kwdef mutable struct ProfileModel{q} <: EvolutionModel{q}
    P :: Vector{Vector{Float64}}
    μ :: Float64 = 1.
    with_code :: Bool = false
    genetic_code :: Matrix{Float64} = zeros(Float64, q, q)
    alphabet :: Alphabet = default_alphabet(q)
    function ProfileModel{q}(P, μ, with_code, genetic_code, alphabet) where q
        for p in P
            @assert isapprox(sum(p), 1) "Probabilities must sum to one - got $(sum(p))"
        end
        @assert all(p -> length(p) == q, P) "Expected probability vectors of length $q"
        @assert μ>0 "Mutation rate should be strictly positive"
        @assert !with_code || q == length(_AA_ALPHABET) "Can only use genetic_code for amino-acids (got q=$q)"
        @assert length(alphabet) == q "Alphabet $alphabet and model (q=$q) must have consistent sizes"
        return new{q}(P, μ, with_code, genetic_code, alphabet)
    end
end
"""
    ProfileModel(P; kwargs...)

Return an `ProfileModel` object using probability `P`. `P` can be
- a vector of probability vectors, *e.g.* `[[1/2, 1/2], [1/3, 2/3], ...]` for `q=2`
- a matrix with each columns being the probability vector for a position, *e.g.* in the same case
```
[
    1/2 1/3 ...
    1/2 2/3 ...
]
```
"""
function ProfileModel(
    P::AbstractVector{<:AbstractVector};
    alphabet=first(P) |> length |> default_alphabet, kwargs...
)
    if !allequal(Iterators.map(length, P))
        error("Incorrect dimensions for probability $P")
    end

    q = length(first(P))
    return ProfileModel{q}(; P, alphabet=Alphabet(alphabet), kwargs...)
end
ProfileModel(P::AbstractMatrix; kwargs...) = ProfileModel(eachcol(P); kwargs...)

Base.length(model::ProfileModel) = length(model.P)

"""
    JukesCantor(L::Int)

The Jukes-Cantor model for sequences of length `L`.
Equivalent to `ProfileModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)`.
"""
JukesCantor(L::Int) = ProfileModel(map(_ -> [1/4, 1/4, 1/4, 1/4], 1:L); μ = 4/3)

"""
    ProfileModel(arnet::ArDCA.ArNet; M=1000)

Convenience. Sample `arnet` to compute single site frequencies, and return corresponding
profile model.
Use a sample of `M` sequences.
"""
function ProfileModel(arnet::ArDCA.ArNet; M = 1000, pc=true, kwargs...)
    q = length(arnet.p0)
    S = ArDCA.sample(arnet, M)'
    P = map(eachcol(S)) do X
        f = countmap(X)
        Z = sum(values(f)) + (pc ? q : 0)
        [(get(f, a, 0) + (pc ? 1 : 0))/Z for a in 1:q]
    end
    return ProfileModel(P; kwargs...)
end
ProfileModel(model::AutoRegressiveModel; kwargs...) = ProfileModel(model.arnet; kwargs...)

"""
    ProfileModel(fastafile::AbstractString; pc=1e-2, alphabet=:aa)
"""
function ProfileModel(
    fastafile::AbstractString;
    pc=1e-2, alphabet=:aa, reweighting=false, θ=0.2,
)
    sequences = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> sequence_to_intvec(sequence(rec); alphabet), reader)
    end
    L, M, q = length(first(sequences)), length(sequences), length(Alphabet(alphabet))
    weights = if reweighting
        W = zeros(Float64, length(sequences))
        for (m, seq) in enumerate(sequences), seq2 in sequences
            if _hamming(seq, seq2)/L <= θ
                W[m] += 1.
            end
        end
        1 ./ W
    else
        ones(Float64, length(sequences))
    end
    Meff = sum(weights)
    @info Meff
    P = map(1:L) do i
        p = zeros(Float64, q)
        for (m, seq) in enumerate(sequences)
            a = seq[i]
            p[a] += weights[m]
        end
        p /= Meff
        p = (1-pc)*p .+ pc/q
    end
    return ProfileModel(P; alphabet)
end
_hamming(X, Y) = sum(z -> z[1] == z[2] ? 0 : 1, zip(X, Y))

function change_alphabet(model::ProfileModel, alphabet::Alphabet)
    Q = map(model.P) do p
        q = map(collect(alphabet.string)) do a
            j = get(model.alphabet.mapping, a, 0)
            j == 0 ? 0. : p[j]
        end
    end
    return ProfileModel(Q; alphabet)
end

#=
########## ordering ##########
=#

ordering(model::ProfileModel) = 1:length(model)

#=
########## set_π ##########
=#
function set_π!(pstate::PosState{q}, model::ProfileModel{q}) where q
    for (a, x) in enumerate(model.P[pstate.pos])
        pstate.weights.π[a] = x
    end
    return nothing
end
set_π!(astate::AState, model::ProfileModel, pos::Int) = set_π!(astate.pstates[pos], model)


#=
########## set_transition_matrix ##########
=#

"""
    set_transition_matrix!(T::Matrix, model::ProfileModel, t, pos)

Convenience for `set_transition_matrix!(T, model.μ*t, model[pos])`
"""
function set_transition_matrix!(T::Matrix, model::ProfileModel, t, pos)
    return set_transition_matrix!(T, model.μ*t, model.P[pos])
end

#=
########## set_transition_rate_matrix ##########
=#

function set_transition_rate_matrix!(
    Q::Matrix{Float64},
    model::ProfileModel{q},
    pos::Int
) where q
    return set_transition_rate_matrix!(Q, model.P[pos])
end
# π useless for this model
function set_transition_rate_matrix!(Q::Matrix, model::ProfileModel, pos::Int, π)
    set_transition_rate_matrix!(Q, model, pos)
end

#=
########## Other ##########
=#

function log_transition_probability(
    old::AbstractVector,
    new::AbstractVector,
    t::Number,
    model::ProfileModel,
)
    model.with_code && error("For now this function cannot use genetic code")
    ν = exp(-model.μ * t)
    return sum(enumerate(zip(old, new))) do (i, (a,b))
        if isnothing(a) || isnothing(b)
            error(
                "Cannot compute transition probability for uninitialized sequence -
                Got `nothing` at position $i"
            )
        end
        log((1-ν)*model.P[i][b] + (a == b ? ν : 0.))
    end
end

function log_probability(seq::AbstractVector{<:Integer}, model::ProfileModel)
    return sum(enumerate(seq)) do (i, s)
        log(model.P[i][s])
    end
end
function log_probability(seq::AbstractString, model::ProfileModel)
    return log_probability(sequence_to_intvec(seq, model.alphabet), model)
end


sample(model::ProfileModel{q}) where q = map(P -> wsample(1:q, P), model.P)
function sample(model::ProfileModel, M::Int)
    X = zeros(Int64, length(model), M)
    for m in 1:M
        X[:, m] .= sample(model)
    end
    return X
end

ml_sequence(model::ProfileModel) = map(argmax, model.P)

entropy(model::ProfileModel, i::Int) = entropy(model.P[i])
function entropy(model::ProfileModel; ignore_columns = Int[])
    idx = filter(!in(ignore_columns), 1:length(model.P))
    sum(entropy, model.P[idx])
end

