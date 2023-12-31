#######################################################################################
################################## Position weights ###################################
#######################################################################################

mutable struct BranchWeights{q}
    π :: Vector{Float64} # eq. probabiltiy of each state
    u :: Vector{Float64} # up-likelihood: probability of data excluding the subtree
    v :: Vector{Float64} # down-likelihood: probability of data below the subtree
    Zu :: Array{Float64, 0} # log-normalization for u --> u*exp(Zu) are the actual likelihoods
    Zv :: Array{Float64, 0} # log-normalization for v // dim 0 array is a trick for mutability
    T :: Matrix{Float64} # propagator matrix to this state, given branch length to ancestor : T[a,b] = T[a-->b] = T[b|a,t]
    c :: Vector{Int} # character state
    function BranchWeights{q}(π, u, v, Zu, Zv, T, c) where q
        @assert isapprox(sum(π), 1) "Probabilities must sum to one - got $(sum(π))"
        @assert all(r -> sum(r)≈1, eachrow(T)) "Rows of transition matrix should sum to 1"
        @assert length(π) == q "Expected frequency vector of dimension $q, got $π"
        @assert length(u) == q "Expected weights vector of dimension $q, got $u"
        @assert length(v) == q "Expected weights vector of dimension $q, got $v"
        @assert length(c) == q "Expected character state vector of dimension $q, got $c"
        @assert size(T,1) == size(T,2) == q "Expected transition matrix of dimension $q, got $T"
        return new{q}(π, u, v, Zu, Zv, T, c)
    end
end
function BranchWeights{q}(π) where q
    return BranchWeights{q}(
        π,
        ones(Float64, q),
        ones(Float64, q),
        fill(0.),
        fill(0.),
        diagm(ones(Float64, q)),
        zeros(Int, q),
    )
end

BranchWeights{q}() where q = BranchWeights{q}(ones(Float64, q)/q)

function Base.copy(W::BranchWeights{q}) where q
    return BranchWeights{q}([copy(getproperty(W, f)) for f in propertynames(W)]...)
end

function reset_weights!(W::BranchWeights{q}) where q
    for a in 1:q
        W.u[a] = 1
        W.v[a] = 1
        W.c[a] = 0
        # foreach(b -> W.T[a,b] = 0, 1:q)
        # W.T[a,a] = 1
    end
    W.Zu[] = 0.
    W.Zv[] = 0.

    return nothing
end

function reset_up_likelihood!(W::BranchWeights{q}) where q
    foreach(a -> W.u[a] = 1, 1:q)
    W.Zu[] = 0.
    return nothing
end
reset_up_likelihood!(n::TreeNode, pos) = reset_up_likelihood!(n.data.pstates[pos].weights)

function reset_down_likelihood!(W::BranchWeights{q}) where q
    foreach(a -> W.v[a] = 1, 1:q)
    W.Zv[] = 0.
    return nothing
end
function reset_down_likelihood!(n::TreeNode, pos)
    reset_down_likelihood!(n.data.pstates[pos].weights)
end

function normalize!(W::BranchWeights)
    Zv = sum(W.v)
    Zu = sum(W.u)
    # check for issues
    if Zv == 0 || Zu == 0
        error("Found likelihood weight equal to 0 - model may not be able to accomodate for data")
    end

    for i in eachindex(W.v)
        W.u[i] = W.u[i]/Zu
        W.v[i] = W.v[i]/Zv
    end
    W.Zu[] += log(Zu)
    W.Zv[] += log(Zv)

    return nothing
end
normalize_weights!(n::TreeNode, pos::Int) = normalize!(n.data.pstates[pos].weights)

# sample(W::BranchWeights{q}) where q = StatsBase.sample(1:q, Weights((W.u' * W.T)' .* W.v))

#######################################################################################
#################################### Position state ###################################
#######################################################################################

@kwdef mutable struct PosState{q}
    pos::Int = 0
    c :: Union{Nothing, Int} = nothing # current state at this position
    lk::Float64 = 0.
    posterior :: Float64 = 0.
    weights::BranchWeights{q} = BranchWeights{q}() # weights for the alg.
end

function Base.copy(pstate::PosState{q}) where q
    return PosState{q}(;
        pos=pstate.pos,
        c = pstate.c,
        lk = pstate.lk,
        posterior = pstate.posterior,
        weights = copy(pstate.weights),
    )
end

function reset_state!(pstate::PosState)
    pstate.c = nothing
    pstate.lk = 0
    pstate.posterior = 0
    reset_weights!(pstate.weights)
    return nothing
end

#######################################################################################
################################### Ancestral state ###################################
#######################################################################################

@kwdef struct AState{q} <: TreeNodeData
    L::Int = 1

    # state of the node during the algorithm
    pstates :: Vector{PosState{q}} = [PosState{q}(; pos=i) for i in 1:L]

    # final result: whole sequence at this node
    sequence :: Vector{Union{Nothing, Int}} = Vector{Nothing}(undef, L) # length L

    function AState{q}(L, pstates, sequence) where q
        @assert length(pstates) == length(sequence) == L """
            Incorrect dimensions: expected sequence of length $L, got $sequence and $pstates
        """
        @assert all(x -> x[2].pos == x[1], enumerate(pstates))
        return new{q}(L, pstates, sequence)
    end
end

function Base.copy(state::AState{q}) where q
    return AState{q}(;
        L = state.L,
        pstates = map(copy, state.pstates),
        sequence = copy(state.sequence),
    )
end

reset_state!(state::AState, pos::Int) = reset_state!(state.pstates[pos])
reset_state!(tree::Tree, pos::Int) = foreach(n -> reset_state!(n.data, pos), nodes(tree))
function reset_state!(node::TreeNode{<:AState})
    !isleaf(node) && data(node).sequence .= nothing
    for pos in 1:data(node).L
        reset_state!(data(node), pos)
    end
    return nothing
end

reconstructed_positions(state::AState) = findall(!isnothing, state.sequence)
is_reconstructed(state::AState, pos::Int) = !isnothing(state.sequence[pos])
hassequence(state::AState{q}) where q = all(i -> is_reconstructed(state, i), 1:state.L)


function Base.show(io::IO, ::MIME"text/plain", state::AState{q}) where q
    if !get(io, :compact, false)
        println(io, "Ancestral state (L: $(state.L), q: $q)")
        println(io, "Sequence $(state.sequence)")
    end
    return nothing
end
function Base.show(io::IO, state::AState)
    print(io, "$(typeof(state)) - \
     $(length(reconstructed_positions(state))) reconstructed positions")
    return nothing
end

#######################################################################################
###################################### ASR Method #####################################
#######################################################################################

"""
    ASRMethod

- `joint::Bool`: joint or marginal inference. Default `false`.
- `ML::Bool`: maximum likelihood, or sampling. Default `false`.
- `alphabet :: Symbol`: alphabet used to map from integers to sequences. Default `:aa`.
- `verbosity :: Int`: verbosity level. Default 0.
- `optimize_branch_length`: Optimize the branch lengths of the tree according to the model.
  Default `true`.
"""
@kwdef mutable struct ASRMethod
    joint::Bool = true
    ML::Bool = false
    alphabet::Symbol = :aa
    verbosity::Int = 0
    optimize_branch_length = true
end
