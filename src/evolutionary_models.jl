"""
    abstract type EvolutionModel{q}

Describes the dynamics of sequences for a state space (alphabet) of size `q`.
Concrete types of `EvolutionModel` should implement `set_π!` and `set_transition_matrix!`.
"""
abstract type EvolutionModel{q} end

###################################################################################
#################################### NECESSARY ####################################
###################################################################################

"""
    set_π!(astate::AState, model::EvolutionModel, pos::Int)

Set equilibrium frequencies `astate.weights.π` at site `pos` using `model`.
"""
function set_π! end

"""
    ordering(::EvolutionModel)

Order in which sequence sites are processed.
"""
ordering(::EvolutionModel) = nothing


###################################################################################
###################################### USEFUL #####################################
###################################################################################



"""
    set_transition_rate_matrix!(Q::Matrix{Float64}, model::EvolutionModel, t, pos, π)

Set transition rate matrix in place, in matrix `Q`.
`π` is the expected equilibrium probability (needed only for AR model).
"""
function set_transition_rate_matrix! end

"""
    log_transition_probability(old::Array{Int}, new::Array{Int}, t, model::EvolutionaryModel)

Log-probability of transition from configurations `old` to `new` in time `t` for `model`.
"""
function log_transition_probability end

"""
    transition_matrix(model::EvolutionModel, t, pos[, π])

Return the transition matrix for `model` at position `pos` and branch length `t`.
"""
function transition_matrix(model::EvolutionModel{q}, t, pos, π=ones(Float64, q)/q) where q
    T = zeros(Float64, q, q)
    set_transition_matrix!(T, model, t, pos, π)
    return T
end
"""
    transition_rate_matrix(model::EvolutionModel, pos)

Return the transition rate matrix for `model` at position `pos`.
"""
function transition_rate_matrix(model::EvolutionModel{q}, pos, π=ones(Float64, q)/q) where q
    Q = zeros(Float64, q, q)
    set_transition_rate_matrix!(Q, model, pos, π)
    return Q
end


###################################################################################
################################# GENERAL FUNCTIONS ###############################
###################################################################################

"""
    set_π!(tree::Tree, model::EvolutionModel, pos::Int)

Set equilibrium frequencies at site `pos` for all nodes.
"""
function set_π!(tree::Tree, model::EvolutionModel, pos::Int)
    foreach(nodes(tree)) do n
        set_π!(n.data, model, pos)
    end
    return nothing
end

"""
    set_transition_rate_matrix!(Q, π; with_code=false)
"""
function set_transition_rate_matrix!(Q, π; with_code=false)
    return if with_code
        set_transition_rate_matrix_gencode!(Q, π)
    else
        set_transition_rate_matrix_simple!(Q, π)
    end
end
function set_transition_rate_matrix_simple!(Q, π)
    q = size(Q, 1)
    for a in 1:q
        Q[a, :] = π
        Q[a, a] -= 1
    end
    return Q
end
function set_transition_rate_matrix_gencode!(Q, π)
    q = 21
    @assert q == length(π)
    for a in 1:q, b in 1:q
        Q[a,b] = gencode_as_mat[a,b]*π[b]
    end
    for a in 1:q
        Q[a,a] -= sum(Q[a,:])
    end
    # normalizing rates
    # have to do something - but my simple matrix isn't normalized either
    return Q
end

"""
    set_transition_matrix!(T::Matrix, t, π; with_code, gen_code)

Set transition matrix with eq. frequencies π in place.
If provided, `gen_code` is a `Matrix`.
"""
function set_transition_matrix!(
    T::Matrix, t::Number, π::AbstractVector;
    with_code=false, gen_code = nothing,
)
    return if with_code
        # isnothing(gen_code) && error("Must provide genetic code! (`; gen_code = Matrix...)")
        set_transition_matrix_gencode!(T, t, π)
    else
        set_transition_matrix_simple!(T, t, π)
    end
end
function set_transition_matrix!(T::Matrix, t::Missing, π::AbstractVector; kwargs...)
    return set_transition_matrix!(T, Inf, π; kwargs...)
end

function set_transition_matrix_gencode!(T, t, π)
    set_transition_rate_matrix_gencode!(T, π)
    if t < Inf
        T .= exp(T*t)
    else
        q = size(T, 1)
        # 1 is gap, on its own for gen code
        T[1,:] .= 0.
        T[:, 1] .= 0.
        T[1,1] = 1.0

        for a in 2:q
            T[2:end, a] .= π[a]
        end
        # normalization not guaranteed with gaps...
        for a in 1:q
            T[a, :] ./= sum(T[a,:])
        end
    end
    return T
end

function set_transition_matrix_simple!(T, t, π)
    ν = exp(-t)
    q = length(π)
    for b in 1:q
        T[:,b] .= (1-ν) * π[b]
        T[b,b] += ν
    end
    return T
end

"""
    set_transition_matrix!(
        astate::AState, model::EvolutionModel, t, pos;
        set_equilibrium_frequencies=true
    )

Set transition matrix to the input ancestral state, using branch length `t`.
Store result in `astate.weights.T`.

## Note
- calls `set_π!(astate, model, pos)` if needed
- then uses `set_transition_matrix!(astate.T, model.μ*t, π; gen code from model)`
"""
function set_transition_matrix!(
    astate::AState, model::EvolutionModel, t, pos;
    set_equilibrium_frequencies=true
)
    set_equilibrium_frequencies && set_π!(astate, model, pos)
    return set_transition_matrix!(
        astate.weights.T, model.μ*t, astate.weights.π;
        with_code = model.with_code, gen_code = model.genetic_code,
    )
end

"""
    set_transition_matrix!(tree::Tree, model::EvolutionModel, pos::Int)

Set the transition matrix for all branches in `tree` at seqeunce position `pos`.
"""
function set_transition_matrix!(tree::Tree, model::EvolutionModel, pos::Int)
    foreach(nodes(tree)) do n
        set_transition_matrix!(n.data, model, branch_length(n), pos)
    end
    return nothing
end



