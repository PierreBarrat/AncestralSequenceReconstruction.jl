@kwdef mutable struct AutoRegressiveModel{q} <: EvolutionModel{q}
    arnet::ArDCA.ArNet
    μ :: Float64 = 1.
    with_code :: Bool = false
    genetic_code :: Matrix{Float64} = zeros(Float64, q, q)
    alphabet :: Alphabet = default_alphabet(q)
    function AutoRegressiveModel{q}(arnet, μ, with_code, genetic_code, alphabet) where q
        alphabet = Alphabet(alphabet) # in case a symbol or a string was passed
        @assert length(arnet.p0) == q "Expected arnet with $q states"
        @assert μ>0 "Mutation rate should be strictly positive"
        @assert !with_code || q == length(_AA_ALPHABET) "Can only use genetic_code for amino-acids (got q=$q)"
        @assert length(alphabet) == q "Alphabet $alphabet and model (q=$q) must have consistent sizes"
        arnet = deepcopy(arnet)
        regularize_p0!(arnet)
        return new{q}(arnet, μ, with_code, genetic_code, alphabet)
    end
end

function regularize_p0!(arnet::ArDCA.ArNet)
    q = round(Int, length(arnet.H) / length(arnet.idxperm))
    pc = 1e-6
    arnet.p0 .= (1-pc)*arnet.p0 .+ pc/q
    return arnet
end

function AutoRegressiveModel(
    arnet::ArDCA.ArNet;
    alphabet = default_alphabet(length(arnet.p0)), kwargs...
)
    q = length(arnet.p0)
    return AutoRegressiveModel{q}(; arnet, alphabet, kwargs...)
end

length(model::AutoRegressiveModel) = length(model.arnet.H) + 1

ordering(model::AutoRegressiveModel) = model.arnet.idxperm

#=
########## set_π ##########
=#
function set_π!(astate::AState{q}, model::AutoRegressiveModel{q}, pos::Int) where q
    idxperm = model.arnet.idxperm
    ar_pos = findfirst(==(pos), idxperm)
    # special case of first ordering position
    if ar_pos == 1
        for (a, x) in enumerate(model.arnet.p0)
            astate.weights.π[a] = x
        end
        return nothing
    end

    J = model.arnet.J[ar_pos-1]
    H = model.arnet.H[ar_pos-1]

    local_field = copy(H)
    for ar_i in 1:(ar_pos-1)
        b = astate.sequence[idxperm[ar_i]]
        isnothing(b) && error("Sequence not yet reconstructed at site $(idxperm[ar_i])")
        for a in 1:q
            local_field[a] += J[a, b, ar_i]
        end
    end
    ArDCA.softmax!(local_field)

    for (a, x) in enumerate(local_field)
        astate.weights.π[a] = x
    end
end


#=

=#

function get_local_field(sequence::AbstractVector, model::AutoRegressiveModel{q}) where q
    # pos is in the reference of arnet
    pos = length(sequence) + 1
    if pos == 1
        return model.arnet.p0
    end

    J = model.arnet.J[pos-1]
    H = model.arnet.H[pos-1]
    local_field = copy(H)
    for (i, b) in enumerate(sequence), a in 1:q
        local_field[a] += J[a, b, i]
    end
    ArDCA.softmax!(local_field)

    return local_field
end

function log_transition_probability(
    old::AbstractArray,
    new::AbstractArray,
    t::Number,
    model::AutoRegressiveModel,
)
    model.with_code && error("For now this function cannot use genetic code")
    ν = exp(-model.μ * t)
    return sum(enumerate(zip(old, new))) do (i,(a,b))
        local_field = get_local_field(new[1:(i-1)], model)
        log((1-ν)*local_field[b] + (a == b ? ν : 0.))
    end
end
function log_probability(sequence::AbstractVector, model::AutoRegressiveModel)
    return ArDCA.loglikelihood(convert(Vector{Int}, sequence), model.arnet)
end
