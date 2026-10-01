#=
Memory layout
-------------
Sites are processed one at a time (see `pruning_alg!`). Each node therefore needs
working memory (`BranchWeights`) for a *single* site, which is overwritten for every
site. Only the results of the reconstruction are stored for all sites, in `AState`:
the reconstructed `sequence` and the `posterior` distribution at each site.
=#

#######################################################################################
################################### Branch weights ####################################
#######################################################################################

"""
    BranchWeights{q}

Working memory of the message passing algorithm for one node, at the site currently
being processed. Notation: `n` is the node, `A` its ancestor.
- `π`: equilibrium probability of each state at `n`
- `T`: transition matrix of the branch `A → n`: `T[a, b] = P(n = b | A = a)`
- `v`: down-likelihood, probability of the data below `n`, as a function of the state of `n`
- `u`: up-likelihood, probability of the data *not* below `n`, as a function of the state
  of `A`
- `Fv`, `Fu`: log-normalization of `v` and `u`: actual likelihoods are `v * exp(Fv)`
- `lm_up`: log of the message `n → A`, as a function of the state of `A`
- `best_state`: for joint ML, best state of `n` as a function of the state of `A`
"""
mutable struct BranchWeights{q}
    π :: Vector{Float64}
    T :: Matrix{Float64}
    v :: Vector{Float64}
    u :: Vector{Float64}
    Fv :: Float64
    Fu :: Float64
    lm_up :: Vector{Float64}
    best_state :: Vector{Int}
end
function BranchWeights{q}() where q
    return BranchWeights{q}(
        fill(1/q, q), diagm(ones(Float64, q)), ones(q), ones(q), 0., 0., zeros(q), zeros(Int, q)
    )
end

function Base.copy(W::BranchWeights{q}) where q
    return BranchWeights{q}(
        copy(W.π), copy(W.T), copy(W.v), copy(W.u), W.Fv, W.Fu, copy(W.lm_up),
        copy(W.best_state),
    )
end

function reset_weights!(W::BranchWeights)
    W.v .= 1
    W.u .= 1
    W.Fv = 0.
    W.Fu = 0.
    W.lm_up .= 0
    W.best_state .= 0
    return nothing
end

#######################################################################################
################################### Ancestral state ###################################
#######################################################################################

"""
    AState{q}

Data attached to each node of the tree.
- `L`: sequence length
- `sequence`: observed (leaves) or reconstructed (internal nodes) sequence, as integers
- `posterior`: `posterior[:, i]` is the distribution of states at site `i` computed during
  the last reconstruction. See `site_posterior`.
- `weights`: working memory for the site being processed, see `BranchWeights`
"""
@kwdef struct AState{q} <: TreeNodeData
    L::Int = 1
    sequence :: Vector{Union{Nothing, Int}} = Vector{Nothing}(undef, L)
    posterior :: Matrix{Float64} = fill(1/q, q, L)
    weights :: BranchWeights{q} = BranchWeights{q}()

    function AState{q}(L, sequence, posterior, weights) where q
        @assert length(sequence) == L "Expected sequence of length $L, got $(length(sequence))"
        @assert size(posterior) == (q, L) "Expected posterior of size $((q, L)), got $(size(posterior))"
        return new{q}(L, sequence, posterior, weights)
    end
end

function Base.copy(state::AState{q}) where q
    return AState{q}(;
        L = state.L,
        sequence = copy(state.sequence),
        posterior = copy(state.posterior),
        weights = copy(state.weights),
    )
end

"""
    site_posterior(state::AState, pos::Int)

Posterior distribution over states at site `pos`, as computed during the last
reconstruction.
"""
site_posterior(state::AState, pos::Int) = view(state.posterior, :, pos)

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
####################################### Alphabet ######################################
#######################################################################################


#=
Defining a new alphabet:
- define the mapping string (like _AA_ALPHABET)
- define the alphabet from the string (like const aa_alphabet = ...)
- define the symbol names for the alphabet
- update the function Alphabet(::Symbol)
- if relevant, update default_alphabet(::Int)
=#

const _AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
const _ALTERNATIVE_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY-"
const _NT_ALPHABET_NOGAP = "ACGT"
const _SPIN_ALPHABET = "01"


_alphabet_mapping(s::AbstractString) = Dict(c => i for (i, c) in enumerate(s))

@kwdef struct Alphabet
    string::String
    mapping::Dict{Char, Int} = _alphabet_mapping(string)
end
Alphabet(s::AbstractString) = Alphabet(; string = s, mapping = _alphabet_mapping(s))
Alphabet(a::Alphabet) = a
function Alphabet(mapping::AbstractDict{Char, Int})
    str = Vector{Char}(undef, length(mapping))
    for (c, i) in mapping
        str[i] = c
    end
    return Alphabet(;string = prod(str), mapping)
end
function Alphabet(rev_mapping::AbstractDict{Int, Char})
    mapping = Dict{Char, Int}(c => i for (i,c) in rev_mapping)
    return Alphabet(mapping)
end

"""
    reverse_mapping(A::Alphabet)

Return a `Dict{Int, Char}`.
"""
reverse_mapping(A::Alphabet) = Dict(i => c for (c,i) in A.mapping)

const aa_alphabet = Alphabet(_AA_ALPHABET)
const aa_alphabet_names = (:aa, :AA, :aminoacids, :amino_acids)

const alternative_aa_alphabet = Alphabet(_ALTERNATIVE_AA_ALPHABET)
const alternative_aa_alphabet_names = (:ardca_aa, :aa_ardca, :alternative_aa)

const nt_alphabet = Alphabet(_NT_ALPHABET_NOGAP)
const nt_alphabet_names = (:nt, :nucleotide, :dna)

const spin_alphabet = Alphabet(_SPIN_ALPHABET)
const sping_alphabet_names = (:spin,)

# Default alphabet from symbol
function Alphabet(alphabet::Symbol)
    return if alphabet in aa_alphabet_names
        aa_alphabet
    elseif alphabet in alternative_aa_alphabet_names
        alternative_aa_alphabet
    elseif alphabet in nt_alphabet_names
        nt_alphabet
    elseif alphabet in sping_alphabet_names
        spin_alphabet
    else
        unknown_alphabet_error(alphabet)
    end
end

Base.convert(::Type{Alphabet}, x::Symbol) = Alphabet(x)

Base.length(a::Alphabet) = length(a.string)

# Default alphabet for given size q
"""
    default_alphabet(q::Int)

Pick an `Alphabet` based on the value of `a`:
- `21` --> `ASR.aa_alphabet`
- `4` --> `ASR.nt_alphabet`
- `2` --> `ASR.spin_alphabet`
"""
function default_alphabet(q::Int)
    return if q == 21
        aa_alphabet
    elseif q == 4
        nt_alphabet
    elseif q == 2
        spin_alphabet
    else
        error("Not default alphabet for q=$q")
    end
end

function unknown_alphabet_error(a)
    throw(ArgumentError("""
        Unrecognized alphabet name `$a`.
        Choose from `$aa_alphabet_names`, `$nt_alphabet_names`, `$sping_alphabet_names`, or provide a string, or construct with `Alphabet`.
    """))
end

# from string to `Vector{Int}`
function sequence_to_intvec(s::AbstractString; alphabet = :aa)
    return sequence_to_intvec(s, Alphabet(alphabet))
end
function sequence_to_intvec(s::AbstractString, alphabet::Alphabet)
    return map(c -> alphabet.mapping[Char(c)], collect(s))
end
sequence_to_intvec(s::AbstractVector{<:Integer}; kwargs...) = s

# from `Vector{Int}` to string
function intvec_to_sequence(X::AbstractVector; alphabet=:aa)
    return intvec_to_sequence(X, Alphabet(alphabet))
end
function intvec_to_sequence(X::AbstractVector, alphabet::Alphabet)
    return map(x -> alphabet.string[x], X) |> String
end


#######################################################################################
###################################### ASR Method #####################################
#######################################################################################

"""
    ASRMethod

- `joint::Bool`: joint or marginal inference. Default `false`.
- `ML::Bool`: maximum likelihood or bayesian (sampling). Default `false` (*i.e.* sampling).

The four combinations are:
- `joint=false, ML=true`: at each node, pick the state with the highest marginal posterior.
- `joint=false, ML=false`: at each node, sample a state from the marginal posterior,
  independently of other nodes.
- `joint=true, ML=true`: the most likely joint assignment of states to all nodes
  (Pupko et al., 2000).
- `joint=true, ML=false`: sample a joint assignment of states to all nodes from the
  posterior, by sampling each node given the state of its ancestor.
- `verbosity :: Int`: verbosity level. Unless you're debugging something, `<=2` Default `0`.
- `optimize_branch_length`: optimize the branch lengths of the tree according to the model.
  Default `false`.
- `optimize_branch_scale`: optimally scale the branches of the input tree, keeping their
  relative lengths fixed. Incompatible with `optimize_branch_length`. Default `false`.
- `optimize_branch_length_cycles`: number of optimization cycles (see Felsenetein's book). Default `3`.
- `repetitions :: Int`: Number of repetitions for the reconstruction.
  Should be set to 1 for the ML reconstruction.
  For Bayesian reconstruction (*i.e.* `ML=false`), this allows for sampling of likely
  ancestors.
  If higher than 1, the `infer_ancestral` function will return an array of sequence
  mappings, one for each reconstruction.
  An array of output fasta files should also be provided.
"""
@kwdef mutable struct ASRMethod
    joint::Bool = false
    ML::Bool = false
    verbosity::Int = 0
    optimize_branch_length::Bool = false
    optimize_branch_length_cycles::Int = 3
    optimize_branch_scale::Bool = false
    repetitions::Int = 1
end
