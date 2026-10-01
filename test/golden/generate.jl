#=
Generate reference ("golden") outputs used by `test/golden/test.jl`.

These were produced with the code *before* the memory refactoring, and serve to check
that the refactoring does not change results. Only re-run this script if a change of
results is intended (and understood).

Usage: from the package root, in an environment where AncestralSequenceReconstruction
is `dev`-ed and ArDCA, JLD2 and TreeTools are installed,
    julia --project=<that environment> test/golden/generate.jl
=#

include(joinpath(@__DIR__, "setup.jl"))

golden = Dict{String, Any}()

# Marginal ML reconstruction with the autoregressive model: deterministic
let
    strategy = ASRMethod(; joint=false, ML=true)
    tree, _ = infer_ancestral(TREE_FILE, FASTA_FILE, ar_model, strategy)
    golden["ar_marginal_ML"] = summarize(tree)
end

# Bayesian sampling with fixed seeds: exact match requires the same order of RNG calls
for (name, strategy) in [
    "ar_marginal_sampling" => ASRMethod(; joint=false, ML=false),
    "ar_joint_sampling" => ASRMethod(; joint=true, ML=false),
]
    Random.seed!(SEED)
    tree, _ = infer_ancestral(TREE_FILE, FASTA_FILE, ar_model, strategy)
    golden[name] = summarize(tree)
end

# Likelihood of the profile model + branch length optimization
let
    tree = initial_profile_tree()
    golden["profile_loglk"] = ASR.tree_likelihood!(tree, profile_model, ASRMethod(; joint=false))
    lk = ASR.optimize_branch_length!(tree, profile_model, ASRMethod(; joint=false))
    golden["profile_branch_opt_lk"] = lk
    golden["profile_branch_lengths"] = Dict(label(n) => branch_length(n) for n in nodes(tree; skiproot=true))
end

JLD2.save(GOLDEN_FILE, golden)
@info "Saved golden outputs to $GOLDEN_FILE"
