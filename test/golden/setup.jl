# Shared by `generate.jl` and `test.jl`: data, models and helpers.
using AncestralSequenceReconstruction
using ArDCA
using JLD2
using Random
using TreeTools

const EXAMPLE_DIR = joinpath(@__DIR__, "..", "..", "example", "PF00014", "reconstruction")
const TREE_FILE = joinpath(EXAMPLE_DIR, "tree_iqtree.nwk")
const FASTA_FILE = joinpath(EXAMPLE_DIR, "PF00014_mgap6_subalignment.fasta")
const ARNET_FILE = joinpath(EXAMPLE_DIR, "arnet_PF00014_lJ0.01_lH0.001.jld2")
const GOLDEN_FILE = joinpath(@__DIR__, "golden.jld2")
const SEED = 42

const ar_model = AutoRegressiveModel(JLD2.load(ARNET_FILE)["arnet"])
const profile_model = ASR.ProfileModel(FASTA_FILE; pc=0.1, alphabet=:aa)

# Internal sequences and posteriors at all internal nodes, keyed by node label
function summarize(tree)
    return Dict(
        label(n) => (
            sequence = copy(n.data.sequence),
            posterior = mapreduce(i -> ASR.site_posterior(n.data, i), hcat, 1:n.data.L),
        )
        for n in internals(tree)
    )
end

# Tree with leaf sequences attached, for the profile model
function initial_profile_tree()
    seqmap = ASR.FASTAReader(open(FASTA_FILE, "r")) do reader
        map(rec -> ASR.identifier(rec) => ASR.sequence(rec), reader)
    end
    return ASR.initialize_tree(read_tree(TREE_FILE), seqmap; alphabet=:aa)
end
