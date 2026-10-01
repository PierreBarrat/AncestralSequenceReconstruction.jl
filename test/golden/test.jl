#=
Compare to reference outputs saved by `generate.jl` (code before the memory refactoring).
=#
using Test
include(joinpath(@__DIR__, "setup.jl"))

golden = JLD2.load(GOLDEN_FILE)

function compare_to_golden(tree, reference; atol=1e-8)
    current = summarize(tree)
    @test keys(current) == keys(reference)
    for (node, ref) in reference
        @test current[node].sequence == ref.sequence
        @test isapprox(current[node].posterior, ref.posterior; atol)
    end
end

@testset "AR model, marginal ML" begin
    strategy = ASRMethod(; joint=false, ML=true)
    tree, _ = infer_ancestral(TREE_FILE, FASTA_FILE, ar_model, strategy)
    compare_to_golden(tree, golden["ar_marginal_ML"])
end

@testset "AR model, seeded sampling: $name" for (name, strategy) in [
    "ar_marginal_sampling" => ASRMethod(; joint=false, ML=false),
    "ar_joint_sampling" => ASRMethod(; joint=true, ML=false),
]
    Random.seed!(SEED)
    tree, _ = infer_ancestral(TREE_FILE, FASTA_FILE, ar_model, strategy)
    compare_to_golden(tree, golden[name])
end

@testset "Profile model: likelihood and branch lengths" begin
    tree = initial_profile_tree()
    loglk = ASR.tree_likelihood!(tree, profile_model, ASRMethod(; joint=false))
    @test loglk ≈ golden["profile_loglk"] rtol=1e-10

    lk = ASR.optimize_branch_length!(tree, profile_model, ASRMethod(; joint=false))
    @test lk[end] ≈ golden["profile_branch_opt_lk"][end] rtol=1e-6
    for (node, t) in golden["profile_branch_lengths"]
        @test branch_length(tree[node]) ≈ t atol=1e-4
    end
end
