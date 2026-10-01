#=
Compare the message passing results to a brute-force enumeration of all internal states,
on a small tree with a profile model (sites are independent).
For each site, the joint probability of an assignment `x` of states to all nodes is
    P(x) = π[x_root] * ∏_{branches a→n} T_n[x_a, x_n]
with leaves fixed to their observed states.
=#
using AncestralSequenceReconstruction
using LinearAlgebra
using Random
using Test
using TreeTools

const NWK = "((A:0.3,B:0.5)I1:0.2,(C:0.7,D:0.1)I2:0.4)R;"

function make_tree(leaf_sequences, q; alphabet)
    L = length(first(values(leaf_sequences)))
    tree = parse_newick_string(NWK; node_data_type = () -> ASR.AState{q}(; L))
    return ASR.initialize_tree(tree, leaf_sequences; alphabet)
end

# Transition matrix T[a,b] = P(b|a) for the branch above `node` at site `pos`
function brute_force_transition(model, node, pos)
    π = model.P[pos]
    q = length(π)
    μt = model.μ * branch_length(node)
    return if model.with_code
        T = zeros(q, q)
        ASR.set_transition_matrix!(T, μt, π; with_code=true)
    else
        ν = exp(-μt)
        ν * I + (1 - ν) * ones(q) * π'
    end
end

"""
Return `(labels, P)` where `P` is an array with one dimension per internal node in `labels`,
and `P[x...]` is the joint probability of internal states `x` and the leaf data at `pos`.
"""
function brute_force_site(tree, model::ASR.ProfileModel{q}, pos) where q
    labels = map(label, internals(tree))
    P = zeros(Float64, ntuple(_ -> q, length(labels)))
    for x in CartesianIndices(P)
        state = Dict(zip(labels, Tuple(x)))
        foreach(n -> state[label(n)] = n.data.sequence[pos], leaves(tree))
        p = model.P[pos][state[label(tree.root)]]
        for n in nodes(tree; skiproot=true)
            T = brute_force_transition(model, n, pos)
            p *= T[state[label(ancestor(n))], state[label(n)]]
        end
        P[x] = p
    end
    return labels, P
end

# marginal distribution of the internal node at dimension `d` of `P`
marginal(P, d) = vec(sum(P; dims = filter(!=(d), 1:ndims(P)))) / sum(P)

function test_against_brute_force(tree, model; joint_lk_broken=false)
    L = length(model.P)
    sites = [brute_force_site(tree, model, pos) for pos in 1:L]
    labels = first(sites[1])

    @testset "Likelihood (marginal)" begin
        loglk = ASR.tree_likelihood!(copy(tree), model, ASRMethod(; joint=false, ML=true))
        @test loglk ≈ sum(((_, P),) -> log(sum(P)), sites) rtol=1e-10
    end

    @testset "Likelihood (joint ML)" begin
        loglk = ASR.tree_likelihood!(copy(tree), model, ASRMethod(; joint=true, ML=true))
        exact = sum(((_, P),) -> log(maximum(P)), sites)
        if joint_lk_broken
            @test_broken isapprox(loglk, exact; rtol=1e-10)
        else
            @test loglk ≈ exact rtol=1e-10
        end
    end

    @testset "Marginal ML: posteriors and states" begin
        t = infer_ancestral(tree, model, ASRMethod(; joint=false, ML=true))
        for pos in 1:L, (d, lab) in enumerate(labels)
            exact = marginal(sites[pos][2], d)
            @test ASR.site_posterior(t[lab].data, pos) ≈ exact rtol=1e-8
            @test t[lab].data.sequence[pos] == argmax(exact)
        end
    end

    @testset "Joint ML: states" begin
        t = infer_ancestral(tree, model, ASRMethod(; joint=true, ML=true))
        for pos in 1:L
            best = Tuple(argmax(sites[pos][2]))
            @test map(lab -> t[lab].data.sequence[pos], labels) == collect(best)
        end
    end
end

@testset "Profile model, q=4" begin
    Random.seed!(1)
    q, L = 4, 3
    model = ASR.ProfileModel([normalize(rand(q) .+ 0.1, 1) for _ in 1:L]; μ = 1.3)
    leaf_sequences = Dict(
        "A" => [1, 2, 3], "B" => [1, 2, 4], "C" => [2, 3, 3], "D" => [2, 1, 3],
    )
    tree = make_tree(leaf_sequences, q; alphabet=:nt)
    test_against_brute_force(tree, model)
end

@testset "Profile model with genetic code, q=21" begin
    Random.seed!(2)
    q, L = 21, 2
    model = ASR.ProfileModel(
        [normalize(rand(q) .+ 0.1, 1) for _ in 1:L]; with_code=true, alphabet=:aa,
    )
    leaf_sequences = Dict("A" => "AV", "B" => "AI", "C" => "VI", "D" => "TV")
    tree = make_tree(leaf_sequences, q; alphabet=:aa)
    test_against_brute_force(tree, model)
end

@testset "Sampling distributions" begin
    # One site, compare empirical frequencies to the exact posterior
    Random.seed!(3)
    q = 4
    model = ASR.ProfileModel([normalize(rand(q) .+ 0.1, 1)])
    leaf_sequences = Dict("A" => [1], "B" => [1], "C" => [2], "D" => [3])
    tree = make_tree(leaf_sequences, q; alphabet=:nt)
    labels, P = brute_force_site(tree, model, 1)
    P ./= sum(P)
    M = 20_000

    # Joint sampling: frequency of full configurations of internal nodes
    counts = zeros(Float64, size(P))
    for _ in 1:M
        t = infer_ancestral(tree, model, ASRMethod(; joint=true, ML=false))
        counts[map(lab -> t[lab].data.sequence[1], labels)...] += 1
    end
    @test sum(abs, counts / M - P) / 2 < 0.03 # total variation distance

    # Marginal sampling: frequency of states at each node
    counts = zeros(Float64, q, length(labels))
    for _ in 1:M
        t = infer_ancestral(tree, model, ASRMethod(; joint=false, ML=false))
        for (d, lab) in enumerate(labels)
            counts[t[lab].data.sequence[1], d] += 1
        end
    end
    for d in eachindex(labels)
        @test sum(abs, counts[:, d] / M - marginal(P, d)) / 2 < 0.02
    end
end
