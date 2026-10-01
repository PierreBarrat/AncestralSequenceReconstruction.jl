####### ProfileModel #######

L, q = (3, 2)
Pmat = [
    1/3 1/2 1/4;
    2/3 1/2 3/4;
]
Pvec = [[1/3, 2/3], [1/2, 1/2], [1/4, 3/4]]
@testset "ProfileModel basics" begin
    I2 = ASR.ProfileModel(Pvec)
    @test length(ASR.ordering(I2)) == L
    @test typeof(I2) == ASR.ProfileModel{q}

    I1 = ASR.ProfileModel(Pmat)
    @test length(ASR.ordering(I1)) == L
    @test typeof(I1) == ASR.ProfileModel{q}


    @test I1.P == I2.P
    @test ASR.ordering(I2) == ASR.ordering(I2)

    @test_throws AssertionError ASR.ProfileModel(Pvec .+ [0.01*rand(q) for _ in 1:L]) # should not use rand but well...
    @test_throws AssertionError ASR.ProfileModel(Pmat .+ rand(q, L))

    JC = ASR.JukesCantor(1)
    @test JC.P[1] == 1/4 * ones(Float64, 4)
end

####### AutoregressiveModel #######

dir = dirname(@__FILE__)
# dir = "basics/"

arnet = JLD2.load(joinpath(dir, "arnet.jld2"))["arnet"] # PF00072
L = size(arnet.H, 1) + 1
q = 21

tree = read_tree(joinpath(dir, "tree_long_branches.nwk"); node_data_type = () -> ASR.AState{q}(;L))
ASR.fasta_to_tree!(tree, joinpath(dir, "alignment_long_branches.fasta"))


ar_model = AutoRegressiveModel(arnet)
x1 = convert(Vector{Int}, tree["A"].data.sequence);
x2 = convert(Vector{Int}, tree["B"].data.sequence);
local_p_x1 = arnet(x1);
local_p_x2 = arnet(x2);

@testset "local field" begin
    t = copy(tree)

    # π is only stored for the current site: check it right after setting it
    for i in ASR.ordering(ar_model)
        ASR.set_π!(t["A"].data, ar_model, i)
        ASR.set_π!(t["B"].data, ar_model, i)
        @test isapprox(local_p_x1[i], t["A"].data.weights.π[x1[i]]; rtol = 1e-6)
        @test isapprox(local_p_x2[i], t["B"].data.weights.π[x2[i]]; rtol = 1e-6)
    end
end

@testset "set π internal node" begin
    t = copy(tree)
    perm = ASR.ordering(ar_model)
    i = perm[1]
    j = perm[2]

    ASR.set_π!(t["R"].data, ar_model, i)
    @test t["R"].data.weights.π ≈ ar_model.arnet.p0

    @test_throws ErrorException ASR.set_π!(t["R"].data, ar_model, j)
    t["R"].data.sequence[i] = x1[i]
    @test isnothing(ASR.set_π!(t["R"].data, ar_model, j))
    @test local_p_x1[j] ≈ t["R"].data.weights.π[x1[j]]
end

@testset "Reconstruction at root: energy" begin
    # the tree has very long branches
    # average energy of the root should be the same as eq.
    Seq = ArDCA.sample(arnet, 1000)
    lk_mean = mean(s -> ArDCA.loglikelihood(convert(Vector{Int}, s), arnet), eachcol(Seq))

    root_sequences = map(1:200) do _
       t = infer_ancestral(tree, ar_model, ASRMethod(ML=false, optimize_branch_length=false))
       convert(Vector{Int}, t.root.data.sequence)
    end
    lk_root = mean(s -> ArDCA.loglikelihood(s, arnet), root_sequences)

    @test isapprox(lk_root, lk_mean, rtol = 2*1e-2)

    best_root = let
        strat = ASRMethod(ML=true, joint=false, optimize_branch_length=false)
        t = infer_ancestral(tree, ar_model, strat)
        convert(Vector{Int}, t.root.data.sequence)
    end
    lk_best_root = ArDCA.loglikelihood(best_root, arnet)

    @test lk_best_root > lk_mean
end
