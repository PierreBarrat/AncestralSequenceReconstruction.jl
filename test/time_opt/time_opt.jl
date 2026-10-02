using AncestralSequenceReconstruction
using TreeTools
using Test

##########################################################################################
##########################################################################################
##########################################################################################
L = 20
q = 4
TT() = ASR.AState{q}(; L)

# Branch above C --> infinite length
# others --> zero length
nwk = "(((A,B),(C,D)I1),(E,F));"
leaf_sequences = Dict(l => repeat([1], L) for l in map(string, collect("ABCDEF")))
leaf_sequences["C"] = repeat([2], L)

model = ASR.JukesCantor(L);

# initial tree
tree = begin
    tree = parse_newick_string(nwk; node_data_type = TT)
    foreach(n -> branch_length!(n, 1.), nodes(tree; skiproot=true))
    ASR.initialize_tree(tree, leaf_sequences; alphabet=:nt)
end

@testset "Case 1" begin
    tree_test, lk = ASR.optimize_branch_length(tree, model)
    for n in nodes(tree_test; skiproot = true)
        if label(n) == "C"
            @test branch_length(n) == ASR.BRANCH_UPR_BOUND(model)
        else
            @test branch_length(n) == ASR.BRANCH_LWR_BOUND(L)
        end
    end
end

# now, branches I1 --> C and I1 --> D should be 0
# anc(I1) --> I1 should be infinite
# others should be 0
tree["D"].data.sequence .= 2
@testset "Case 2" begin
    tree_test, lk = ASR.optimize_branch_length(tree, model)
    for n in nodes(tree_test; skiproot = true)
        if label(n) == "I1"
            @test branch_length(n) == ASR.BRANCH_UPR_BOUND(model)
        else
            @test branch_length(n) == ASR.BRANCH_LWR_BOUND(L)
        end
    end
end

##########################################################################################
##########################################################################################
##########################################################################################


L = 2
model = ASR.JukesCantor(L);
q = 4
TT() = ASR.AState{q}(; L)

# Branch above C --> infinite length
# others --> zero length
nwk = "(A,B);"
leaf_sequences = Dict("A" => [1,2], "B" => [1,1])


# initial tree
tree = begin
    tree = parse_newick_string(nwk; node_data_type = TT)
    foreach(n -> branch_length!(n, 1.), nodes(tree; skiproot=true))
    ASR.initialize_tree(tree, leaf_sequences; alphabet=:nt)
end
tree_test, lk = ASR.optimize_branch_length(tree, model)

@testset "Analytical 1" begin
    @test ≈(
        map(branch_length, nodes(tree_test; skiproot=true)) |> sum,
        -1/model.μ * log(0.5/1.5);
        rtol = 1e-3
    )
end
