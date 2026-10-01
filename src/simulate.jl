module Simulate

using AncestralSequenceReconstruction
using FASTX
using StatsBase: wsample
using TreeTools

"""
    evolve(tree::Tree, model; alphabet, leaves_fasta, internals_fasta, root)

Simulate sequences along `tree` using `model`.
Return the tree, the leaf sequences and the internal sequences, as a named tuple.
Write output to alignments `leaves_fasta` and `internals_fasta`.
`tree` can be any type of tree: it will be converted to `Tree{AState}` inside.
"""
function evolve(
    tree::Tree, model::EvolutionModel{q};
    leaves_fasta = "", internals_fasta = "", kwargs...
) where q
    L = length(model)
    tc = convert(Tree{ASR.AState{q}}, tree)
    foreach(n -> n.data = ASR.AState{q}(;L), nodes(tc))
    leaf_sequences, internal_sequences = evolve!(tc, model; kwargs...)

    # write sequences to fasta if asked
    if !isempty(leaves_fasta)
        FASTAWriter(open(leaves_fasta, "w")) do writer
            for (name, seq) in leaf_sequences
                write(writer, FASTARecord(name, seq))
            end
        end
    end
    if !isempty(internals_fasta)
        FASTAWriter(open(internals_fasta, "w")) do writer
            for (name, seq) in internal_sequences
                write(writer, FASTARecord(name, seq))
            end
        end
    end

    return (leaf_sequences=leaf_sequences, internal_sequences=internal_sequences, tree=tc)
end

function evolve!(
    tree::Tree{ASR.AState{q}}, model::EvolutionModel;
    alphabet=model.alphabet, root=nothing, translate=true,
) where q
    # simulation, site by site
    for pos in ASR.ordering(model)
        # set equilibrium frequencies and transition matrices for all branches
        ASR.set_transition_matrix!(tree, model, pos)
        root_state = isnothing(root) ? wsample(tree.root.data.weights.π) : root[pos]
        sample_from_ancestor!(tree.root, root_state, pos)
    end

    # collect sequences
    leaf_sequences = map(leaves(tree)) do n
        s = translate ? ASR.intvec_to_sequence(n.data.sequence; alphabet) : n.data.sequence
        label(n) => s
    end |> Dict
    internal_sequences = map(internals(tree)) do n
        s = translate ? ASR.intvec_to_sequence(n.data.sequence; alphabet) : n.data.sequence
        label(n) => s
    end |> Dict

    return leaf_sequences, internal_sequences
end

# set the state of `node` at `pos`, then sample its children from the transition matrices
function sample_from_ancestor!(node::TreeNode, state::Int, pos)
    node.data.sequence[pos] = state
    for c in children(node)
        sample_from_ancestor!(c, wsample(view(c.data.weights.T, state, :)), pos)
    end
    return nothing
end

end
