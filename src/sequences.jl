#######################################################################################
################################# Sequences and trees #################################
#######################################################################################

"""
    fasta_to_tree!(tree::Tree{AState}, fastafile::AbstractString)

Add sequences of `fastafile` to nodes of `tree`.
"""
function fasta_to_tree!(
    tree::Tree{AState{q}}, fastafile::AbstractString;
    warn = true, default=missing, alphabet = :aa,
) where q
    all_headers_in_tree = true
    all_leaves_in_fasta = true

    reader = open(FASTA.Reader, fastafile)
    record = FASTA.Record()
    while !eof(reader)
        read!(reader, record)
        if in(identifier(record), tree)
            seq = sequence_to_intvec(sequence(record); alphabet)
            if maximum(seq) > q
                error("""
                    $(typeof(Tree)) with $q states, found $(maximum(seq)) in sequence
                    Problem with alphabet?
                """)
            end
            tree[identifier(record)].data.sequence .= seq
        else
            all_headers_in_tree = false
        end
    end
    close(reader)

    for n in leaves(tree)
        if any(isnothing, n.data.sequence)
            all_leaves_in_fasta = false
            break
        end
    end
    !all_leaves_in_fasta && @warn "Not all leaves had a corresponding sequence \
        in the alignment (file: $fastafile)."
    !all_headers_in_tree && @warn "Some sequence headers in the alignment are \
        not found in the tree (file: $fastafile)."
    return nothing
end

"""
    sequences_to_tree!(tree::Tree{<:AState}, seqmap; alphabet=:aa, safe)

Iterating `seqmap` should yield pairs `label => sequence`.
"""
function sequences_to_tree!(
    tree::Tree{AState{q}}, seqmap;
    alphabet=:aa, safe=true,
) where q
    for (label, seq) in seqmap
        if safe && !isleaf(tree[label])
            error("Cannot assign an observed sequence to internal node. Use `safe=false`?")
        end
        tree[label].data = AState{q}(;
            L = length(seq), sequence = sequence_to_intvec(seq; alphabet)
        )
    end
    if any(n -> !hassequence(n.data), leaves(tree))
        @warn "Somes leaves do not have sequences"
    end
    return nothing
end

"""
    initialize_tree(tree::Tree, seqmap; alphabet=:aa)
"""
function initialize_tree(tree::Tree, seqmap; alphabet=:aa)
    L = first(seqmap)[2] |> length
    # q = alphabet_size(alphabet)
    q = length(Alphabet(alphabet))
    tree = convert(Tree{AState{q}}, tree)
    foreach(n -> n.data = AState{q}(;L), nodes(tree))
    sequences_to_tree!(tree, seqmap; alphabet)
    return tree
end
