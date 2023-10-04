const AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
const NT_ALPHABET = "ACGT"

aa_alphabet_names = (:aa, :AA, :aminoacids, :amino_acids)
nt_alphabet_names = (:nt, :nucleotide, :dna)

function alphabet_map(alphabet)
    return if alphabet in aa_alphabet_names
        AA_ALPHABET
    elseif alphabet in nt_alphabet_names
        NT_ALPHABET
    else
        unknown_alphabet_error(alphabet)
    end
end
alphabet_size(alphabet) = length(alphabet_map(alphabet))

"""
    compute_mapping(s::AbstractString)

`Dict(i => c for (i,c) in enumerate(s))`.
"""
compute_mapping(s::AbstractString) = Dict(c => i for (i, c) in enumerate(s))

const AA_MAPPING = compute_mapping(AA_ALPHABET)
const NT_MAPPING = compute_mapping(NT_ALPHABET)

function unknown_alphabet_error(a)
    throw(ArgumentError("""
        Incorrect alphabet type `$a`.
        Choose from `$aa_alphabet_names` or `$nt_alphabet_names`.
    """))
end

function sequence_to_intvec(s; alphabet = :aa)
    return if alphabet in aa_alphabet_names
        map(c -> AA_MAPPING[Char(c)], collect(s))
    elseif alphabet in nt_alphabet_names
        map(c -> NT_MAPPING[Char(c)], collect(s))
    else
        unknown_alphabet_error(alphabet)
    end
end
sequence_to_intvec(s::AbstractVector{<:Integer}; kwargs...) = s

function intvec_to_sequence(X::AbstractVector; alphabet=:aa)
    amap = alphabet_map(alphabet)
    return map(x -> amap[x], X) |> String
end

"""
    fasta_to_tree!(tree::Tree{AState}, fastafile::AbstractString)

Add sequences of `fastafile` to nodes of `tree`.
"""
function fasta_to_tree!(
    tree::Tree{AState{L,q}}, fastafile::AbstractString, key = :seq;
    warn = true, default=missing, alphabet = :aa
) where {L,q}
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
            tree[identifier(record)].data.sequence = seq
        else
            all_headers_in_tree = false
        end
    end
    close(reader)

    for n in leaves(tree)
        if isempty(n.data.sequence)
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
    tree::Tree{AState{L,q}}, seqmap;
    alphabet=:aa, safe=true,
) where {L,q}
    for (label, seq) in seqmap
        if safe && !isleaf(tree[label])
            error("Cannot assign an observed sequence to internal node. Use `safe=false`?")
        end
        if length(seq) != L
            error("Sequence of incorrect length: got $(length(seq)), expected $L")
        end

        tree[label].data.sequence = sequence_to_intvec(seq; alphabet)
    end
    if any(n -> !hassequence(n.data), leaves(tree))
        @warn "Somes leaves do not have sequences"
    end
    return nothing
end

function initialize_tree(tree::Tree, seqmap; alphabet=:aa)
    L = first(seqmap)[2] |> length
    q = alphabet_size(alphabet)
    tree = convert(Tree{AState{L,q}}, tree)
    sequences_to_tree!(tree, seqmap; alphabet)
    return tree
end

