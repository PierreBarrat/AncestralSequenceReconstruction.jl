const AA_ALPHABET = "-ACDEFGHIKLMNPQRSTVWY"
const NT_ALPHABET = "ACGT"

aa_alphabet_names = (:aa, :AA, :aminoacids, :amino_acids)
nt_alphabet_names = (:nt, :nucleotide, :dna)

"""
    compute_mapping(s::AbstractString)

`Dict(i => c for (i,c) in enumerate(s))`.
"""
compute_mapping(s::AbstractString) = Dict(c => i for (i, c) in enumerate(s))

const AA_MAPPING = compute_mapping(AA_ALPHABET)
const NT_MAPPING = compute_mapping(NT_ALPHABET)

function sequence_to_int(s::AbstractString; alphabet = :aa)
    return if alphabet in aa_alphabet_names
        map(c -> AA_MAPPING[c], collect(s))
    elseif alphabet in nt_alphabet_names
        map(c -> NT_MAPPING[c], collect(s))
    else
        throw(ArgumentError("""
        Incorrect alphabet type `$alphabet`. \
            Choose from `$aa_alphabet_names` or `$nt_alphabet_names`.
        """))
    end
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
            seq = sequence_to_int(sequence(record); alphabet)
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



