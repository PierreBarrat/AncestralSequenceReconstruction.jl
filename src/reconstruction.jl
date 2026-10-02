"""
    infer_ancestral(
        newick_file::AbstractString, fastafile::AbstractString, model, strategy; kwargs...
        outfasta = nothing, outnewick = nothing, outtable = nothing,
        alignment_per_node = false,
        node_list = nothing,
        alignment_per_node_name = name -> fasta_from_node_name(name, outfasta),
    )
    infer_ancestral(
        tree::Tree, leaf_sequences::AbstractVector, model, strategy;
    )

Writing output:
- if `outnewick` is a `String`, write the tree to the corresponding file. Default `nothing`.
- if `outfasta` is a `String`, write the reconstructed internal sequences to the
  corresponding file in Fasta format.
  If it is a `Vector` of `String`, then its length should be equal to
  `strategy.repetitions`, and reconstruction from each repetition will be written to
  the corresponding file in the vector. Default `nothing`.
- `outtable` is for a table in the style of Iqtree's `.state` file: it contains
  the posterior distribution of states at each internal node. Default `nothing`.
- `alignment_per_node`: write one alignment for each internal node in `node_list`
  (if `nothing`, all internals of the tree).
  The names of alignments can be set functionally by providing a function, *e.g.*
  `alignment_per_node_name(node_name) -> "something.fasta"`.
  By default, the names are based on the string `outfasta`: `outfasta_node_name.fasta`.
  Default `false`.
- `node_list`: used if `alignment_per_node`. Default `nothing`.
"""
function infer_ancestral(
    newick_file::AbstractString, fastafile::AbstractString, model, strategy;
    kwargs...
)
    # read sequences
    sequences = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> identifier(rec) => sequence(rec),reader)
    end

    tree = read_tree(newick_file)

    return infer_ancestral(tree, sequences, model, strategy; kwargs...)
end

function infer_ancestral(
    tree::Tree, leaf_sequences::Union{AbstractVector, AbstractDict}, model, strategy;
    outfasta = nothing, outnewick = nothing, outtable = nothing,
    alignment_per_node = false,
    node_list = nothing,
    alignment_per_node_name = name -> fasta_from_node_name(name, outfasta),
    table_style = :short,
)
    # pre-emptive check of parameters
    if strategy.repetitions>1 && strategy.ML
        @warn "Asked for more than one repetition with ML strategy. Are you sure?"
    end

    model.with_code && @warn "Using genetic code - Gaps are not well modeled"

    if alignment_per_node && isnothing(node_list)
        @warn """Asked for one alignment per node, but no `node_list` kwarg provided.
        Outputting alignment for every internal node in the tree."""
        node_list = map(label, internals(tree))
    end

    # set parameters and build tree of correct type
    L = length(first(leaf_sequences)[2])
    q = length(model.alphabet)
    if any(x -> length(x[2]) != L, leaf_sequences)
        error("All leaf sequences must have the same length")
    end

    tree = convert(AState{q}, tree)
    foreach(n -> n.data = AState{q}(;L), nodes(tree))
    sequences_to_tree!(tree, leaf_sequences; alphabet=model.alphabet)

    # re-infer branch length
    if strategy.optimize_branch_length && strategy.optimize_branch_scale
        error(
            """Got `optimize_branch_length` and `optimize_branch_scale`.
            Choose only one of the two. Tree left unchanged."""
        )
    elseif strategy.optimize_branch_length
        opt_strat = @set strategy.joint=false
        optimize_branch_length!(tree, model, opt_strat)
    elseif strategy.optimize_branch_scale
        opt_strat = @set strategy.joint=false
        optimize_branch_scale!(tree, model, opt_strat)
    end

    # reconstruct
    reconstructions = map(1:strategy.repetitions) do _
        # branch length optimization already performed before - do not repeat it
        strat = @set strategy.optimize_branch_length = false
        strat = @set strat.optimize_branch_scale = false

        infer_ancestral!(tree, model, strat)
        iseqs = map(internals(tree)) do node
            label(node) => intvec_to_sequence(
                node.data.sequence; alphabet = model.alphabet
            )
        end |> Dict
        tab = if table_style == :verbose
            generate_verbose_state_table(tree, model.alphabet)
        else
            generate_short_state_table(tree)
        end
        (iseqs, tab)
    end
    internal_sequences = [x[1] for x in reconstructions]
    state_tables = [x[2] for x in reconstructions]

    # write internal sequences to fasta if asked
    if !isnothing(outfasta)
        write_sequences(
            internal_sequences,
            outfasta,
            strategy,
            alignment_per_node,
            node_list,
            alignment_per_node_name
        )
    elseif alignment_per_node || !isnothing(node_list)
        @warn """Asked for individual alignments for specific nodes
        ($(length(node_list)) nodes), but no fasta file provided at `outfasta`.
        Not writing sequences."""
    end

    # Write output tree (potentially with re-inferred branch lengths)
    if !isnothing(outnewick)
        write(outnewick, tree; internal_labels=true)
    end

    # Write ASR state table if asked
    if !isnothing(outtable)
        write_state_table(
            state_tables,
            outtable,
            strategy,
            alignment_per_node,
            node_list,
            alignment_per_node_name,
        )
    end

    return tree, strategy.repetitions == 1 ? internal_sequences[1] : internal_sequences
end

"""
    infer_ancestral(tree::Tree{<:AState}, model, strategy)

Create a copy of `tree` and infer ancestral states at internal nodes.
Leaves of `tree` should already be initialized with observed sequences.
"""
function infer_ancestral(tree::Tree{<:AState}, model, strategy)
    tree_copy = copy(tree)
    res = infer_ancestral!(tree_copy, model, strategy)
    return tree_copy
end

function infer_ancestral!(
    tree::Tree{<:AState},
    model::EvolutionModel,
    strategy::ASRMethod,
)
    # Opt branches
    if strategy.optimize_branch_length && strategy.optimize_branch_scale
        error(
            """Got `optimize_branch_length` and `optimize_branch_scale`.
            Choose one of the two. Tree left unchanged."""
        )
    elseif strategy.optimize_branch_length
        opt_strat = @set strategy.joint=false
        optimize_branch_length!(tree, model, opt_strat)
    elseif strategy.optimize_branch_scale
        opt_strat = @set strategy.joint=false
        optimize_branch_scale!(tree, model, opt_strat)
    end
    # Reconstruction: sets `sequence` and `posterior` at each node
    pruning_alg!(tree, model, strategy)
    return nothing
end

"""
    tree_likelihood!(tree, model, strategy)

Log-likelihood of the leaf sequences. For joint ML strategies, this is the likelihood of
the best joint reconstruction.
"""
function tree_likelihood!(tree::Tree, model::EvolutionModel, strategy::ASRMethod)
    return pruning_alg!(tree, model, strategy; set_state=false)
end


function write_sequences(
    internal_sequences::AbstractArray,
    outfasta,
    strategy,
    alignment_per_node = false,
    node_list = nothing,
    alignment_per_node_name = name -> fasta_from_node_name(name, outfasta),
)
    # alignment for each internal node in a list
    # outfasta should be a string (if not use first element)
    # will write to `outfasta_nodename.fasta` (removing the extension first ofc)
    if alignment_per_node
        isnothing(node_list) && error("Got empty `node_list`.")
        # Write one alignment with `strategy.repetitions` sequences for each `node_list`
        for node in node_list
            fasta_file = alignment_per_node_name(node)
            mkpath(dirname(fasta_file))
            FASTAWriter(open(fasta_file, "w")) do writer
                for rep in 1:strategy.repetitions
                    seq = internal_sequences[rep][node]
                    write(writer, FASTARecord(node * "_rep$rep", seq))
                end
            end
        end
        return nothing
    end

    # Else, write an alignment for the whole tree with one sequence per internal node
    if typeof(outfasta) <: AbstractString
        # Single file provided
        if strategy.repetitions > 1
            @warn """Asked for $(strategy.repetitions) independent inferences \
            but got only one output fasta file $outfasta. Writing the first repetition only.
            """
        end
        mkpath(dirname(outfasta))
        FASTAWriter(open(outfasta, "w")) do writer
            for (name, seq) in internal_sequences[1]
                write(writer, FASTARecord(name, seq))
            end
        end
    elseif typeof(outfasta) <: AbstractVector
        # Vector of files provided
        if strategy.repetitions != length(outfasta)
            error("""Asked for $(strategy.repetitions)
                but only $(length(outfasta)) output fasta files provided.
            """)
        end
        for (file, iseqs) in zip(outfasta, internal_sequences)
            mkpath(dirname(file))
            FASTAWriter(open(file, "w")) do writer
                for (name, seq) in iseqs
                    write(writer, FASTARecord(name, seq))
                end
            end
        end
    end
end

function write_state_table(
    state_tables,
    outtable,
    strategy,
    alignment_per_node,
    node_list,
    alignment_per_node_name,
)
# state_tables is a vector of length `strategy.repetitions`
# each element is the state table generated for a given reconstruction
# each element has one row per tree node
    if alignment_per_node
        isnothing(node_list) && error("Got empty `node_list`.")
        # Write one table per node in `node_list`, with `strategy.repetitions` rows
        header = vcat(state_tables |> first |> eachrow |> first |> collect, "repetition")
        for node in node_list
            # indices of the row containing `node` in all tables of `state_tables`
            idx = map(state_tables) do tab
                findall(r -> r[1] == node, eachrow(tab))
            end
            filter!(!isnothing, idx)
            if length(idx) < length(state_tables)
                error("""Node $node not found in some reconstructions.
                    Cannot write state table""")
            end
            # rows of the table
            # appending the repetition number to each row
            rows = map(i -> hcat(state_tables[i][idx[i], :], i), 1:length(idx))
            values = vcat(rows...)
            # writing output
            table = vcat(reshape(header, 1, length(header)), values)
            table_name = fasta_from_node_name(node, outtable)
            writedlm(table_name, table, '\t')
        end
        return nothing
    end

    if typeof(outtable) <: AbstractString
        # Single file provided
        if strategy.repetitions > 1
            @warn """Asked for $(strategy.repetitions) independent inferences \
            but got only one output table file $outtable. Writing the first repetition only.
            """
        end
        writedlm(outtable, state_tables[1], '\t')
    elseif typeof(outtable) <: AbstractVector
        # Vector of files provided
        if strategy.repetitions != length(outtable)
            error("""Asked for $(strategy.repetitions)
                but only $(length(outtable)) output table files provided.
            """)
        end
        for (file, table) in zip(outtable, state_tables)
            writedlm(file, table, '\t')
        end
    else
        error("Got $outtable for `outtable` argument, expected string or array of strings")
    end

    return nothing
end

function fasta_from_node_name(node_name::AbstractString, base_name::AbstractString)
    bn, ext = splitext(base_name)
    isempty(ext) && (ext = ".fasta")
    return prod([bn, "_", node_name, ext])
end
function fasta_from_node_name(node_name::AbstractString, base_name)
    @warn """Expected a `String` for the name of output fasta. Got $(typeof(base_name)).
        Trying to use the first element."""
    return fasta_from_node_name(node_name, base_name[1])
end


