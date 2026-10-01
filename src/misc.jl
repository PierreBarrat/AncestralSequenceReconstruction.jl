#=
Need every node of tree to have `seq` filled in
=#
function reconstruction_likelihood(tree::Tree{<:AState}, model::EvolutionModel)
    llk = sum(nodes(tree; skiproot=true)) do n
        log_transition_probability(
            data(ancestor(n)).sequence, data(n).sequence, branch_length(n), model
        )
    end
    llk += log_probability(data(tree.root).sequence, model)
    return llk
end

function entropy(X::AbstractVector)
    if !isapprox(sum(X), 1, rtol=1e-5)
        error("Probability vector does not sum to one - got $(sum(X))")
    end

    return sum(X) do x
        x == 0 ? 0 : -x*log(x)
    end
end

function generate_short_state_table(node::TreeNode{AState{q}}) where q
    L = node.data.L
    header = vcat(
        ["Node", "Total_LogLikelihood"],
        # map(i -> "lk_$i", 1:L) # uncomment for site likelihood in file
    )
    # likelihood of reconstruction at position i (array)
    site_likelihoods = map(1:L) do i
        site_posterior(node.data, i)[node.data.sequence[i]]
    end

    if any(isapprox(0), site_likelihoods)
        i = findfirst(isapprox(0), site_likelihoods)
        @warn """Found reconstruction with likelihood $(site_likelihoods[i])
        at position $i for node $(label(node))."""
    end
    if any(<(0), site_likelihoods)
        i = findfirst(<(0), site_likelihoods)
        @error """Found reconstruction with negative likelihood $(site_likelihoods[i])
        at position $i for node $(label(node))."""
    end

    R(x) = round(x; sigdigits=3) # rounding for file size
    row = vcat(
        label(node),
        mapreduce(R ∘ log, +, site_likelihoods),
        # map(R ∘ log, site_likelihoods), # uncomment for site lk in file
    )

    return header, row
end

function generate_short_state_table(tree::Tree{AState{q}}; node_list = nothing) where q
    node_list = if isnothing(node_list)
        collect(internals(tree))
    else
        map(x -> tree[x], node_list)
    end
    isempty(node_list) && error("Cannot generate a state table for empty `node_list`")

    n = length(node_list)
    header = generate_short_state_table(first(node_list))[1]
    tab = Matrix{Any}(undef, n+1, length(header))
    tab[1,:] .= header
    foreach(enumerate(node_list)) do (i, node)
        tab[i+1,:] .= generate_short_state_table(node)[2]
    end

    return tab
end

function generate_verbose_state_table(tree::Tree{AState{q}}, alphabet) where q
    alphabet = Alphabet(alphabet)
    header = vcat(
        ["Node", "Site", "State", "LogLikelihood"],
        map(a -> "p_" * alphabet.string[a], 1:q)
    )

    n = length(nodes(tree)) - length(leaves(tree))
    L = first(nodes(tree)).data.L
    tab = Matrix{Any}(undef, n*L+1, length(header))
    tab[1, :] .= header
    R(x) = round(x; sigdigits=3)

    for (counter, (pos, node)) in enumerate(Iterators.product(1:L, internals(tree)))
        c = node.data.sequence[pos]
        posterior = site_posterior(node.data, pos)
        tab[counter+1, 1] = label(node)
        tab[counter+1, 2] = pos
        tab[counter+1, 3] = alphabet.string[c]
        tab[counter+1, 4] = @sprintf("%1.4f", R(log(posterior[c])))
        tab[counter+1, 5:end] .= map(x -> @sprintf("%1.4f", R(x)), posterior)
    end
    tab
end
