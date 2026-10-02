#=
Message passing on the tree, one site at a time.

For each site `pos`:
1. `prepare_site!`: compute `π` and `T` for every node, set the observed states at leaves.
2. `messages_from_leaves!`: compute down-likelihoods `v` (and messages `lm_up`), from the
   leaves to the root. This gives the likelihood of the site.
3. Depending on the strategy:
   - marginal (ML or sampling): `messages_from_root!` computes up-likelihoods `u`, from
     the root to the leaves. The posterior at each node then follows from `u`, `T` and `v`.
   - joint sampling: sample the root, then sample each node given its ancestor's state.
   - joint ML (Pupko et al.): messages use max instead of sum, and states are found by
     going back from the root to the leaves (`best_state`).

See `BranchWeights` for the meaning of `π`, `T`, `u`, `v`.
=#

#######################################################################################
####################################### Main alg ######################################
#######################################################################################

"""
    pruning_alg!(tree, model::EvolutionModel, strategy::ASRMethod; set_state = true)

Run the message passing algorithm on `tree`, one site at a time, in the order given by
`ordering(model)`. Return the log-likelihood.
If `set_state`, also reconstruct the sequence and posterior at each node,
stored in `node.data.sequence` and `node.data.posterior`.
"""
function pruning_alg!(
    tree::Tree{AState{q}}, model::EvolutionModel, strategy::ASRMethod;
    set_state = true,
) where q
    if isa(model, AutoRegressiveModel) && !set_state
        error("Inconsistent `model::AutoRegressiveModel` and `set_state=false`")
    end
    return sum(ordering(model)) do pos
        process_site!(tree, model, strategy, pos; set_state)
    end
end

"""
    process_site!(tree, model, strategy, pos; set_state=true)

Compute messages at site `pos` and, if `set_state`, reconstruct the state and posterior of
each node at this site. Return the log-likelihood of the site.
"""
function process_site!(tree::Tree, model, strategy::ASRMethod, pos; set_state=true)
    prepare_site!(tree, model, pos)

    use_max = strategy.joint && strategy.ML
    loglk = messages_from_leaves!(tree.root, use_max)

    if strategy.joint && strategy.ML
        set_state && reconstruct_joint_ML!(tree.root, pos)
    elseif strategy.joint
        set_state && sample_joint!(tree.root, pos)
    else
        messages_from_root!(tree.root)
        set_state && reconstruct_marginal!(tree.root, pos, strategy.ML)
    end

    return loglk
end

function prepare_site!(tree::Tree, model::EvolutionModel, pos)
    for node in nodes(tree)
        reset_weights!(node.data.weights)
        # also sets equilibrium frequencies π
        set_transition_matrix!(node.data, model, branch_length(node), pos)
        isleaf(node) && set_leaf_state!(node.data, pos)
    end
    return nothing
end

function set_leaf_state!(leaf::AState, pos)
    a = leaf.sequence[pos]
    if isnothing(a)
        error("""Tried to initialize leaf state at position $(pos), got `nothing`.
            Are sequences attached to the leaves of the tree?""")
    end
    leaf.weights.v .= 0
    leaf.weights.v[a] = 1
    return nothing
end

#######################################################################################
####################################### Messages ######################################
#######################################################################################

"""
    messages_from_leaves!(node, use_max::Bool)

Compute the down-likelihood `v` of `node` and of all nodes below it, and the messages
`lm_up` that they send to their ancestors.
If `node` is the root, return the log-likelihood of the data.

`use_max` is for joint ML reconstruction: messages are maximized over the states of
children instead of summed.
"""
function messages_from_leaves!(node::TreeNode{<:AState}, use_max::Bool)
    W = node.data.weights
    if !isleaf(node) # leaves: `v` is already set by `set_leaf_state!`
        # log v = sum of log-messages from children
        W.v .= 0
        for c in children(node)
            messages_from_leaves!(c, use_max)
            W.v .+= c.data.weights.lm_up
        end
        W.Fv = exp_normalize!(W.v)
    end

    return if isroot(node)
        lk = use_max ? maximum(W.π .* W.v) : sum(W.π .* W.v)
        log(lk) + W.Fv
    else
        message_to_ancestor!(W, use_max)
        nothing
    end
end

"""
    message_to_ancestor!(W::BranchWeights, use_max)

Set `W.lm_up[a] = log(sum_b T[a,b] v[b]) + Fv`: log-probability of the data below the node,
given that its ancestor is in state `a`.
With `use_max`, the sum is replaced by a max, and the best `b` is stored in `W.best_state[a]`.
"""
function message_to_ancestor!(W::BranchWeights{q}, use_max::Bool) where q
    if use_max
        for a in 1:q
            W.lm_up[a], W.best_state[a] = findmax(b -> W.T[a, b] * W.v[b], 1:q)
        end
    else
        mul!(W.lm_up, W.T, W.v)
    end
    W.lm_up .= log.(W.lm_up) .+ W.Fv
    return nothing
end

"""
    messages_from_root!(node)

Compute the up-likelihood `u` of all nodes below `node`. Requires `messages_from_leaves!`.

For a child `c` of `node`, `u_c` is the probability of the data not below `c`, as a function
of the state `a` of `node`. It is the product of
- the message coming from above `node`: `π[a]` if `node` is the root, otherwise
  `sum_r u_node[r] T_node[r, a]`;
- the messages `lm_up` from the other children of `node` (the sisters of `c`).
"""
function messages_from_root!(node::TreeNode{<:AState})
    W = node.data.weights
    for c in children(node)
        Wc = c.data.weights
        # message from above `node`
        if isroot(node)
            Wc.u .= log.(W.π)
        else
            mul!(Wc.u, W.T', W.u)
            Wc.u .= log.(Wc.u) .+ W.Fu
        end
        # messages from sisters
        for sister in children(node)
            sister != c && (Wc.u .+= sister.data.weights.lm_up)
        end
        Wc.Fu = exp_normalize!(Wc.u)

        messages_from_root!(c)
    end
    return nothing
end

"""
    exp_normalize!(x)

Replace log-weights `x` by normalized weights `exp.(x) / Z` in place, and return `log(Z)`.
"""
function exp_normalize!(x)
    xmax = maximum(x)
    x .= exp.(x .- xmax)
    Z = sum(x)
    x ./= Z
    return log(Z) + xmax
end

#######################################################################################
################################### Reconstruction ####################################
#######################################################################################

#=
For each strategy, a function going from the root to the leaves and setting
`sequence[pos]` and `posterior[:, pos]` at each node.
=#

"""
    reconstruct_marginal!(node, pos, ML::Bool)

Marginal reconstruction: the posterior of each node is computed independently of the
states chosen at other nodes. Pick the most likely state if `ML`, otherwise sample.
"""
function reconstruct_marginal!(node::TreeNode{<:AState}, pos, ML::Bool)
    W = node.data.weights
    p = site_posterior(node.data, pos)
    if isroot(node)
        p .= W.π .* W.v
    else
        mul!(p, W.T', W.u) # sum_a u[a] T[a, b]
        p .*= W.v
    end
    p ./= sum(p)
    node.data.sequence[pos] = ML ? argmax(p) : wsample(p)

    for c in children(node)
        reconstruct_marginal!(c, pos, ML)
    end
    return nothing
end

"""
    sample_joint!(node, pos[, ancestor_state])

Sample the state of `node` given the state of its ancestor and the data below it:
`P(b | ancestor_state) ∝ T[ancestor_state, b] * v[b]` (`π[b] * v[b]` at the root).
The posterior stored is this conditional distribution.
"""
function sample_joint!(node::TreeNode{<:AState}, pos, ancestor_state=nothing)
    p = conditional_distribution!(node, pos, ancestor_state)
    node.data.sequence[pos] = wsample(p)
    for c in children(node)
        sample_joint!(c, pos, node.data.sequence[pos])
    end
    return nothing
end

"""
    reconstruct_joint_ML!(node, pos[, ancestor_state])

Joint maximum likelihood reconstruction (Pupko et al., 2000).
Requires `messages_from_leaves!(node, true)`, which stored in `best_state` the best state
of each node given the state of its ancestor.
The posterior stored is the conditional distribution given the ancestor's state, computed
with max-messages: it is only indicative.
"""
function reconstruct_joint_ML!(node::TreeNode{<:AState}, pos, ancestor_state=nothing)
    p = conditional_distribution!(node, pos, ancestor_state)
    node.data.sequence[pos] = if isroot(node)
        argmax(p)
    elseif isleaf(node)
        node.data.sequence[pos] # observed
    else
        node.data.weights.best_state[ancestor_state]
    end
    for c in children(node)
        reconstruct_joint_ML!(c, pos, node.data.sequence[pos])
    end
    return nothing
end

# `posterior[:, pos] ∝ T[ancestor_state, :] .* v`, or `π .* v` at the root
function conditional_distribution!(node::TreeNode{<:AState}, pos, ancestor_state)
    W = node.data.weights
    p = site_posterior(node.data, pos)
    if isroot(node)
        p .= W.π .* W.v
    else
        p .= view(W.T, ancestor_state, :) .* W.v
    end
    p ./= sum(p)
    return p
end
