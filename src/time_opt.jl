###########################################################################################
####################################### BRANCH OPT ########################################
###########################################################################################


"""
    optimize_branch_length(
        newick_file::AbstractString,
        fastafile::AbstractString,
        model::EvolutionModel,
        strategy = ASRMethod(; joint=false);
        outnewick=nothing,
    )
"""
function optimize_branch_length(
    newick_file::AbstractString,
    fastafile::AbstractString,
    model::EvolutionModel{q},
    strategy = ASRMethod(; joint=false);
    outnewick=nothing,
    kwargs...
) where q
    # read sequences
    seqmap = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> identifier(rec) => sequence(rec),reader)
    end

    # set parameters and read tree
    L = length(first(seqmap)[2])
    if any(x -> length(x[2]) != L, seqmap)
        error("All sequences must have the same length in $fastafile")
    end
    T() = AState{q}(;L)
    tree = read_tree(newick_file; node_data_type = T)
    sequences_to_tree!(tree, seqmap; alphabet=model.alphabet)

    # re-infer branch lengths
    opt_strat = @set strategy.joint=false
    optimize_branch_length!(tree, model, opt_strat; kwargs...)

    # write output if needed
    if !isnothing(outnewick)
        write(outnewick, tree; internal_labels=true)
    end

    return tree
end

function optimize_branch_length!(tree::Tree, model::AutoRegressiveModel, strat; kwargs...)
    @warn """
        Cannot optimize branches of tree using an autoregressive model.
        Constructing a profile model using the autoregressive, and using it for branch
        length optimization.
        """
    profile_model = ProfileModel(model)
    return optimize_branch_length!(tree, profile_model, strat; kwargs...)
end

function optimize_branch_length!(
    tree::Tree{AState{q}}, model::ProfileModel{q}, strategy = ASRMethod(; joint=false);
    rconv = 1e-3,
) where q
    set_verbose(strategy.verbosity)
    strategy.joint && error("Branch length optimization requires `strategy.joint = false`")
    if model.with_code
        @warn "Branch length optimization ignores the genetic code (`model.with_code`)"
    end
    verbose() > 0 && @info "Optimizing branch length."
    verbose() > 2 && @info "Branch lengths" map(branch_length, tree)

    lk = [tree_likelihood!(tree, model, strategy)]
    verbose() > 1 && @info "Initial log-likelihood $(lk[1])"

    for cycle in 1:strategy.optimize_branch_length_cycles
        t = @elapsed optimize_branch_lengths_cycle!(tree, model, strategy)
        push!(lk, tree_likelihood!(tree, model, strategy))
        lk[end] < lk[end-1] && @warn "Likelihood decreased during optimization: something's wrong"
        rel_delta_lk = (lk[end-1] - lk[end]) / lk[end-1]
        verbose() > 1 && @info "Cycle $cycle: log-likelihood $(lk[end]) - $t seconds"
        verbose() > 1 && @info "Relative lk increase: $(rel_delta_lk)"
        verbose() > 2 && @info "Branch lengths" map(branch_length, tree)
        abs(rel_delta_lk) < rconv && break
    end

    return lk
end
"""
    optimize_branch_length(
        tree::Tree, model::ProfileModel[, strategy::ASRMethod];
        rconv = 1e-2
    )

Optimize branch lengths to maximize likelihood of sequences at leaves of `tree`.
"""
function optimize_branch_length(
    tree, model::EvolutionModel, strategy = ASRMethod(; joint=false); kwargs...
)
    tc = copy(tree)
    lk = optimize_branch_length!(tc, model, strategy; kwargs...)
    return tc, lk
end



# Post-order traversal of the nodes of `tree` (children before ancestors).
# TreeTools < 0.7 provides `POT`, newer versions `postorder_traversal`.
@static if isdefined(TreeTools, :postorder_traversal)
    postorder_nodes(tree) = TreeTools.postorder_traversal(tree)
else
    postorder_nodes(tree) = TreeTools.POT(tree)
end

"""
    optimize_branch_lengths_cycle!(tree, model::ProfileModel, strategy)

Optimize the length of each branch in turn (post-order), keeping the others fixed.
"""
function optimize_branch_lengths_cycle!(tree::Tree, model::ProfileModel, strategy)
    L = length(model)
    a, b = zeros(Float64, L), zeros(Float64, L)
    opt = branch_length_optimizer(model)
    for node in Iterators.filter(!isroot, postorder_nodes(tree))
        # one pass of the message passing algorithm, with current branch lengths
        branch_coefficients!(a, b, tree, node, model, strategy)
        optimize_branch_length!(node, opt, a, b, model.μ)
    end
    return nothing
end

"""
    branch_coefficients!(a, b, tree, node, model::ProfileModel, strategy)

For the profile model, `T = ν I + (1 - ν) 1 π'` with `ν = exp(-μ t)`.
Up to a constant factor, the likelihood of site `i` as a function of the length `t` of the
branch above `node` is then
```
u' T v = ν (u ⋅ v) + (1 - ν) sum(u) (π ⋅ v) = ν a[i] + (1 - ν) b[i]
```
where `u` and `v` are the up and down likelihoods at `node`.
Compute `a` and `b` for all sites, with one pass of the message passing algorithm.
"""
function branch_coefficients!(a, b, tree, node, model::ProfileModel, strategy)
    for pos in ordering(model)
        process_site!(tree, model, strategy, pos; set_state=false)
        W = node.data.weights
        a[pos] = dot(W.u, W.v)
        b[pos] = sum(W.u) * dot(W.π, W.v)
    end
    return nothing
end

"""
    branch_loglk_and_grad(t, a, b, μ)

Log-likelihood `sum_i log(ν a[i] + (1 - ν) b[i])` with `ν = exp(-μ t)`, and its
derivative with respect to `t`. See `branch_coefficients!`.
"""
function branch_loglk_and_grad(t, a, b, μ)
    ν = exp(-μ * t)
    loglk, grad = 0., 0.
    for (ai, bi) in zip(a, b)
        lk = ν * ai + (1 - ν) * bi
        loglk += log(lk)
        grad += -μ * ν * (ai - bi) / lk # dν/dt = -μ ν
    end
    return loglk, grad
end

function branch_length_optimizer(model::ProfileModel)
    opt = Opt(:LD_LBFGS, 1)
    lower_bounds!(opt, BRANCH_LWR_BOUND(length(model); style=:ml))
    upper_bounds!(opt, BRANCH_UPR_BOUND(model; style=:bayes))
    ftol_rel!(opt, 1e-2)
    maxeval!(opt, 100)
    return opt
end

function optimize_branch_length!(node::TreeNode, opt::NLopt.Opt, a, b, μ)
    max_objective!(opt, (t, grad) -> begin
        loglk, g = branch_loglk_and_grad(t[1], a, b, μ)
        if !isempty(grad)
            grad[1] = g
        end
        loglk
    end)
    t0 = clamp(branch_length(node), opt.lower_bounds[1], opt.upper_bounds[1])
    result = optimize(opt, [t0])
    if !in(result[3], [:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED])
        @warn "Branch length opt. above $(label(node)): $result"
    end
    branch_length!(node, result[2][1])
    return result
end

###########################################################################################
##################################### BRANCH SCALING ######################################
###########################################################################################
function optimize_branch_scale(
    newick_file::AbstractString,
    fastafile::AbstractString,
    model::EvolutionModel{q},
    strategy = ASRMethod(; joint=false);
    outnewick=nothing,
) where q
    # read sequences
    seqmap = FASTAReader(open(fastafile, "r")) do reader
        map(rec -> identifier(rec) => sequence(rec),reader)
    end

    # set parameters and read tree
    L = length(first(seqmap)[2])
    if any(x -> length(x[2]) != L, seqmap)
        error("All sequences must have the same length in $fastafile")
    end
    T() = AState{q}(;L)
    tree = read_tree(newick_file; node_data_type = T)
    sequences_to_tree!(tree, seqmap; alphabet=model.alphabet)

    # re-infer branch lengths
    optimize_branch_scale!(tree, model, strategy)

    # write output if needed
    if !isnothing(outnewick)
        write(outnewick, tree; internal_labels=true)
    end

    return tree
end

function optimize_branch_scale!(tree::Tree, model::EvolutionModel, strategy)
    set_verbose(strategy.verbosity)

    # parameters
    params = (tree = tree, model = model, strategy = strategy)

    # optimizer
    lw_bound = 1e-5
    up_bound = 1e5
    epsconv = 1e-5
    maxit = 200

    # opt = Opt(:LN_COBYLA, 1)
    opt = Opt(:LN_SBPLX, 1)
    lower_bounds!(opt, lw_bound)
    upper_bounds!(opt, up_bound)
    ftol_rel!(opt, epsconv)
    maxeval!(opt, maxit)

    max_objective!(opt, (μ, g) -> optim_wrapper_branch_scale!(μ, g, params))
    μ0 = Float64[1]

    verbose() > 0 && @info """
        Finding optimal scaling for branch length using model of type $(typeof(model))
        """
    verbose() > 1 && @info "Initial tree likelihood $(tree_likelihood!(tree, model, params.strategy))"
    result = optimize(opt, μ0)
    if !in(result[3], [:SUCCESS, :STOPVAL_REACHED, :FTOL_REACHED, :XTOL_REACHED])
        @warn "Branch scaling: $result"
    end
    verbose() > 1 && @info "Found scaling $(result[2][1])"
    verbose() > 1 && @info "Final log likelihood $(result[1])"

    scale_branches!(tree, result[2][1])
    return result
end

function optim_wrapper_branch_scale!(μ, grad, params)
    scale_branches!(params.tree, μ[1])
    loglk = tree_likelihood!(params.tree, params.model, params.strategy)
    scale_branches!(params.tree, 1/μ[1])
    # I think I always need this
    if length(grad)>0
    end
    return loglk
end

function scale_branches!(tree::Tree, μ::Number)
    foreach(n -> branch_length!(n, μ*branch_length(n)), nodes(tree; skiproot=true))
end

#========================#
######### Bounds #########
#========================#

# using the idea (1 - e(-t)) ∼ (n+1) / (L+2) (fraction of mutated sites, with a pc.)
# for n = L and n = 0 we get the limits below
BRANCH_LWR_BOUND_BAYES(L) = log(L+2) - log(L+1)
BRANCH_UPR_BOUND_BAYES(L) = log(L+2) * 0.75 # this over-estimates since saturation occurs before -- would need to put long term eq. of model

function BRANCH_UPR_BOUND_BAYES(profile::ProfileModel)
    L = length(profile)
    H_inf = sum(x -> x^2, Iterators.flatten(profile.P))
    return log(H_inf+2)*0.75
end

BRANCH_LWR_BOUND_ML(L) = 0
BRANCH_UPR_BOUND_ML(L) = Inf

function BRANCH_LWR_BOUND(L; style = :ML)
    return if style == :bayes
        BRANCH_LWR_BOUND_BAYES(L)
    elseif style == :ml || style == :ML
        BRANCH_LWR_BOUND_ML(L)
    else
        error("Unknown style $style")
    end
end
function BRANCH_UPR_BOUND(L; style = :bayes)
    return if style == :bayes
        BRANCH_UPR_BOUND_BAYES(L)
    elseif style == :ml || style == :ML
        BRANCH_UPR_BOUND_ML(L)
    else
        error("Unknown style $style")
    end
end
