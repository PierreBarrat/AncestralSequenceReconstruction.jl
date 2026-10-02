### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ 650e35ce-7bf8-11ef-3388-b19ac2abf7c1
begin
	using Pkg
	Pkg.activate("../../") # assumes this is in examples/PF00014/reconstruction
	using AncestralSequenceReconstruction # core package
	using FASTX # for writing reconstruction to fasta 
	using JLD2 # used to load ArDCA models
	using PlutoUI # for ToC
	using TreeTools # to handle phylogenetic trees
end

# ╔═╡ ab72ea71-9fe3-418f-b2e1-010ee733248c
PlutoUI.TableOfContents()

# ╔═╡ d9fc969f-7390-46c7-95f6-05e7f5e06ef8
md"## Evolutionary model"

# ╔═╡ 5f269ed2-55b9-4902-bd35-48510084e616
md"""
Constructing the autoregressive evolutionary model is done in two steps. 
1. Load the ArDCA model from a `.jld2` file. Alternatively, one can infer the model directly from an alignment using the `ArDCA.jl` package. 
2. Convert the model, which is an `ArDCA.ArNet` object, to an 	`AncestralSequenceReconstruction.AutoRegressiveModel` object. This is essentially a wrapper around the autoregressive model. 
"""

# ╔═╡ 0ed03189-e872-43de-a800-ba6b62550fde
arnet = let 
	jld2_data = JLD2.load("arnet_PF00014_lJ0.01_lH0.001.jld2") # Dict containing the saved variables
	jld2_data["arnet"] # extract the ardca model
end

# ╔═╡ 51bd8795-a8db-4b9e-8eb1-54025f31c3be
ar_model = AutoRegressiveModel(arnet);

# ╔═╡ 42f9abc0-a6c2-40f2-9a0e-92ad9ddeae40
md"""
The autoregressive model has the following fields $(propertynames(ar_model)). 
- `arnet` contains the ArDCA model
- `alphabet` represents a mapping from integers to biological symbols. If the dimension of the arnet is `q=21`, an amino acid alphabet will be picked automatically. See `?ASR.default_alphabet` 	
- `with_code`: whether to use the genetic code when constructing tansition matrices (defaults to `false`).
"""

# ╔═╡ b0737a58-5e84-4853-b04c-795c3b9ff719
md"## Tree and alignment"

# ╔═╡ 5a4dae4b-bb59-41f2-8716-0225076bf338
md"We will directly provide the files to the `infer_ancestral` function"

# ╔═╡ 18924fe6-639e-4578-b0d1-a0c4c93d04fe
# 10 sequences taken from the PF00014 family (trypsin inhibitor)
fasta_file = "PF00014_mgap6_subalignment.fasta"

# ╔═╡ 7eb1d358-b970-41ef-8c07-37f4c5ebc096
# the tree was inferred using iqtree, and then midpoint-rooted
tree_file = "tree_iqtree.nwk"

# ╔═╡ 07652358-cef7-49cf-bac9-0a1c68e247de
md"## Reconstruction options: `ASRMethod`"

# ╔═╡ 304c7f8a-1bd9-4821-96c9-62e2fa87c726
md"""
See `?ASR.ASRMethod` for a docstring. Here, we go for a simple maximum-likelihood AncestralSequenceReconstruction. For the sake of documentation, we will still manually set all the arguments of `ASRMethod`, indicating when we are just using the default. 
"""

# ╔═╡ 0b53e764-0f4e-47f0-a30f-14c65451e4bd
strategy = ASRMethod(;
	joint = false, # (default) - marginal reconstruction. `true` for joint reconstruction, see `?ASR.ASRMethod`
	ML = true, # most likely state. The default is `false`, which samples states instead
	verbosity = 1, # the default is 0. 
	optimize_branch_length = true, # (default: false) - optimize the branch lengths of the tree using the evolutionary model
	optimize_branch_scale = false, # (default) - would rescale all branches by a common factor. Incompatible with the previous, and not available for autoregressive models. 
	repetitions = 1, # (default) - for Bayesian reconstruction, multiple repetitions of the reconstruction process can be done to sample likely ancestors
)

# ╔═╡ f571541d-8643-4312-8934-4cacb64e0505
md"## Performing the reconstruction"

# ╔═╡ 8db5fbbb-302a-4b54-bda2-4f2e32b5ac28
opt_tree, reconstructed_sequences = infer_ancestral(
	tree_file, fasta_file, ar_model, strategy
)	

# ╔═╡ 6996baf3-e4fc-4f13-9bd5-3a54ae10cca1
opt_tree

# ╔═╡ 4380b48b-78d0-4324-b78f-5338da9ed0db
reconstructed_sequences

# ╔═╡ 5c1aeb37-7bfa-4d65-963b-41eedd8b3a09
md"""
## Saving the results of the reconstruction
"""

# ╔═╡ 4ca7f957-6558-42d4-95e7-9bc0f59ca0b1
md"There are two ways to do this. The first is to save the optimized tree and the sequences at internal nodes 'manually'"

# ╔═╡ 458d0d08-ded0-4c62-964a-280476e5f334
begin
	mkpath("ml_reconstruction")
	outfasta = "ml_reconstruction/internals.fasta"
	outnewick = "ml_reconstruction/tree_optimized.nwk"
end

# ╔═╡ 346bb240-2c67-47ad-b467-3b6bc615005b
begin
	write("tree_optimized.nwk", opt_tree)
	# to write sequences, we use the FASTX package
	FASTAWriter(open(outfasta, "w")) do writer
		for (name, seq) in reconstructed_sequences
			write(writer, FASTARecord(name, seq))
		end
	end
end

# ╔═╡ 0fcb6657-89f2-4cb4-81cd-e527d75b07fd
md"The second is to provide `outnewick` and `outfasta` to the `infer_ancestral` call"

# ╔═╡ 44e6366b-97eb-4366-a265-578517342b72
infer_ancestral(
	tree_file, fasta_file, ar_model, strategy; outnewick, outfasta
);

# ╔═╡ 1f268e4e-ac3f-4934-bcf8-e06bcdce20a8
md"## Interpreting the results"

# ╔═╡ 929fcaa3-ce0b-4305-8286-c1a3824435a8
md"""
The `reconstructed_sequences` variable is a dictionary mapping names of internal nodes to sequences. The `.fasta` file that we created above also associated reconstructed sequences to internal node names. Since tree inference softwares usually do not attribute names to internal nodes, they are given default labels using the `TreeTools.jl` package. 
To know what internal nodes correspond to, there are various ways:
- use an input Newick with known internal node names, as these should not be modified during the inference; 
- with the Newick file representing the tree, use whatever tree vizualisation software (*e.g.* `icytree.org`) to display node names and what clade they correspond to; 
- using the TreeTools package from inside julia, as described below.
"""

# ╔═╡ 1a1e388e-4fe5-4567-b6f7-1ad260e5aaf8
md"**Using the `TreeTools` package to explore results**"

# ╔═╡ c29314eb-ea86-408a-82ec-b07e5393002b
leaf_names = map(label, leaves(opt_tree)) # label and leaves are TreeTools functions

# ╔═╡ a3ef6685-0a3b-410f-ac26-b7a3e75acb5b
internal_names = map(label, internals(opt_tree))

# ╔═╡ cf7d8439-dfb8-4874-ba92-b0aec17d969c
md"""
Suppose we want the sequence of the MRCA of the two first leaves in the list above. We first extract the label of the corresponding internal node, and then access the reconstructed sequences.
"""

# ╔═╡ 07847d72-34b7-4a5b-8487-af10418072a0
# lca(tree, label_1, label_2) is another TreeTools function
interesting_internal_node = lca(opt_tree, leaf_names[1], leaf_names[2]) |> label

# ╔═╡ ebb4f14d-2d95-40b1-bd59-54d40e3fdeb9
reconstructed_sequences[interesting_internal_node]

# ╔═╡ 748d1b49-7c18-42d1-af23-4a724dcb720f
md"Now suppose that we are curious as to what \"NODE_3\" represents. We can easily display the label of the leaves in the clade below it"

# ╔═╡ 8419183b-2c98-4002-8334-23c856b2e49a
reconstructed_sequences["NODE_3"]

# ╔═╡ 666620a2-c34e-4df3-bd93-acdeccb783c7
# list of leaves below NODE_3
leaf_labels(node) = isleaf(node) ? [label(node)] : reduce(vcat, map(leaf_labels, children(node)))
leaf_labels(opt_tree["NODE_3"])

# ╔═╡ fe32ade5-6429-470d-a0c9-0cb3a5add1c7
md"## Bayesian reconstruction"

# ╔═╡ ec22c089-7833-4dcd-abbd-78df01007bf8
md"""
We now want to have a Bayesian reconstruction: we will sample internal sequences of internal nodes according to the distribution implied by the leaves and the evolutionary model. 

The tree, alignment and model remain the same. We will change the `ASRMethod` object and some arguments to `infer_ancestral`. 
"""

# ╔═╡ 97326b1e-7013-42de-8aa6-34b9d5bfca59
strategy_bayesian = ASRMethod(;
	joint = false, # (default) - marginal reconstruction. `true` for joint reconstruction, see `?ASR.ASRMethod`
	ML = false, # (default) - sample states from their posterior distribution
	verbosity = 1, # the default is 0. 
	optimize_branch_length = true, # (default: false) - optimize the branch lengths of the tree using the evolutionary model
	optimize_branch_scale = false, # (default) - would rescale all branches by a common factor. Incompatible with the previous, and not available for autoregressive models. 
	repetitions = 10, # (the default is 1) - for Bayesian reconstruction, multiple repetitions of the reconstruction process can be done to sample likely ancestors
)

# ╔═╡ 91d7b5d6-2ad3-40f0-bea1-f1634131559d
md"The first way to do this is to output one alignment of internal sequences for each Bayesian reconstruction. For this, we provide a list of output alignment names in the `outfasta` keyword-argument."

# ╔═╡ e717ff93-ed39-49d2-a305-71272f351612
begin
	mkpath("bayesian_reconstruction")
	outfasta_bayesian = map(1:strategy_bayesian.repetitions) do i
		joinpath("bayesian_reconstruction", "internals_rep$i.fasta")
	end
end

# ╔═╡ f3851ef5-ce50-4670-a1fe-04a9cbcdfeae
_, reconstructed_sequences_bayesian = infer_ancestral(
	tree_file, fasta_file, ar_model, strategy_bayesian;
	outfasta = outfasta_bayesian, 
);

# ╔═╡ db546de4-bb24-430d-9d11-868fe4f71032
reconstructed_sequences_bayesian # this is now an array of dictionaries, one for each reconstruction

# ╔═╡ 47af8f46-958d-4974-b578-8782ece03c1a
md"""
Sometimes, it is more practical to have one alignment for each node of the tree, containing all the reconstructions for this node. To achieve this, we use the `alignment_per_node` keyword in `infer_ancestral`. 
"""

# ╔═╡ 2be814ba-6700-4ab9-a9c3-75f9191ca005
# the alignment for node named `N` will be saved at `outfasta_N.fasta`, where `outfasta` is set below.  
infer_ancestral(
	tree_file, fasta_file, ar_model, strategy_bayesian;
	alignment_per_node=true, outfasta = "bayesian_reconstruction/reconstruction", 
)

# ╔═╡ efcffbef-7285-4156-b85d-aff1ea39924a
md"If only a subset of internal nodes are of interest, their names can be provided using the `node_list` keyword argument, *e.g.* `node_list=[\"NODE_3\", \"NODE_4\"]. "

# ╔═╡ Cell order:
# ╠═650e35ce-7bf8-11ef-3388-b19ac2abf7c1
# ╠═ab72ea71-9fe3-418f-b2e1-010ee733248c
# ╟─d9fc969f-7390-46c7-95f6-05e7f5e06ef8
# ╟─5f269ed2-55b9-4902-bd35-48510084e616
# ╠═0ed03189-e872-43de-a800-ba6b62550fde
# ╠═51bd8795-a8db-4b9e-8eb1-54025f31c3be
# ╟─42f9abc0-a6c2-40f2-9a0e-92ad9ddeae40
# ╟─b0737a58-5e84-4853-b04c-795c3b9ff719
# ╟─5a4dae4b-bb59-41f2-8716-0225076bf338
# ╠═18924fe6-639e-4578-b0d1-a0c4c93d04fe
# ╠═7eb1d358-b970-41ef-8c07-37f4c5ebc096
# ╟─07652358-cef7-49cf-bac9-0a1c68e247de
# ╟─304c7f8a-1bd9-4821-96c9-62e2fa87c726
# ╠═0b53e764-0f4e-47f0-a30f-14c65451e4bd
# ╟─f571541d-8643-4312-8934-4cacb64e0505
# ╠═8db5fbbb-302a-4b54-bda2-4f2e32b5ac28
# ╠═6996baf3-e4fc-4f13-9bd5-3a54ae10cca1
# ╠═4380b48b-78d0-4324-b78f-5338da9ed0db
# ╟─5c1aeb37-7bfa-4d65-963b-41eedd8b3a09
# ╟─4ca7f957-6558-42d4-95e7-9bc0f59ca0b1
# ╠═458d0d08-ded0-4c62-964a-280476e5f334
# ╠═346bb240-2c67-47ad-b467-3b6bc615005b
# ╟─0fcb6657-89f2-4cb4-81cd-e527d75b07fd
# ╠═44e6366b-97eb-4366-a265-578517342b72
# ╟─1f268e4e-ac3f-4934-bcf8-e06bcdce20a8
# ╟─929fcaa3-ce0b-4305-8286-c1a3824435a8
# ╟─1a1e388e-4fe5-4567-b6f7-1ad260e5aaf8
# ╠═c29314eb-ea86-408a-82ec-b07e5393002b
# ╠═a3ef6685-0a3b-410f-ac26-b7a3e75acb5b
# ╟─cf7d8439-dfb8-4874-ba92-b0aec17d969c
# ╠═07847d72-34b7-4a5b-8487-af10418072a0
# ╠═ebb4f14d-2d95-40b1-bd59-54d40e3fdeb9
# ╟─748d1b49-7c18-42d1-af23-4a724dcb720f
# ╠═8419183b-2c98-4002-8334-23c856b2e49a
# ╠═666620a2-c34e-4df3-bd93-acdeccb783c7
# ╟─fe32ade5-6429-470d-a0c9-0cb3a5add1c7
# ╟─ec22c089-7833-4dcd-abbd-78df01007bf8
# ╠═97326b1e-7013-42de-8aa6-34b9d5bfca59
# ╟─91d7b5d6-2ad3-40f0-bea1-f1634131559d
# ╠═e717ff93-ed39-49d2-a305-71272f351612
# ╠═f3851ef5-ce50-4670-a1fe-04a9cbcdfeae
# ╠═db546de4-bb24-430d-9d11-868fe4f71032
# ╟─47af8f46-958d-4974-b578-8782ece03c1a
# ╠═2be814ba-6700-4ab9-a9c3-75f9191ca005
# ╟─efcffbef-7285-4156-b85d-aff1ea39924a
