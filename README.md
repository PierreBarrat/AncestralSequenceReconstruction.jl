# AncestralSequenceReconstruction

[![Build Status](https://github.com/PierreBarrat/AncestralSequenceReconstruction.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/PierreBarrat/AncestralSequenceReconstruction.jl/actions/workflows/CI.yml?query=branch%3Amaster)

This package implements the algorithm described in this article
```
Reconstruction of ancestral protein sequences using autoregressive generative models
Matteo De Leonardis, Andrea Pagnani, Pierre Barrat-Charlaix
bioRxiv 2024
```

It relies heavily on the ArDCA method for generative models which is described here 
```
Efficient generative modeling of protein sequences using simple autoregressive models
Jeanne Trinquier, Guido Uguzzoni, Andrea Pagnani, Francesco Zamponi, Martin Weigt
Nature Communications 2021
```
and whose code can be found at https://github.com/pagnani/ArDCA.jl
Please cite these works if you use this code. 

# Installation

For now, there is no command line version of this software. 
The only way to use it is through a Julia script, notebook or REPL session. 
I recommend having a look to the example notebook to learn how to use the software. 
Here are the steps to installation. 

1. Install Julia (version 1.11 or later): https://julialang.org/. If you're never used the language, it can be useful to have a look at https://docs.julialang.org/en/v1/manual/getting-started/  
2. Open an REPL session, and install the package by running
  ```
  using Pkg
  Pkg.add(url="https://github.com/PierreBarrat/AncestralSequenceReconstruction.jl")
  ```
  You can now use it from inside the julia session: `using AncestralSequenceReconstruction`  
3. To see the **example notebook**, you need to install [Pluto](https://github.com/fonsp/Pluto.jl): `Pkg.add("Pluto")`. 
  Launch it by running `using Pluto; Pluto.run()`. 
  Then, open the notebook `example/PF00014/reconstruction/reconstruction_tutorial.jl` from inside Pluto.   
4. To use the package in practice, you might need to install other dependencies, including  
  - [ArDCA](https://github.com/pagnani/ArDCA.jl) to infer and manipulate autoregressive protein models  
  - [JLD2](https://github.com/JuliaIO/JLD2.jl) to save (or load) ArDCA models to (from) files  
  Other useful packages are loaded in the example notebook. 

Scripts that were used to generate the results in the article can be found at https://github.com/PierreBarrat/AutoRegressiveASR, and provide various examples of how to use it. 

# Contents

## Reconstructing ancestral sequences

See the example notebook and the docstrings `?infer_ancestral` and `?ASRMethod`. 
Two short examples, to run from the folder `example/PF00014/reconstruction`.

**Most likely ancestors from a tree and an alignment**
```julia
using AncestralSequenceReconstruction, JLD2

arnet = JLD2.load("arnet_PF00014_lJ0.01_lH0.001.jld2")["arnet"] # model inferred with ArDCA
model = AutoRegressiveModel(arnet)

strategy = ASRMethod(; joint=false, ML=true) # most likely state at each node
tree, sequences = infer_ancestral(
    "tree_iqtree.nwk", "PF00014_mgap6_subalignment.fasta", model, strategy;
    outfasta = "internals.fasta", # optional: write internal sequences to a fasta file
)
sequences["NODE_1"] # reconstructed sequence of node `NODE_1`, a `String`
```

**Sampling ancestors and reading posterior distributions**
```julia
using TreeTools # for `tree["NODE_1"]`

# 100 independent samples of the internal sequences, instead of the most likely one
strategy = ASRMethod(; joint=false, ML=false, repetitions=100)
tree, samples = infer_ancestral(
    "tree_iqtree.nwk", "PF00014_mgap6_subalignment.fasta", model, strategy
)
samples[1]["NODE_1"] # `samples` has one `Dict` (node name => sequence) per repetition

# `tree` holds the last repetition. At each node:
node = tree["NODE_1"]
node.data.sequence                      # reconstructed sequence, as integers
ASR.site_posterior(node.data, 10)       # P(state | data) at site 10, a vector of length q
node.data.posterior                     # the same for all sites, a q x L matrix
model.alphabet.string[argmax(ASR.site_posterior(node.data, 10))] # amino acid for the most likely state
```
`strategy` has four modes, see `?ASRMethod`: `joint` and `ML` can each be `true` or `false`. 
For `joint=false`, the posterior is the marginal distribution at each node. 
For `joint=true`, it is the distribution given the state chosen at the ancestor of the node.

Memory use grows linearly with sequence length: for each node and site, only the sequence and the posterior distribution are kept. 

## Evolutionary models

The package can accommodate other evolutionary models than the autoregressive one. 
Main functions for reconstruction take arguments of the type `EvolutionModel`, which currently has two subtypes  

- the `AutoRegressiveModel` described in the example notebook  
- `ProfileModel`, where all sites evolve independently.   

For information about the latter, see `?ProfileModel`. 

## Simulating sequences

It is possible to simulate evolution by using an evolutionary model and a tree. 
See the functions in `src/simulate.jl`. 

# Current issues/limitations

## Branch lengths

`optimize_branch_length=true` is not available directly for the autoregressive model. 
The branches are optimized using a `ProfileModel` built from 1000 sequences sampled from the autoregressive model, 
so the optimized lengths vary slightly between runs (call `Random.seed!` first to reproduce them). 
Branch length optimization never uses the genetic code, even if the model has `with_code=true`. 
Finally, `optimize_branch_scale=true` does not work with the autoregressive model, only with a `ProfileModel`. 

## Genetic code

It is possible to incorporate the effects of the genetic code into the dynamics of the autoregressive model. 
This is done by passing `with_code=true` when constructing the model: `AutoRegressiveModel(arnet; with_code=true)` (see example notebook).  
An important limitation of using this option is that *gaps are not correctly handled*. In ArDCA as in most energy-based protein generative models, gaps are treated as an extra amino acid. 
However, it is not trivial how one should deal with them in a way that is consistent with the genetic code. 
While this is not a fundamental limitation and will be fixed at some point, please know that you should only use this option if your input sequences have *no gaps*. 

## Mapping from amino-acids to integers

Both in this package and in ArDCA, amino-acids are mapped to integers for easier use. 
By default, this package uses the mapping implied by the string `"-ACDEFGHIKLMNPQRSTVWY"`, *i.e.* `'-' => 1`, `'A' => 2`, etc...
However, the ArDCA package uses a different one: `ArNet` models will by default use `"ACDEFGHIKLMNPQRSTVWY-"`. 
This leads to the following issue: if *(i)* an `ArNet` model is inferred using the default settings of ArDCA and *(ii)* is used for ancestral reconstruction in this package also using default settings, there will be an inconsistency between the mappings and the results of the reconstruction will likely be nonsense. 

There are two ways to work around this.   

1. Infer the autoregressive model using the default mapping of AncestralSequenceReconstruction.jl. This would be done by converting the training alignment used in ArDCA to an integer matrix within Julia, using the mapping `"-ACDEFGHIKLMNPQRSTVWY"`, and then use this matrix to infer the model (see docstring of the `ArDCA.ardca` function).  
  This is the method I currently use, and I have some tools that facilitate it, unfortunately not well documented yet.   

2. Do not use the default mapping when building the evolutionary model from the `ArNet`. In the example notebook, the line `ar_model = AutoRegressiveModel(arnet)` should then be modified to `ar_model = AutoRegressiveModel(arnet; alphabet=:ardca_aa)`. 

In any case, it is important to keep track of the mapping with which the autoregressive model was initially inferred. 
Currently, there is no "natural" way to do this (apart from *e.g.* file naming). 
This is obviously not ideal, and I want to make this easier in the future. 



  
