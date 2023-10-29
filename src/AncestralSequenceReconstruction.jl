module AncestralSequenceReconstruction

const ASR = AncestralSequenceReconstruction
export ASR

import Base: copy, length

using Accessors
using DelimitedFiles
using FASTX
using LinearAlgebra
using NLopt
using StatsBase
using TreeTools

include("constants.jl")

include("objects.jl")
export ASRMethod

include("evolutionary_models.jl")
export EvolutionModel
include("profile_model.jl")
export ProfileModel, JukesCantor

include("sequences.jl")

include("reconstruction.jl")
export infer_ancestral

include("time_opt.jl")

end
