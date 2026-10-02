using AncestralSequenceReconstruction
using ArDCA
using JLD2
using StatsBase
using Test
using TreeTools


# run a test file in its own module, so that its global names do not clash with others
function include_isolated(file)
    m = Module()
    Core.eval(m, :(include(f) = Base.include($m, f)))
    return Base.include(m, joinpath(@__DIR__, file))
end

@testset "AncestralSequenceReconstruction.jl" begin
    @testset "basics" begin
        # Basic tests for evolution models
        include("basics/emodels.jl")
    end
    @testset "Felsenstein" begin
        # Example in Felsenstein's "Inferring phylogenies" in section 16.4
        include("Felsenstein/test.jl")
    end
    @testset "Brute force" begin
        # Exact enumeration of internal states on a small tree
        include_isolated("brute_force/test.jl")
    end
    @testset "Golden outputs" begin
        # Reference outputs from the code before the memory refactoring
        include_isolated("golden/test.jl")
    end
    @testset "Smoke tests" begin
        # Simulation and output files
        include_isolated("smoke/test.jl")
    end
    @testset "time_opt" begin
        # Bousseau alg: update neighbours
        include("time_opt/test.jl")
    end
end
