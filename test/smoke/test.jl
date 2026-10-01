#=
Check that the main entry points run and write what they should.
=#
using AncestralSequenceReconstruction
using DelimitedFiles
using Random
using Test
using TreeTools

Random.seed!(4)
L, q = 5, 4
model = ASR.JukesCantor(L)
tree = parse_newick_string("((A:0.1,B:0.2)I1:0.3,(C:0.2,D:0.4)I2:0.1)R;")

@testset "Simulation" begin
    sim = ASR.Simulate.evolve(tree, model; alphabet=:nt)
    @test sort(collect(keys(sim.leaf_sequences))) == ["A", "B", "C", "D"]
    @test sort(collect(keys(sim.internal_sequences))) == ["I1", "I2", "R"]
    @test all(s -> length(s) == L, values(sim.leaf_sequences))
end

@testset "Output files" begin
    leaf_sequences = ASR.Simulate.evolve(tree, model; alphabet=:nt).leaf_sequences
    mktempdir() do dir
        strategy = ASRMethod(; ML=false, repetitions=2)
        outfasta = [joinpath(dir, "rec_$i.fasta") for i in 1:2]
        outtable = [joinpath(dir, "rec_$i.tsv") for i in 1:2]
        _, internal_sequences = infer_ancestral(
            tree, leaf_sequences, model, strategy; outfasta, outtable,
        )
        @test length(internal_sequences) == 2
        @test all(isfile, outfasta)
        @test all(isfile, outtable)
        # one header row + one row per internal node
        @test size(readdlm(outtable[1], '\t'), 1) == 4

        # verbose table: one row per (internal node, site), with per-site log-likelihood
        verbose_table = joinpath(dir, "verbose.tsv")
        infer_ancestral(
            tree, leaf_sequences, model, ASRMethod(; ML=true);
            outtable=verbose_table, table_style=:verbose,
        )
        tab = readdlm(verbose_table, '\t')
        @test size(tab, 1) == 3 * L + 1
        # ML reconstruction: log-likelihood of a site is the log of the largest posterior
        for row in eachrow(tab[2:end, :])
            @test exp(row[4]) ≈ maximum(row[5:end]) atol=1e-3
        end
    end
end
