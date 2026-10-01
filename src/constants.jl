let v::Int = 0
    global set_verbose(val) = (v=val)
    global verbose() = v
end




#==========================================#
############### Genetic code ###############
#==========================================#


const aa_order = ['K', 'N', 'K', 'N', 'T', 'T', 'T', 'T', 'R', 'S', 'R', 'S', 'I', 'I', 'M', 'I', 'Q', 'H', 'Q', 'H', 'P', 'P', 'P', 'P', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'L', 'E', 'D', 'E', 'D', 'A', 'A', 'A', 'A', 'G', 'G', 'G', 'G', 'V', 'V', 'V', 'V', '*', 'Y', '*', 'Y', 'S', 'S', 'S', 'S', '*', 'C', 'W', 'C', 'L', 'F', 'L', 'F']
const nt_order = ['A', 'C', 'G', 'T']

const gencode_as_dict = let
    D = Dict()
    i = 1
    for a in nt_order, b in nt_order, c in nt_order
        D[prod([a, b, c])] = aa_order[i]
        i += 1
    end
    D
end

function _mut_codons(codon)
    X = mapreduce(vcat, 1:3) do i
        map(nt_order) do a
            c = collect(codon)
            c[i] = a
            prod(c)
        end
    end
    return X
end

const gencode_as_mat = let
    alphabet = ASR.Alphabet(:aa)
    q = length(alphabet)
    R = zeros(Int, q, q)
    codon_list = [prod([a,b,c]) for a in nt_order for b in nt_order for c in nt_order]
    for codon in codon_list
        row = if haskey(alphabet.mapping, gencode_as_dict[codon])
             alphabet.mapping[gencode_as_dict[codon]]
        else
            continue
        end
        for mut_codon in _mut_codons(codon)
            col = if haskey(alphabet.mapping, gencode_as_dict[mut_codon])
                alphabet.mapping[gencode_as_dict[mut_codon]]
            else
                continue
            end
            R[row, col] += 1
        end
    end
    for a in 1:q
        R[a,a] = 0
    end
    R ./ mean(R)
end
