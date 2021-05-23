using MKL
using LinearAlgebra
# LinearAlgebra.BLAS.lbt_forward("libmklopenblas64_.dll")

using LapackExtra
using Random
using BenchmarkTools

rng = MersenneTwister(0)
for T in [Float64] #, ComplexF64]
    a = rand(rng, T, (4096, 4096))
    a = a + a'
    for uplo in ['U']
        @show T, uplo
        println("SYEVD")
        display(@benchmark LapackExtra.syevd!('N', $uplo, $(copy(a))))
        println()
        println("SYEV")
        display(@benchmark LinearAlgebra.LAPACK.syev!('N', $uplo, $(copy(a))))
        println()
   end
end