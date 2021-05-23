using Test
using LapackExtra
using LinearAlgebra
using Random

@testset "LapackExtra" begin
    rng = MersenneTwister(0)
    for T in [Float32, Float64, ComplexF32, ComplexF64]
        a = rand(rng, T, (13, 13))
        a = a + a'
        for uplo in ['U', 'L']
            r1 = LapackExtra.syevd!('N', uplo, copy(a))
            r2 = LinearAlgebra.LAPACK.syev!('N', uplo, copy(a))
            @test isapprox(r1, r2)
            r3 = LapackExtra.syevd!('V', 'U', copy(a))
            r4 = LinearAlgebra.LAPACK.syev!('V', 'U', copy(a))
            @test isapprox(r3[1], r4[1])
            @test isapprox(r3[2], r4[2])
        end
    end
end