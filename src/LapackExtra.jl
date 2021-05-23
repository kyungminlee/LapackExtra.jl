module LapackExtra

import LinearAlgebra
using LinearAlgebra: chkstride1, checksquare
using LinearAlgebra.LAPACK: liblapack, @blasfunc, BlasInt, chklapackerror

function syevd! end

# Symmetric (real) eigensolvers
for (syev, syevr, syevd, sygvd, elty) in
    ((:dsyev_,:dsyevr_,:dsyevd_,:dsygvd_,:Float64),
     (:ssyev_,:ssyevr_,:ssyevd_,:ssygvd_,:Float32))
    @eval begin
        #       SUBROUTINE DSYEV( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, INFO )
        # *     .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LWORK, N
        # *     .. Array Arguments ..
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )


        #       SUBROUTINE DSYEVD( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, IWORK,
        #                          LIWORK, INFO )
        #       .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LIWORK, LWORK, N
        #       ..
        #       .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   A( LDA, * ), W( * ), WORK( * )
        function syevd!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
            W     = similar(A, $elty, n)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($syevd), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                      Ptr{$elty}, Ptr{$elty}, Ref{BlasInt}, Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt}, Clong, Clong),
                      jobz, uplo, n, A, max(1,stride(A,2)), W, work, lwork, iwork, liwork, info, 1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    liwork = BlasInt(iwork[1])
                    resize!(iwork, liwork)
                end
            end
            jobz == 'V' ? (W, A) : W
        end
    end
end



# Hermitian eigensolvers
for (syev, syevd, syevr, sygvd, elty, relty) in
    ((:zheev_,:zheevd_, :zheevr_,:zhegvd_,:ComplexF64,:Float64),
     (:cheev_,:cheevd_, :cheevr_,:chegvd_,:ComplexF32,:Float32))
    @eval begin
        # SUBROUTINE ZHEEVD( JOBZ, UPLO, N, A, LDA, W, WORK, LWORK, RWORK,
        #                    LRWORK, IWORK, LIWORK, INFO )
        #       .. Scalar Arguments ..
        #       CHARACTER          JOBZ, UPLO
        #       INTEGER            INFO, LDA, LIWORK, LRWORK, LWORK, N
        #       ..
        #       .. Array Arguments ..
        #       INTEGER            IWORK( * )
        #       DOUBLE PRECISION   RWORK( * ), W( * )
        #       COMPLEX*16         A( LDA, * ), WORK( * )
        function syevd!(jobz::AbstractChar, uplo::AbstractChar, A::AbstractMatrix{$elty})
            chkstride1(A)
            n = checksquare(A)
            W     = similar(A, $relty, n)
            work  = Vector{$elty}(undef, 1)
            lwork = BlasInt(-1)
            rwork = Vector{$relty}(undef, 1)
            lrwork = BlasInt(-1)
            iwork = Vector{BlasInt}(undef, 1)
            liwork = BlasInt(-1)
            info  = Ref{BlasInt}()
            for i = 1:2  # first call returns lwork as work[1]
                ccall((@blasfunc($syevd), liblapack), Cvoid,
                      (Ref{UInt8}, Ref{UInt8}, Ref{BlasInt}, Ptr{$elty}, Ref{BlasInt},
                      Ptr{$relty}, Ptr{$elty}, Ref{BlasInt}, Ptr{$relty}, Ref{BlasInt},
                      Ptr{BlasInt}, Ref{BlasInt}, Ptr{BlasInt},
                      Clong, Clong),
                      jobz, uplo, n, A, stride(A,2), W, work, lwork, rwork, lrwork, iwork, liwork, info,
                      1, 1)
                chklapackerror(info[])
                if i == 1
                    lwork = BlasInt(real(work[1]))
                    resize!(work, lwork)
                    lrwork = BlasInt(rwork[1])
                    resize!(rwork, lrwork)
                    liwork = BlasInt(iwork[1])
                    resize!(iwork, liwork)
                end
            end
            jobz == 'V' ? (W, A) : W
        end
    end
end




end # module
