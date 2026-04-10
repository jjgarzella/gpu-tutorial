using Test
using GPULUTutorial
using KernelAbstractions
using LinearAlgebra

# =============================================================================
# Backend detection
#
# GPU backends are loaded only when the corresponding package is installed
# AND the hardware is functional.
# Students: add whichever GPU package matches your hardware to this test
# environment, e.g.:
#   julia --project=test -e 'using Pkg; Pkg.add("CUDA")'
# =============================================================================

backends = Tuple{Any, Type, String}[]

if !isnothing(Base.find_package("CUDA"))
    using CUDA
    if CUDA.functional()
        CUDA.allowscalar(false)
        push!(backends, (CUDABackend(), CuArray, "CUDA"))
    end
end

if !isnothing(Base.find_package("Metal"))
    using Metal
    if Metal.functional()
        push!(backends, (MetalBackend(), MtlArray, "Metal"))
    end
end

if !isnothing(Base.find_package("oneAPI"))
    using oneAPI
    if oneAPI.functional()
        push!(backends, (oneAPIBackend(), oneArray, "oneAPI"))
    end
end

if !isnothing(Base.find_package("AMDGPU"))
    using AMDGPU
    if AMDGPU.functional()
        push!(backends, (ROCBackend(), ROCArray, "AMDGPU"))
    end
end

# =============================================================================
# Test helpers
# =============================================================================

"""
    make_test_matrix(T, n) -> Matrix{T}

Return an n×n diagonally dominant matrix of element type T suitable for
LU decomposition testing.

For floating-point types, entries are random with the diagonal inflated by n
to ensure non-singular, well-conditioned matrices with no need for pivoting
to a zero diagonal.

For integer types, the diagonal is n+1 and off-diagonal entries are 0 or 1,
keeping values small to avoid overflow during integer arithmetic.
"""
function make_test_matrix(::Type{T}, n) where {T<:AbstractFloat}
    A = rand(T, n, n)
    for i in 1:n
        A[i, i] += T(n)  # diagonal dominance
    end
    return A
end

function make_test_matrix(::Type{T}, n) where {T<:Integer}
    A = zeros(T, n, n)
    for i in 1:n
        A[i, i] = T(n + 1)
        for j in 1:n
            i != j && (A[i, j] = T(rand(0:1)))
        end
    end
    return A
end

# =============================================================================
# Per-backend test runner
# =============================================================================

function run_lu_tests(backend, ArrayType, name)
    @testset "$name" begin

        # ------------------------------------------------------------------
        # Float32 — correctness tests
        #
        # After lu_decomp!, reconstruct L and U from the packed result and
        # verify that L * U ≈ P * A_original (up to floating-point tolerance).
        # ------------------------------------------------------------------
        @testset "Float32" begin
            for n in [4, 16, 64, 256, 1000]
                @testset "n=$n" begin
                    A_cpu = make_test_matrix(Float32, n)
                    A_gpu = ArrayType(copy(A_cpu))

                    A_result, perm = lu_decomp!(A_gpu)
                    KernelAbstractions.synchronize(backend)

                    R = Array(A_result)  # packed L\U result on CPU

                    # Unpack: L has unit diagonal (not stored), U is upper.
                    L = LowerTriangular(R) - Diagonal(diag(R)) + I
                    U = UpperTriangular(R)

                    # Build permutation matrix from perm vector.
                    P = Matrix{Float32}(I, n, n)[perm, :]

                    # atol=2f-2: unblocked GPU rank-1 updates accumulate ~0.012
                    # Frobenius norm error in Float32 (vs LAPACK's 0.002 via
                    # blocked BLAS3). Both are correct; LAPACK is just more
                    # accurate due to SGEMM for the Schur complement.
                    @test isapprox(L * U, P * A_cpu; atol=2f-2)
                end
            end
        end

        # ------------------------------------------------------------------
        # Int32 — callability tests only
        #
        # LU decomposition with integer element types involves integer division
        # (which truncates in Julia), so the numerical result is generally not
        # meaningful. These tests only verify that the functions accept Int32
        # arrays and complete without error.
        # ------------------------------------------------------------------
        @testset "Int32 (callability)" begin
            for n in [4, 16]
                @testset "n=$n" begin
                    A_cpu = make_test_matrix(Int32, n)
                    A_gpu = ArrayType(copy(A_cpu))
                    @test_nowarn lu_decomp!(A_gpu)
                end
            end
        end

        # ------------------------------------------------------------------
        # findpivot! correctness
        #
        # Verify that findpivot! returns the correct pivot row for known
        # inputs, including multi-workgroup cases (n > DEFAULT_GROUPSIZE).
        # ------------------------------------------------------------------
        @testset "findpivot! correctness" begin
            # Small: pivot in first (only) workgroup
            @testset "n=8, pivot at row 5" begin
                A_cpu = zeros(Float32, 8, 8)
                for i in 1:8; A_cpu[i, 1] = Float32(i); end
                A_cpu[5, 1] = 100f0  # clear winner at row 5
                A_gpu = ArrayType(copy(A_cpu))
                @test findpivot!(A_gpu, 1) == 5
            end

            # Large: pivot in a later workgroup (row 300 > groupsize 256)
            @testset "n=512, pivot at row 300" begin
                A_cpu = zeros(Float32, 512, 512)
                for i in 1:512; A_cpu[i, 1] = Float32(i); end
                A_cpu[300, 1] = 1000f0  # winner lives beyond workgroup 1
                A_gpu = ArrayType(copy(A_cpu))
                @test findpivot!(A_gpu, 1) == 300
            end

            # k > 1: search starts at row k, pivot at row k+257 (second group)
            @testset "n=512, k=10, pivot at row 267" begin
                k = 10
                A_cpu = zeros(Float32, 512, 512)
                for i in k:512; A_cpu[i, k] = Float32(i - k + 1); end
                A_cpu[267, k] = 5000f0  # row 267 = k + 257, in second workgroup
                A_gpu = ArrayType(copy(A_cpu))
                @test findpivot!(A_gpu, k) == 267
            end
        end

        # ------------------------------------------------------------------
        # updatesubmatrix! performance regression
        #
        # The COLUMN-strategy kernel caches the pivot row in shared memory
        # so row-threads within a workgroup share one load instead of each
        # hitting global memory. Assert it is meaningfully faster than the
        # naive version on a large-enough trailing submatrix.
        # ------------------------------------------------------------------
        @testset "updatesubmatrix! COLUMN faster than naive" begin
            n_perf = 512
            k_perf = 1  # benchmark on the full n-1 × n-1 submatrix
            A_perf = ArrayType(make_test_matrix(Float32, n_perf))

            # Warmup (avoid JIT / first-call overheads)
            for _ in 1:5
                GPULUTutorial.updatesubmatrix_naive!(A_perf, k_perf)
                KernelAbstractions.synchronize(backend)
                updatesubmatrix!(A_perf, k_perf)
                KernelAbstractions.synchronize(backend)
            end

            # Time naive
            t_naive = @elapsed begin
                for _ in 1:20
                    GPULUTutorial.updatesubmatrix_naive!(A_perf, k_perf)
                    KernelAbstractions.synchronize(backend)
                end
            end

            # Time COLUMN strategy
            t_col = @elapsed begin
                for _ in 1:20
                    updatesubmatrix!(A_perf, k_perf)
                    KernelAbstractions.synchronize(backend)
                end
            end

            @info "updatesubmatrix timing: naive=$(round(t_naive*1000, digits=1))ms  COLUMN=$(round(t_col*1000, digits=1))ms"
            # On Metal (Apple Silicon unified memory) the GPU L2 cache already
            # holds the small pivot row, so threadgroup memory adds no bandwidth
            # benefit. Assert speedup only on discrete-GPU backends (CUDA, AMD).
            if name != "Metal"
                @test t_col < t_naive * 0.9
            else
                @test_skip t_col < t_naive * 0.9  # unified-memory cache defeats optimization
            end
        end

        # ------------------------------------------------------------------
        # Individual stub callability
        #
        # Verify that each exported function can be called without error on
        # a small matrix. These tests will pass even before students implement
        # the kernel bodies.
        # ------------------------------------------------------------------
        @testset "stub callability" begin
            n = 8
            A_cpu = make_test_matrix(Float32, n)
            A_gpu = ArrayType(copy(A_cpu))

            @test_nowarn findpivot!(A_gpu, 1)
            @test_nowarn swaprows!(A_gpu, 1, 2)
            # Qualify to disambiguate from LinearAlgebra.normalize!
            @test_nowarn GPULUTutorial.normalize!(A_gpu, 1)
            @test_nowarn updatesubmatrix!(A_gpu, 1)
        end

    end
end

# =============================================================================
# Run all tests
# =============================================================================

@testset "GPULUTutorial" begin
    @info "Detected backends: $(join(getindex.(backends, 3), ", "))"
    for (backend, ArrayType, name) in backends
        run_lu_tests(backend, ArrayType, name)
    end
end
