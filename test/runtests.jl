using Test
using GPULUTutorial
using KernelAbstractions
using LinearAlgebra

# =============================================================================
# Backend detection
#
# We always include the CPU backend. GPU backends are loaded only when
# the corresponding package is installed AND the hardware is functional.
# Students: add whichever GPU package matches your hardware to this test
# environment, e.g.:
#   julia --project=test -e 'using Pkg; Pkg.add("CUDA")'
# =============================================================================

backends = Tuple{Any, Type, String}[]

# CPU is always available
push!(backends, (CPU(), Array, "CPU"))

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

                    @test isapprox(L * U, P * A_cpu; atol=1f-2)
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
