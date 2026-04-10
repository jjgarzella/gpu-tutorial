#!/usr/bin/env julia
# benchmark/bench_lu.jl
#
# Compares GPU lu_decomp! (GPULUTutorial) against Julia's built-in
# LinearAlgebra.lu! (LAPACK, 1 BLAS thread) across a range of matrix sizes.
#
# Usage:
#   julia --project=benchmark benchmark/bench_lu.jl
#
# Timing is done on data already resident in the appropriate memory
# (CPU array / GPU array), so the comparison reflects compute throughput
# rather than host↔device transfer overhead.

using Pkg
Pkg.activate(joinpath(@__DIR__))

using BenchmarkTools, LinearAlgebra, KernelAbstractions, Printf
using GPULUTutorial

# ─── Backend detection ────────────────────────────────────────────────────────
if !isnothing(Base.find_package("Metal"))
    using Metal
    const BACKEND   = MetalBackend()
    const ArrayType = MtlArray
    const BKNAME    = "Metal"
elseif !isnothing(Base.find_package("CUDA"))
    using CUDA
    CUDA.allowscalar(false)
    const BACKEND   = CUDABackend()
    const ArrayType = CuArray
    const BKNAME    = "CUDA"
elseif !isnothing(Base.find_package("AMDGPU"))
    using AMDGPU
    const BACKEND   = ROCBackend()
    const ArrayType = ROCArray
    const BKNAME    = "AMDGPU"
else
    error("No GPU backend found. Install Metal.jl, CUDA.jl, or AMDGPU.jl.")
end

# ─── Configuration ────────────────────────────────────────────────────────────
BLAS.set_num_threads(1)

BenchmarkTools.DEFAULT_PARAMETERS.seconds  = 5.0   # max time per benchmark
BenchmarkTools.DEFAULT_PARAMETERS.samples  = 200   # max samples
BenchmarkTools.DEFAULT_PARAMETERS.evals    = 1     # one eval per sample (in-place ops)

const SIZES = [64, 128, 256, 512, 1024, 2048]

# ─── Helpers ──────────────────────────────────────────────────────────────────
"""Return an n×n diagonally-dominant Float32 matrix (well-conditioned)."""
function make_matrix(n)
    A = rand(Float32, n, n)
    for i in 1:n
        A[i, i] += Float32(n)
    end
    A
end

# ─── Run benchmark ────────────────────────────────────────────────────────────
println("GPU backend : $BKNAME")
println("BLAS threads: $(BLAS.get_num_threads())")
println()
println("  n    │   CPU min   CPU median │   GPU min   GPU median │  speedup (median)")
println("───────┼──────────────────────────────────────────────────┼──────────────────")

for n in SIZES
    A_cpu = make_matrix(n)
    A_gpu = ArrayType(copy(A_cpu))

    # CPU: lu! modifies in-place; setup= copies the matrix fresh each sample.
    b_cpu = @benchmark lu!(B) setup=(B = copy($A_cpu)) evals=1

    # GPU: same pattern — copy on GPU, decompose, then synchronise so the
    # elapsed time includes all kernel launches.
    b_gpu = @benchmark begin
        lu_decomp!(B)
        KernelAbstractions.synchronize($BACKEND)
    end setup=(B = copy($A_gpu)) evals=1

    t_cpu_min    = minimum(b_cpu).time    / 1e6   # ns → ms
    t_cpu_median = median(b_cpu).time     / 1e6
    t_gpu_min    = minimum(b_gpu).time    / 1e6
    t_gpu_median = median(b_gpu).time     / 1e6
    speedup      = t_cpu_median / t_gpu_median

    @printf("%6d │ %7.3f ms  %7.3f ms │ %7.3f ms  %7.3f ms │     %6.2f×\n",
            n, t_cpu_min, t_cpu_median, t_gpu_min, t_gpu_median, speedup)
end
