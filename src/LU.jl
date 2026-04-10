using KernelAbstractions

# Type-stable division: keeps Float types in their own precision,
# and keeps Integer types in Int (truncating), avoiding Float64 promotion
# that Metal's GPU compiler rejects.
@inline _div(a::T, b::T) where {T<:AbstractFloat} = a / b
@inline _div(a::T, b::T) where {T<:Integer} = div(a, b)

# =============================================================================
# Kernels
#
# Each @kernel below corresponds to one step of LU decomposition with partial
# pivoting. The bodies are stubs — students fill in the implementation.
#
# Reference: Kurzak & Dongarra, "Implementing LU Factorization on GPUs",
# IEEE IPDPS 2010.
# =============================================================================

"""
    findpivot_kernel!(A, k, nrows, pivot_val, pivot_idx)

Workgroup tree reduction to find the row with the maximum absolute value in
column `k` of `A`, searching rows `k:k+nrows-1`.

Each workgroup reduces its chunk of the column into a single `(max_val,
row_index)` pair stored in shared memory, then writes that pair to
`pivot_val[gi]` and `pivot_idx[gi]`. The host picks the global argmax from
those `ngroups` values.

Launch with `ndrange = ngroups * groupsize` (full workgroups) so that
out-of-range threads can write a neutral value `0` rather than leaving
shared-memory slots uninitialized.
"""
@kernel function findpivot_kernel!(A, @Const(k), @Const(nrows), pivot_val, pivot_idx)
    gi  = @index(Group, Linear)
    li  = @index(Local, Linear)
    i   = (gi - 1) * DEFAULT_GROUPSIZE + li

    # Use the compile-time constant for @localmem size (Metal requires static dims).
    svals = @localmem eltype(A) (DEFAULT_GROUPSIZE,)
    sidxs = @localmem Int32 (DEFAULT_GROUPSIZE,)

    if i <= nrows
        row = k + i - 1
        svals[li] = abs(A[row, k])
        sidxs[li] = Int32(row)
    else
        svals[li] = eltype(A)(0)
        sidxs[li] = Int32(0)
    end
    @synchronize()

    # Explicitly unrolled tree reduction — avoids @synchronize inside a loop.
    # Metal's driver miscompiles barrier-inside-loop, treating it as a no-op;
    # each stride block must be a separate statement with its own @synchronize.
    # Assumes DEFAULT_GROUPSIZE == 256.
    if li <= 128; if svals[li+128] > svals[li]; svals[li] = svals[li+128]; sidxs[li] = sidxs[li+128]; end; end; @synchronize()
    if li <=  64; if svals[li+ 64] > svals[li]; svals[li] = svals[li+ 64]; sidxs[li] = sidxs[li+ 64]; end; end; @synchronize()
    if li <=  32; if svals[li+ 32] > svals[li]; svals[li] = svals[li+ 32]; sidxs[li] = sidxs[li+ 32]; end; end; @synchronize()
    if li <=  16; if svals[li+ 16] > svals[li]; svals[li] = svals[li+ 16]; sidxs[li] = sidxs[li+ 16]; end; end; @synchronize()
    if li <=   8; if svals[li+  8] > svals[li]; svals[li] = svals[li+  8]; sidxs[li] = sidxs[li+  8]; end; end; @synchronize()
    if li <=   4; if svals[li+  4] > svals[li]; svals[li] = svals[li+  4]; sidxs[li] = sidxs[li+  4]; end; end; @synchronize()
    if li <=   2; if svals[li+  2] > svals[li]; svals[li] = svals[li+  2]; sidxs[li] = sidxs[li+  2]; end; end; @synchronize()
    if li <=   1; if svals[li+  1] > svals[li]; svals[li] = svals[li+  1]; sidxs[li] = sidxs[li+  1]; end; end; @synchronize()

    if li == 1
        pivot_val[gi] = svals[1]
        pivot_idx[gi] = sidxs[1]
    end
end

"""
    swaprows_kernel!(A, k, pivot_row)

Swap rows `k` and `pivot_row` of matrix `A`.

Each thread is responsible for one column index `j` (1-based).

TODO: Implement this kernel.
  - Read `A[k, j]` and `A[pivot_row, j]`.
  - Write them back in swapped order.
"""
@kernel function swaprows_kernel!(A, @Const(k), @Const(pivot_row))
    j   = (@index(Group, Linear) - 1) * @groupsize()[1] + @index(Local, Linear)
    tmp = A[k, j]
    A[k, j] = A[pivot_row, j]
    A[pivot_row, j] = tmp
end

"""
    normalize_kernel!(A, k)

Divide column `k` of `A`, rows `k+1` through `n`, by the pivot element
`A[k, k]`. This stores the multipliers in the lower triangular part.

Each thread is responsible for one row offset `i` (1-based), corresponding
to actual row `k + i`.

TODO: Implement this kernel.
  - Compute `A[row, k] = A[row, k] / A[k, k]`.
"""
@kernel function normalize_kernel!(A, @Const(k))
    i   = (@index(Group, Linear) - 1) * @groupsize()[1] + @index(Local, Linear)
    row = k + i
    A[row, k] = _div(A[row, k], A[k, k])
end

"""
    updatesubmatrix_naive_kernel!(A, k)

Naive rank-1 update of the trailing submatrix:
    A[k+1:n, k+1:n] -= A[k+1:n, k] * A[k, k+1:n]

Each thread is responsible for one `(i, j)` pair (1-based offsets),
corresponding to actual indices `(k+i, k+j)`. Each thread reads
`A[k, col]` directly from global memory.
"""
@kernel function updatesubmatrix_naive_kernel!(A, @Const(k))
    gs   = @groupsize()
    gi, gj = @index(Group, NTuple)
    li, lj = @index(Local, NTuple)
    i    = (gi - 1) * gs[1] + li
    j    = (gj - 1) * gs[2] + lj
    row  = k + i
    col  = k + j
    A[row, col] = muladd(-A[row, k], A[k, col], A[row, col])
end

"""
    updatesubmatrix_kernel!(A, k)

COLUMN-strategy rank-1 update of the trailing submatrix:
    A[k+1:n, k+1:n] -= A[k+1:n, k] * A[k, k+1:n]

Each workgroup covers a (ROW_BLOCK × COL_BLOCK) tile. All ROW_BLOCK threads
in the same workgroup column share the same pivot-row element `A[k, col]`.
The COLUMN strategy caches that element in shared memory so only one global
load is needed per workgroup column instead of ROW_BLOCK loads.

Uses `@localmem` sized by `GROUPSIZE_2D[2]` (a compile-time constant) to
avoid Metal's restriction on dynamic threadgroup memory sizes.
"""
@kernel function updatesubmatrix_kernel!(A, @Const(k))
    gs   = @groupsize()
    gi, gj = @index(Group, NTuple)
    li, lj = @index(Local, NTuple)
    i    = (gi - 1) * gs[1] + li
    j    = (gj - 1) * gs[2] + lj
    row  = k + i
    col  = k + j

    # COLUMN strategy: each thread in the j-direction loads one pivot-row
    # element into shared memory; all row-threads then reuse the cached value.
    u_tile = @localmem eltype(A) (GROUPSIZE_2D[2],)
    u_tile[lj] = A[k, col]
    @synchronize()

    A[row, col] = muladd(-A[row, k], u_tile[lj], A[row, col])
end

# =============================================================================
# Array-level wrappers
#
# Each function below launches the corresponding kernel on whatever backend
# backs the array `A` (CPU, CUDA, Metal, oneAPI, or AMDGPU).
#
# Students: once your kernel body is correct, the wrappers should need no
# changes — only the @kernel stubs above.
# =============================================================================

const DEFAULT_GROUPSIZE = 256
# 2-D groupsize for the rank-1 update kernels: (ROW_BLOCK, COL_BLOCK).
# With COL_BLOCK > 1, all row-threads for a given column share the same
# pivot-row element, enabling the COLUMN-strategy shared-memory optimisation.
const GROUPSIZE_2D = (16, 16)

"""
    findpivot!(A, k) -> Int

Find the row index (1-based) of the maximum absolute value in column `k` of
`A`, searching rows `k:n`. Launches `findpivot_kernel!` on the backend
inferred from `A`.

Uses a workgroup-level tree reduction so only `ceil(nrows/groupsize)` values
are transferred to the host instead of all `nrows` values.

Returns the pivot row index as a CPU `Int`.
"""
function findpivot!(A::AbstractMatrix, k::Int)
    n = size(A, 1)
    backend = get_backend(A)

    nrows = n - k + 1
    ngroups = cld(nrows, DEFAULT_GROUPSIZE)
    pivot_val = KernelAbstractions.zeros(backend, eltype(A), ngroups)
    pivot_idx = KernelAbstractions.zeros(backend, Int32, ngroups)

    kernel! = findpivot_kernel!(backend, DEFAULT_GROUPSIZE)
    # Full workgroups so out-of-range threads write neutral values.
    kernel!(A, k, nrows, pivot_val, pivot_idx; ndrange=ngroups * DEFAULT_GROUPSIZE)

    vals_cpu = Array(pivot_val)
    idxs_cpu = Array(pivot_idx)
    return Int(idxs_cpu[argmax(vals_cpu)])
end

"""
    swaprows!(A, k, pivot_row)

Swap rows `k` and `pivot_row` of matrix `A` in-place. Launches
`swaprows_kernel!` on the backend inferred from `A`.
"""
function swaprows!(A::AbstractMatrix, k::Int, pivot_row::Int)
    n = size(A, 2)
    backend = get_backend(A)
    kernel! = swaprows_kernel!(backend, DEFAULT_GROUPSIZE)
    kernel!(A, k, pivot_row; ndrange=n)
end

"""
    normalize!(A, k)

Divide column `k` of `A`, rows `k+1:n`, by `A[k,k]` in-place. Launches
`normalize_kernel!` on the backend inferred from `A`.
"""
function normalize!(A::AbstractMatrix, k::Int)
    n = size(A, 1)
    backend = get_backend(A)
    nrows = n - k
    nrows == 0 && return
    kernel! = normalize_kernel!(backend, DEFAULT_GROUPSIZE)
    kernel!(A, k; ndrange=nrows)
end

"""
    updatesubmatrix_naive!(A, k)

Perform the rank-1 update `A[k+1:n, k+1:n] -= A[k+1:n, k] * A[k, k+1:n]`
in-place using the naive global-memory kernel. Launches
`updatesubmatrix_naive_kernel!` on the backend inferred from `A`.
"""
function updatesubmatrix_naive!(A::AbstractMatrix, k::Int)
    n = size(A, 1)
    backend = get_backend(A)
    m = n - k
    m == 0 && return
    kernel! = updatesubmatrix_naive_kernel!(backend, GROUPSIZE_2D)
    kernel!(A, k; ndrange=(m, m))
end

"""
    updatesubmatrix!(A, k)

Perform the rank-1 update `A[k+1:n, k+1:n] -= A[k+1:n, k] * A[k, k+1:n]`
in-place using the COLUMN-strategy shared-memory kernel. Launches
`updatesubmatrix_kernel!` on the backend inferred from `A`.
"""
function updatesubmatrix!(A::AbstractMatrix, k::Int)
    n = size(A, 1)
    backend = get_backend(A)
    m = n - k
    m == 0 && return
    kernel! = updatesubmatrix_kernel!(backend, GROUPSIZE_2D)
    kernel!(A, k; ndrange=(m, m))
end

# =============================================================================
# LU decomposition driver
# =============================================================================

"""
    lu_decomp!(A) -> (A, perm)

Perform in-place LU decomposition with partial pivoting on square matrix `A`
using GPU kernels via KernelAbstractions.

On return:
- The lower triangular part of `A` (below the diagonal) contains the
  multipliers `L` (unit diagonal is implicit, not stored).
- The upper triangular part of `A` (including the diagonal) contains `U`.
- `perm` is a CPU `Vector{Int}` recording the row permutations applied.

`A` can live on any KernelAbstractions-compatible backend (CPU, CUDA, Metal,
oneAPI, AMDGPU). Element type `T` can be any numeric type, though floating-
point types are recommended for numerical stability.

# Example
```julia
using KernelAbstractions, GPULUTutorial
backend = CPU()
A = KernelAbstractions.zeros(backend, Float32, 4, 4)
# ... fill A ...
A_factored, perm = lu_decomp!(A)
```
"""
function lu_decomp!(A::AbstractMatrix{T}) where {T}
    n = size(A, 1)
    @assert size(A, 2) == n "Matrix must be square, got $(size(A))"

    backend = get_backend(A)
    perm = collect(1:n)  # permutation vector lives on CPU

    for k in 1:(n - 1)
        # Step 1: find the pivot row
        pivot_row = findpivot!(A, k)
        perm[k], perm[pivot_row] = perm[pivot_row], perm[k]

        # Step 2: swap rows k and pivot_row
        if pivot_row != k
            swaprows!(A, k, pivot_row)
        end
        KernelAbstractions.synchronize(backend)

        # Step 3: normalize column k below the diagonal
        normalize!(A, k)
        KernelAbstractions.synchronize(backend)

        # Step 4: rank-1 update of the trailing submatrix
        updatesubmatrix!(A, k)
        KernelAbstractions.synchronize(backend)
    end

    return A, perm
end
