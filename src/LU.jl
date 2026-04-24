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
    findpivot_firstpass_kernel!(A, k, nrows, vals_out, idxs_out)

First pass of the Harris-pattern parallel argmax reduction over column `k` of
`A`, searching rows `k:k+nrows-1`. Each workgroup consumes `2 *
DEFAULT_GROUPSIZE` consecutive elements of the column and emits a single
`(max_abs_val, row_index)` pair to `vals_out[gi]`, `idxs_out[gi]`.

Harris "first add during load": each thread pre-reduces two strided elements
of the column before writing to shared memory, halving the number of shared-
memory slots used by the tree reduction.

Launch with `ndrange = ngroups * DEFAULT_GROUPSIZE` where `ngroups = cld(nrows,
2 * DEFAULT_GROUPSIZE)`. Out-of-range threads contribute a neutral `(0, 0)`.
"""
@kernel function findpivot_firstpass_kernel!(A, @Const(k), @Const(nrows), vals_out, idxs_out)
    gi = @index(Group, Linear)
    li = @index(Local, Linear)

    svals = @localmem eltype(A) (DEFAULT_GROUPSIZE,)
    sidxs = @localmem Int32 (DEFAULT_GROUPSIZE,)

    base = (gi - 1) * 2 * DEFAULT_GROUPSIZE
    i1   = base + li
    i2   = base + li + DEFAULT_GROUPSIZE

    v1 = eltype(A)(0); x1 = Int32(0)
    v2 = eltype(A)(0); x2 = Int32(0)
    if i1 <= nrows
        row1 = k + i1 - 1
        v1 = abs(A[row1, k])
        x1 = Int32(row1)
    end
    if i2 <= nrows
        row2 = k + i2 - 1
        v2 = abs(A[row2, k])
        x2 = Int32(row2)
    end
    if v2 > v1
        svals[li] = v2; sidxs[li] = x2
    else
        svals[li] = v1; sidxs[li] = x1
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
        vals_out[gi] = svals[1]
        idxs_out[gi] = sidxs[1]
    end
end

"""
    findpivot_reduce_kernel!(vals_in, idxs_in, vals_out, idxs_out, len)

Subsequent pass of the multi-pass argmax reduction. Reads `len` pre-reduced
pairs from `(vals_in, idxs_in)` and emits `cld(len, 2*DEFAULT_GROUPSIZE)`
further-reduced pairs into `(vals_out, idxs_out)`. Same Harris pre-reduce +
unrolled tree reduction shape as the first pass; out-of-range threads
contribute a neutral `(0, 0)`.
"""
@kernel function findpivot_reduce_kernel!(vals_in, idxs_in, vals_out, idxs_out, @Const(len))
    gi = @index(Group, Linear)
    li = @index(Local, Linear)

    svals = @localmem eltype(vals_in) (DEFAULT_GROUPSIZE,)
    sidxs = @localmem Int32 (DEFAULT_GROUPSIZE,)

    base = (gi - 1) * 2 * DEFAULT_GROUPSIZE
    p1   = base + li
    p2   = base + li + DEFAULT_GROUPSIZE

    v1 = eltype(vals_in)(0); x1 = Int32(0)
    v2 = eltype(vals_in)(0); x2 = Int32(0)
    if p1 <= len
        v1 = vals_in[p1]
        x1 = idxs_in[p1]
    end
    if p2 <= len
        v2 = vals_in[p2]
        x2 = idxs_in[p2]
    end
    if v2 > v1
        svals[li] = v2; sidxs[li] = x2
    else
        svals[li] = v1; sidxs[li] = x1
    end
    @synchronize()

    if li <= 128; if svals[li+128] > svals[li]; svals[li] = svals[li+128]; sidxs[li] = sidxs[li+128]; end; end; @synchronize()
    if li <=  64; if svals[li+ 64] > svals[li]; svals[li] = svals[li+ 64]; sidxs[li] = sidxs[li+ 64]; end; end; @synchronize()
    if li <=  32; if svals[li+ 32] > svals[li]; svals[li] = svals[li+ 32]; sidxs[li] = sidxs[li+ 32]; end; end; @synchronize()
    if li <=  16; if svals[li+ 16] > svals[li]; svals[li] = svals[li+ 16]; sidxs[li] = sidxs[li+ 16]; end; end; @synchronize()
    if li <=   8; if svals[li+  8] > svals[li]; svals[li] = svals[li+  8]; sidxs[li] = sidxs[li+  8]; end; end; @synchronize()
    if li <=   4; if svals[li+  4] > svals[li]; svals[li] = svals[li+  4]; sidxs[li] = sidxs[li+  4]; end; end; @synchronize()
    if li <=   2; if svals[li+  2] > svals[li]; svals[li] = svals[li+  2]; sidxs[li] = sidxs[li+  2]; end; end; @synchronize()
    if li <=   1; if svals[li+  1] > svals[li]; svals[li] = svals[li+  1]; sidxs[li] = sidxs[li+  1]; end; end; @synchronize()

    if li == 1
        vals_out[gi] = svals[1]
        idxs_out[gi] = sidxs[1]
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
    findpivot!(A, k; pivot_temp=nothing) -> Int

Find the row index (1-based) of the maximum absolute value in column `k` of
`A`, searching rows `k:n`. Uses a multi-pass Harris-pattern reduction that
completes the argmax entirely on-device; only a single `Int32` is transferred
to the host per call.

`pivot_temp` is a ping-pong scratch buffer `(; vals, idxs)` with each array
sized `2 * cld(n, 2 * DEFAULT_GROUPSIZE)`. If omitted, it is allocated
internally (convenient but adds an allocation per call; the `lu_decomp!`
driver hoists this allocation out of the loop).

Returns the pivot row index as a CPU `Int`.
"""
function findpivot!(A::AbstractMatrix, k::Int; pivot_temp=nothing)
    n = size(A, 1)
    backend = get_backend(A)
    nrows = n - k + 1

    max_half = cld(n, 2 * DEFAULT_GROUPSIZE)
    if pivot_temp === nothing
        vals = KernelAbstractions.zeros(backend, eltype(A), 2 * max_half)
        idxs = KernelAbstractions.zeros(backend, Int32,     2 * max_half)
        pivot_temp = (; vals, idxs)
    end

    vals_a = @view pivot_temp.vals[1:max_half]
    vals_b = @view pivot_temp.vals[max_half+1:2*max_half]
    idxs_a = @view pivot_temp.idxs[1:max_half]
    idxs_b = @view pivot_temp.idxs[max_half+1:2*max_half]

    # First pass: column of A -> `len` pair-buffer entries.
    len = cld(nrows, 2 * DEFAULT_GROUPSIZE)
    firstpass! = findpivot_firstpass_kernel!(backend, DEFAULT_GROUPSIZE)
    firstpass!(A, k, nrows, vals_a, idxs_a; ndrange=len * DEFAULT_GROUPSIZE)

    # Ping-pong reduce passes until a single pair remains.
    front_v, front_i = vals_a, idxs_a
    back_v,  back_i  = vals_b, idxs_b
    reduce! = findpivot_reduce_kernel!(backend, DEFAULT_GROUPSIZE)
    while len > 1
        new_len = cld(len, 2 * DEFAULT_GROUPSIZE)
        reduce!(front_v, front_i, back_v, back_i, len; ndrange=new_len * DEFAULT_GROUPSIZE)
        front_v, front_i, back_v, back_i = back_v, back_i, front_v, front_i
        len = new_len
    end

    return Int(Array(front_i[1:1])[1])
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
