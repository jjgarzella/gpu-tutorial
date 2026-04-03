using KernelAbstractions

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
    findpivot_kernel!(A, k, pivot, pivot_val)

Find the row index of the maximum absolute value in column `k`, searching
rows `k` through `n`. The index is stored in `pivot[1]` and the value in
`pivot_val[1]` (both are 1-element device arrays).

Each thread is responsible for one row offset `i` (1-based), corresponding
to actual row `k + i - 1`.

TODO: Implement this kernel.
  - Compute `val = abs(A[row, k])` for the thread's row.
  - Use atomics or a reduction to find the global maximum across threads
    and record its row index in `pivot[1]`.
"""
@kernel function findpivot_kernel!(A, @Const(k), pivot, pivot_val)
    i = @index(Global, Linear)
    row = k + i - 1
    # TODO: implement atomic max reduction to find pivot row
    _ = abs(A[row, k])  # placeholder — suppress unused warning
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
    j = @index(Global, Linear)
    # TODO: swap A[k, j] and A[pivot_row, j]
    _ = A[k, j]  # placeholder
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
    i = @index(Global, Linear)
    row = k + i
    # TODO: divide A[row, k] by A[k, k]
    _ = A[row, k]  # placeholder
end

"""
    updatesubmatrix_kernel!(A, k)

Perform the rank-1 update of the trailing submatrix:
    A[k+1:n, k+1:n] -= A[k+1:n, k] * A[k, k+1:n]

Each thread is responsible for one `(i, j)` pair (1-based offsets),
corresponding to actual indices `(k+i, k+j)`.

TODO: Implement this kernel.
  - Compute `A[row, col] -= A[row, k] * A[k, col]`.
"""
@kernel function updatesubmatrix_kernel!(A, @Const(k))
    i, j = @index(Global, NTuple)
    row = k + i
    col = k + j
    # TODO: perform the rank-1 update at (row, col)
    _ = A[row, col]  # placeholder
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

"""
    findpivot!(A, k) -> Int

Find the row index (1-based) of the maximum absolute value in column `k` of
`A`, searching rows `k:n`. Launches `findpivot_kernel!` on the backend
inferred from `A`.

Returns the pivot row index as a CPU `Int`.
"""
function findpivot!(A::AbstractMatrix, k::Int)
    n = size(A, 1)
    backend = get_backend(A)

    # Allocate 1-element device arrays for the reduction result.
    # Initialize pivot to k so that the stub (which leaves pivot unchanged)
    # returns a valid row index instead of 0.
    pivot     = similar(A, Int, 1)
    copyto!(pivot, [k])
    pivot_val = KernelAbstractions.zeros(backend, eltype(A), 1)

    nrows = n - k + 1
    kernel! = findpivot_kernel!(backend, DEFAULT_GROUPSIZE)
    kernel!(A, k, pivot, pivot_val; ndrange=nrows)

    # Transfer result to CPU.
    return Int(Array(pivot)[1])
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
    updatesubmatrix!(A, k)

Perform the rank-1 update `A[k+1:n, k+1:n] -= A[k+1:n, k] * A[k, k+1:n]`
in-place. Launches `updatesubmatrix_kernel!` on the backend inferred from `A`.
"""
function updatesubmatrix!(A::AbstractMatrix, k::Int)
    n = size(A, 1)
    backend = get_backend(A)
    m = n - k
    m == 0 && return
    kernel! = updatesubmatrix_kernel!(backend, DEFAULT_GROUPSIZE)
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
