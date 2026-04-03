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
    findpivot_kernel!(A, k, pivot_val)

Write the absolute value of `A[row, k]` for each thread's row into the
scratch buffer `pivot_val`. Each thread writes its own slot so no atomics
are needed; the host picks the argmax after a blocking device-to-host copy.

Each thread is responsible for one row offset `i` (1-based), corresponding
to actual row `k + i - 1`.
"""
@kernel function findpivot_kernel!(A, @Const(k), pivot_val)
    gs       = @groupsize()[1]
    i_global = @index(Global, Linear)
    group_id = (i_global - 1) ÷ gs        # 0-indexed workgroup
    local_id = (i_global - 1) % gs        # 0-indexed lane within workgroup
    i        = group_id * gs + local_id + 1
    row = k + i - 1
    pivot_val[i] = abs(A[row, k])
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
    gs       = @groupsize()[1]
    j_global = @index(Global, Linear)
    group_id = (j_global - 1) ÷ gs
    local_id = (j_global - 1) % gs
    j        = group_id * gs + local_id + 1
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
    gs       = @groupsize()[1]
    i_global = @index(Global, Linear)
    group_id = (i_global - 1) ÷ gs
    local_id = (i_global - 1) % gs
    i        = group_id * gs + local_id + 1
    row = k + i
    A[row, k] = _div(A[row, k], A[k, k])
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
    gs               = @groupsize()
    i_global, j_global = @index(Global, NTuple)
    group_i  = (i_global - 1) ÷ gs[1]
    group_j  = (j_global - 1) ÷ gs[2]
    local_i  = (i_global - 1) % gs[1]
    local_j  = (j_global - 1) % gs[2]
    i        = group_i * gs[1] + local_i + 1
    j        = group_j * gs[2] + local_j + 1
    row = k + i
    col = k + j
    A[row, col] = muladd(-A[row, k], A[k, col], A[row, col])
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

    nrows = n - k + 1
    pivot_val = KernelAbstractions.zeros(backend, eltype(A), nrows)

    kernel! = findpivot_kernel!(backend, DEFAULT_GROUPSIZE)
    kernel!(A, k, pivot_val; ndrange=nrows)

    # Array() is a blocking device-to-host transfer; argmax gives the offset.
    return argmax(Array(pivot_val)) + k - 1
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
