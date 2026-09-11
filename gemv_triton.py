import torch
import triton
import triton.language as tl

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False

def triton_gemv(
    W: torch.Tensor,
    x: torch.Tensor,
    BLOCK_SIZE: int = 1024,
) -> torch.Tensor:
    if W.shape[1] != x.shape[0]:
        raise ValueError
    M, N = W.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    gemv_kernel[(M,)](
        W,
        x,
        y,
        N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return y

@triton.jit
def gemv_kernel(
    W_ptr,
    x_ptr,
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)

    acc = tl.zeros([BLOCK_SIZE, ], dtype=tl.float32)

    for start in range(0, N, BLOCK_SIZE):
        cols = start + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        W = tl.load(W_ptr + row * N + cols, mask=mask, other=0.0)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        acc += W.to(tl.float32) * x.to(tl.float32)

    result = tl.sum(acc, axis=0)

    tl.store(out_ptr + row, result)

def triton_tiled_gemv(
    W: torch.Tensor,
    x: torch.Tensor,
    BLOCK_K: int = 1024,
    BLOCK_N: int = 2,
) -> torch.Tensor:
    if W.shape[1] != x.shape[0]:
        raise ValueError
    M, N = W.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    gemv_tiled_kernel[(triton.cdiv(M, BLOCK_N),)](
        W,
        x,
        y,
        N,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    return y

@triton.jit
def gemv_tiled_kernel(
    W_ptr,
    x_ptr,
    out_ptr,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)

    for rows in range(BLOCK_N):
        acc = tl.zeros([BLOCK_K], dtype=tl.float32)
        for start in range(0, N, BLOCK_K):
            cols = start + tl.arange(0, BLOCK_K)
            mask = cols < N
            W = tl.load(W_ptr + (row * N * BLOCK_N + rows * N) + cols, mask=mask, other=0.0)
            x = tl.load(x_ptr + cols, mask=mask, other=0.0)
            acc += W.to(tl.float32) * x.to(tl.float32)
        result = tl.sum(acc, axis=0)
        tl.store(out_ptr + (row * BLOCK_N + rows), result)

# —————

def triton2d_tiled_gemv(
    W: torch.Tensor,
    x: torch.Tensor,
    BLOCK_K: int = 1024,
    BLOCK_N: int = 2,
) -> torch.Tensor:
    if W.shape[1] != x.shape[0]:
        raise ValueError
    M, N = W.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    gemv2d_tiled_kernel[(triton.cdiv(M, BLOCK_N),)](
        W,
        x,
        y,
        N,
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
    )
    return y

@triton.jit
def gemv2d_tiled_kernel(
    W_ptr,
    x_ptr,
    out_ptr,
    N,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr
):
    row = tl.program_id(0)

    acc = tl.zeros([BLOCK_N, BLOCK_K], dtype=tl.float32)

    rows = row * BLOCK_N + tl.arange(0, BLOCK_N)

    for start in range(0, N, BLOCK_K):
        cols = start + tl.arange(0, BLOCK_K)
        mask = cols < N
        W = tl.load(W_ptr + rows[:, None] * N + cols[None, :], mask=mask, other=0.0)
        x = tl.load(x_ptr + cols, mask=mask, other=0.0)
        acc += W.to(tl.float32) * x.to(tl.float32)[None, :]
    result = tl.sum(acc, axis=1)
    tl.store(out_ptr + rows, result)


# —————
print('——— Config ———')
print('dtype=bfloat16')
print('device=cuda')
print()
for size in [2048, 4096, 8192, 16384]:
    W = torch.randn(size, size, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(size, device="cuda", dtype=torch.bfloat16)
    print('——— cuBLAS ———')
    print(f"{'N':>5} {'MB moved':>8} {'ms':>9} {'GB/s':>8} {'all_close':>9}")

    y_ref = W @ x
    print(y_ref)
    ms_native = triton.testing.do_bench(lambda: W @ x)

    bytes_needed: int = (
        W.numel() * W.element_size() + x.numel() * x.element_size() * 2
    )

    print(f"{size:>5} {bytes_needed / 1e6:>8.2f} {ms_native:>9.6f} {(bytes_needed / (ms_native / 1000)) / 1e9:>8.2f} {'GT'}")
    print()

    print('——— Triton1d ———')
    print(f"{'Block_size':>10} {'ms':>9} {'GB/s':>8} {'all_close':>9}")
    for BLOCK_SIZE in [256, 512, 1024, 2048]:
        y_gemv = triton_gemv(
            W, x, BLOCK_SIZE
        )
        ms_triton = triton.testing.do_bench(lambda: triton_gemv(
            W, x, BLOCK_SIZE
        ))
        gbps = bytes_needed / (ms_triton / 1000) / 1e9

        print(f"{BLOCK_SIZE:>10} {ms_triton:>9.6f} {gbps:>8.2f} {torch.allclose(y_ref, y_gemv, atol=1e-2, rtol=1e-2):>9}")
    print()

    print('——— Triron2d NAIVE ———')
    print(f"{'BLOCK_N':>7} {'BLOCK_K':>7} {'ms':>9} {'GB/s':>8} {'all_close':>9}")

    for rows_at_same_kernel in [2, 4, 8]:
        for BLOCK_SIZE in [256, 512, 1024, 2048]:
            y_gemv_tiled = triton_tiled_gemv(
                W, x, BLOCK_SIZE, rows_at_same_kernel
            )
            ms_triton = triton.testing.do_bench(lambda: triton_tiled_gemv(
                W, x, BLOCK_SIZE, rows_at_same_kernel
            ))
            gbps = bytes_needed / (ms_triton / 1000) / 1e9

            print(f"{rows_at_same_kernel:>6} {BLOCK_SIZE:>6} {ms_triton:>9.6f} {gbps:>8.2f} {torch.allclose(y_gemv_tiled, y_ref, atol=1e-2, rtol=1e-2):>9}")
    print()

    print('——— Triron2d TRUE ———')
    print(f"{'BLOCK_N':>7} {'BLOCK_K':>7} {'ms':>9} {'GB/s':>8} {'all_close':>9}")
    for rows_at_same_kernel in [2, 4, 8]:
        for BLOCK_SIZE in [256, 512, 1024, 2048]:
            y_gemv_tiled2d = triton2d_tiled_gemv(
                W, x, BLOCK_SIZE, rows_at_same_kernel
            )
            ms_triton = triton.testing.do_bench(lambda: triton2d_tiled_gemv(
                W, x, BLOCK_SIZE, rows_at_same_kernel
            ))
            gbps = bytes_needed / (ms_triton / 1000) / 1e9

            print(f"{rows_at_same_kernel:>6} {BLOCK_SIZE:>6} {ms_triton:>9.6f} {gbps:>8.2f} {torch.allclose(y_gemv_tiled2d, y_ref, atol=1e-1, rtol=1e-1):>9}")
    print()
