import triton
import triton.language as tl
import torch
from time import time
N = 2



@triton.jit
def matmul_kernel(x_ptr, y_ptr, n, out_ptr, out_len, BLOCK_SIZE:tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1)

    row_offsets = tl.arange(0, BLOCK_SIZE)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row_mask = row_offsets < n
    col_mask = col_offsets < n
    
    # row * col = C[row, col]
    
    acc = tl.zeros((BLOCK_SIZE,), dtype= tl.float32)
    
    for _ in range(0, tl.cdiv(n, BLOCK_SIZE)):
        x_row = tl.load(x_ptr + (row*n) + col_offsets, mask=col_mask)
        y_col = tl.load(y_ptr + col + (row_offsets*n), mask=row_mask)
        for k in range(0, BLOCK_SIZE):
            acc += x_row[k] * y_col[k]
        x_ptr += BLOCK_SIZE 
        y_ptr += BLOCK_SIZE * n
        
    out_ptr = out_ptr + (row*n) + col
    tl.store(out_ptr, acc, mask=out_ptr<out_len)
    
    
def triton_matmul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.shape == y.shape # only allow equal shape
    cols = x.shape[1]
    out = torch.empty_like(x)
    grid = (cols, cols)
    matmul_kernel[grid](x, y, cols, out, out.shape[0] * out.shape[1], BLOCK_SIZE=256)
    return out
    
    
def main():
    x = torch.randn((N, N), dtype=torch.float32).to("cuda")
    y = torch.randn((N, N), dtype=torch.float32).to("cuda")
    start = time()
    res = torch.matmul(x, y)
    end = time()
    pytorch_time = end-start
    print("TIme for pytorch", pytorch_time)
    start=time()
    res = triton_matmul(x, y)
    end = time()
    triton_time = end-start
    print("Time for triton", triton_time)
    
if __name__ == '__main__':
    main()