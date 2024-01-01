import torch
import triton
import triton.language as tl
from time import time
N = 8192*10000



@triton.jit
def add_kernel(x_ptr, y_ptr, out_ptr, x_len, BLOCK_SIZE: tl.constexpr
                ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < x_len
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    out = x+y
    tl.store(out_ptr+offsets, out, mask=mask)
    


def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_len = torch.numel(x)
    block_size = triton.next_power_of_2(x_len)
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(x_len, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, out, x_len, BLOCK_SIZE=256)  # block size seems to be the quickest from testing
    return out
    

def main():
    a = torch.arange(0, N+1, dtype=torch.float32).to("cuda")
    b = torch.arange(0, N+1, dtype=torch.float32).to("cuda")
    start = time()
    result = a + b
    end = time()
    torch_time = end - start 
    print("Sum of tensors: ", result, " time: ", torch_time)
    start=time()
    result = triton_add(a,b)
    end=time()
    triton_time = end - start 
    print("Sum of tensors", result, "time", triton_time)
    relative_gain = torch_time/triton_time
    print("Relative speedup", relative_gain)
    
if __name__ == '__main__':
    main()