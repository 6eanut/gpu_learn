import torch
import triton
import triton.language as tl
import triton.testing

# 逐块求结果矩阵(分组)
@triton.autotune(
    configs=[
        # 针对小方阵的配置 (256-512)
        triton.Config({'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=3,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=3,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        
        # 针对中等方阵的配置 (512-1024)
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),
        
        # 针对大方阵的配置 (1024-2048)
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=5,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),
        
        # 针对超大方阵的配置 (2048-4096)
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':256,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':1024,'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=32),
        triton.Config({'BLOCK_SIZE_M':1024,'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=32),
        
        # 非对称配置，针对某些特定尺寸的优化
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        
        # 不同阶段数的配置
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=2,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=5,num_warps=8),
        
        # 不同warp数的配置
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=32),
        
        # 不同GROUP_SIZE_M值的配置
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':1},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':4},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':16},num_stages=3,num_warps=8),
        
        # 针对特定尺寸的专门优化
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=3,num_warps=4),  # 256-512
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=8),  # 512-1024
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=16), # 1024-2048
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=4,num_warps=16), # 2048-4096
        
        # 更多的组合
        triton.Config({'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=4,num_warps=4),
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=4,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=5,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=5,num_warps=16),
        
        # 针对大K值的配置
        triton.Config({'BLOCK_SIZE_M':128,'BLOCK_SIZE_N':128,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=8),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':256,'GROUP_SIZE_M':8},num_stages=3,num_warps=16),
        
        # 针对不同GPU架构的优化
        triton.Config({'BLOCK_SIZE_M':64,'BLOCK_SIZE_N':64,'BLOCK_SIZE_K':32,'GROUP_SIZE_M':8},num_stages=5,num_warps=2),  # 针对低端GPU
        triton.Config({'BLOCK_SIZE_M':256,'BLOCK_SIZE_N':256,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=4,num_warps=8),  # 针对中端GPU
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=4,num_warps=16),  # 针对高端GPU
        
        # 极端配置，用于探索性能边界
        triton.Config({'BLOCK_SIZE_M':1024,'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':64,'GROUP_SIZE_M':8},num_stages=2,num_warps=32),
        triton.Config({'BLOCK_SIZE_M':1024,'BLOCK_SIZE_N':1024,'BLOCK_SIZE_K':128,'GROUP_SIZE_M':8},num_stages=2,num_warps=32),
        triton.Config({'BLOCK_SIZE_M':512,'BLOCK_SIZE_N':512,'BLOCK_SIZE_K':256,'GROUP_SIZE_M':8},num_stages=2,num_warps=32),
    ],
    key=['M','N','K'],
)
@triton.jit
def matmul_kernel(
      a_ptr, b_ptr, c_ptr,
      M, N, K,
      stride_am, stride_ak,
      stride_bk, stride_bn,
      stride_cm, stride_cn,
      BLOCK_SIZE_M:tl.constexpr, BLOCK_SIZE_N:tl.constexpr, BLOCK_SIZE_K:tl.constexpr, GROUP_SIZE_M:tl.constexpr
      ):
  # program <-> one block of matrix c <-> pid_m, pid_n, BLOCK_SIZE_M, BLOCK_SIZE_N
  pid = tl.program_id(axis=0)
  # how many blocks in m and n ?
  num_pid_m = tl.cdiv(M,BLOCK_SIZE_M)
  num_pid_n = tl.cdiv(N,BLOCK_SIZE_N)
  # how many blocks in a group ?
  num_pid_in_group = GROUP_SIZE_M * num_pid_n
  # which group current program exists in ?
  group_id = pid // num_pid_in_group
  # start of the group ?
  first_pid_m = group_id * GROUP_SIZE_M
  # if the number of blocks in m isn't divisible by GROUP_SIZE_M, then the last group has not GROUP_SIZE_M lines
  group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
  # the idx of block in group
  pid_in_group = pid % num_pid_in_group
  # pid_m
  pid_m = first_pid_m + pid_in_group % group_size_m
  # pid_n
  pid_n = pid_in_group // group_size_m

  tl.assume(pid_m>=0)
  tl.assume(pid_n>=0)
  tl.assume(stride_am>0)
  tl.assume(stride_ak>0)
  tl.assume(stride_bk>0)
  tl.assume(stride_bn>0)
  tl.assume(stride_cm>0)
  tl.assume(stride_cn>0)

  # element in block
  offs_am = (pid_m*BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M))%M
  offs_bn = (pid_n*BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N))%N
  offs_k = tl.arange(0,BLOCK_SIZE_K)
  a_ptrs = a_ptr + (offs_am[:,None]*stride_am + offs_k[None,:]*stride_ak)
  b_ptrs = b_ptr + (offs_k[:,None]*stride_bk + offs_bn[None,:]*stride_bn)

  # do the multiply
  accumulator = tl.zeros((BLOCK_SIZE_M,BLOCK_SIZE_N),dtype=tl.float32)
  for k in range(0,tl.cdiv(K,BLOCK_SIZE_K)):
    a=tl.load(a_ptrs, mask=offs_k[None,:]<K-k*BLOCK_SIZE_K,other=0.0)
    b=tl.load(b_ptrs, mask=offs_k[:,None]<K-k*BLOCK_SIZE_K,other=0.0)
    accumulator = tl.dot(a,b, accumulator)
    a_ptrs+=BLOCK_SIZE_K*stride_ak
    b_ptrs+=BLOCK_SIZE_K*stride_bk
  c = accumulator.to(tl.float16)

  # write the result in c
  offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0,BLOCK_SIZE_M)
  offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0,BLOCK_SIZE_N)
  c_ptrs = c_ptr+stride_cm*offs_cm[:,None] + stride_cn * offs_cn[None,:]
  c_mask =(offs_cm[:,None]<M)&(offs_cn[None,:]<N)
  tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b):
  assert a.shape[1] == b.shape[0], "Incompatible dimensions"
  assert a.is_contiguous(), "Matrix A bust be contiguous"
  M, K = a.shape
  K, N = b.shape
  c = torch.empty((M,N),device=a.device, dtype=torch.float16)
  grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M'])*triton.cdiv(N, META['BLOCK_SIZE_N']),)
  matmul_kernel[grid](a,b,c,M,N,K,a.stride(0),a.stride(1),b.stride(0),b.stride(1),c.stride(0),c.stride(1))
  return c
  
DEVICE = 'cuda'

ref_lib = 'cuBLAS'

configs = [
    triton.testing.Benchmark(
        x_names=["M", "N", "K"],
        x_vals=[128 * i for i in range(2, 33)],  # 从 256 到 4096
        line_arg="provider",
        line_vals=[ref_lib.lower(), "triton"],
        line_names=[ref_lib, "Triton"],
        styles=[("blue", "-"), ("green", "-")],
        ylabel="TFLOPS",
        plot_name="matmul-performance-fp16",
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark(M, N, K, provider):
    a = torch.randn((M, K), device=DEVICE, dtype=torch.float16)
    b = torch.randn((K, N), device=DEVICE, dtype=torch.float16)

    quantiles = [0.5, 0.2, 0.8]
    if provider == ref_lib.lower():
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)

    perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
    return perf(ms), perf(max_ms), perf(min_ms)

benchmark.run(show_plots=True, print_data=True)