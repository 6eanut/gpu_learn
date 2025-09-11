import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
      a_ptr, b_ptr, c_ptr,
      M, N, K,
      stride_am, stride_ak,
      stride_bk, stride_bn,
      stride_cm, stride_cn,
      BLOCK_SIZE_M: tl.constexpr,
      BLOCK_SIZE_N: tl.constexpr,
      BLOCK_SIZE_K: tl.constexpr,
      GROUP_SIZE_M: tl.constexpr
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
    accumulator += tl.dot(a,b)
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
  grid = lambda META: (
      triton.cdiv(M, 128) * triton.cdiv(N, 128),
  )
  matmul_kernel[grid](a, b, c, M, N, K, 
                    a.stride(0), a.stride(1), 
                    b.stride(0), b.stride(1), 
                    c.stride(0), c.stride(1),
                    BLOCK_SIZE_M=128, 
                    BLOCK_SIZE_N=128, 
                    BLOCK_SIZE_K=32, 
                    GROUP_SIZE_M=8, 
                    num_warps=4)
  return c
  
DEVICE = 'cuda'

ref_lib = 'cuBLAS'

a = torch.randn((1024, 1024), device=DEVICE, dtype=torch.float16)
b = torch.randn((1024, 1024), device=DEVICE, dtype=torch.float16)

c_triton = matmul(a, b)

c_ref = torch.matmul(a, b)