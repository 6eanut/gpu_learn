import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.jit
def flash_attention(
    q_ptr, k_ptr, v_ptr, o_ptr,
    seq_len, d_model: tl.constexpr,
    stride_qm, stride_km, stride_vm,
    BLOCK_M: tl.constexpr,  # Br: Q的分块大小
    BLOCK_N: tl.constexpr,  # Bc: K,V的分块大小
):
    # 只对序列维度进行并行化
    pid = tl.program_id(0)
    start_m = pid * BLOCK_M
    
    offs_m = start_m + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, d_model)  # 处理整个特征维度
    
    # 初始化状态变量
    m_prev = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_prev = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, d_model), dtype=tl.float32)
    
    # 加载Q分块 (Br × d)
    q_mask = offs_m[:, None] < seq_len
    q = tl.load(
        q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :],
        mask=q_mask,
        other=0.0
    )
    
    # 按照FlashAttention算法，遍历K,V的分块
    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        
        # 加载K分块 (Bc × d)
        k = tl.load(
            k_ptr + offs_n[:, None] * stride_km + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0
        )
        
        # 加载V分块 (Bc × d)
        v = tl.load(
            v_ptr + offs_n[:, None] * stride_vm + offs_d[None, :],
            mask=mask_n[:, None],
            other=0.0
        )
        
        # 计算Sij = Qi @ Kj^T
        s = tl.dot(q, k.T)
        s *= 1.0 / tl.sqrt(tl.cast(d_model, s.dtype))
        
        # 掩码无效位置
        s = tl.where(mask_n[None, :], s, float('-inf'))
        
        # 在线Softmax计算(FLASH2)
        m_current = tl.maximum(tl.max(s, axis=1), m_prev)
        
        # 数值稳定的指数计算
        # 
        exp_m_prev = tl.exp(m_prev - m_current)
        exp_s = tl.exp(s - m_current[:, None])
        
        l_current = exp_m_prev * l_prev + tl.sum(exp_s, axis=1)
        
        # 更新累加器
        acc = acc * exp_m_prev[:, None] + tl.dot(exp_s, v)
        
        # 更新状态
        m_prev = m_current
        l_prev = l_current
    
    # 归一化并写入结果
    acc = acc / l_prev[:, None]
    tl.store(
        o_ptr + offs_m[:, None] * stride_qm + offs_d[None, :],
        acc,
        mask=q_mask
    )

def call_flash_attention(q, k, v, block_m, block_n):
    """
    block_m: Q的分块大小Br
    block_n: K,V的分块大小Bc
    """
    assert q.shape == k.shape == v.shape
    seq_len, d_model = q.shape

    o = torch.empty_like(q)
    
    # 只在序列维度上并行化
    grid = (triton.cdiv(seq_len, block_m),)
    
    flash_attention[grid](
        q, k, v, o,
        seq_len, d_model,
        q.stride(0), k.stride(0), v.stride(0),
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
    return o

def calculate_block_sizes(d_model, sram_size_kb=48):
    """
    计算FlashAttention的最优分块大小
    
    Args:
        d_model: 特征维度
        sram_size_kb: 共享内存大小（KB）
    
    Returns:
        block_m, block_n: Q和K/V的分块大小
    """
    # 将KB转换为字节
    sram_size = sram_size_kb * 1024
    
    # float32占4字节
    bytes_per_elem = 4
    
    # 根据论文，Bc的计算公式
    # 先计算Bc的上界
    bc = sram_size // (4 * d_model * bytes_per_elem)
    
    # 根据论文，Br = min(Bc, d_model)
    br = min(bc, d_model)
    
    # 确保是2的幂
    bc = min(bc, 128) 
    bc = 2 ** (bc.bit_length() - 1) if bc > 0 else 1
    
    br = min(bc, d_model)
    br = 2 ** (br.bit_length() - 1) if br > 0 else 1
    
    return br, bc

def torch_attention(q, k, v):
    # d_k = d_model
    d_k = q.size(-1)
    # 为了防止注意力分数方差过大导致softmax梯度消失，需要根号下d_k这个缩放因子
    # 方差​​就是​​衡量一组数据与其平均值的偏离程度​
    # softmax函数对极端输入值非常敏感
    attn_scores = q @ k.transpose(-2, -1) / (d_k ** 0.5) 
    # 在最后一个维度上进行softmax操作
    attn_probs = torch.softmax(attn_scores, dim=-1)
    return attn_probs @ v 

# 序列长度
seq_len = 1024 
# 特征维度
d_model = 64

# 初始化Q K V输入
q = torch.randn(seq_len, d_model, device="cuda", dtype=torch.float32)
k = torch.randn_like(q)
v = torch.randn_like(q)

block_m, block_n = calculate_block_sizes(d_model)
    
print(f"BLOCK_M (Br): {block_m}")
print(f"BLOCK_N (Bc): {block_n}")

# 用 Triton 计算
o_triton = call_flash_attention(q, k, v, block_m, block_n)
# 用 PyTorch 计算
o_torch = torch_attention(q, k, v)

print("最大绝对误差:", (o_triton - o_torch).abs().max().item())
# 对于两个张量中的每个对应元素都应该满足
# |o_triton - o_torch| ≤ atol + rtol × |o_torch|
print("是否近似相等:", torch.allclose(o_triton, o_torch, atol=1e-2, rtol=1e-2))