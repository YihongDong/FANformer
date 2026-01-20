# FANformer & StackTrans 代码实现与论文对照报告

本报告详细分析了两种对 Transformer 架构的改进：**FANformer**（傅里叶分析网络）和 **StackTrans**（栈增强Transformer），并展示代码实现与论文公式的对应关系。

---

## 目录

1. [FANformer：周期性建模增强](#1-fanformer周期性建模增强)
   - [1.1 核心思想](#11-核心思想)
   - [1.2 FANLayer 实现](#12-fanlayer-实现)
   - [1.3 ATF 模块](#13-atf-模块)
   - [1.4 Block 集成](#14-block-集成)
2. [StackTrans：栈记忆增强](#2-stacktrans栈记忆增强)
   - [2.1 核心思想](#21-核心思想)
   - [2.2 软更新操作](#22-软更新操作)
   - [2.3 栈掩码机制](#23-栈掩码机制)
   - [2.4 全局读取](#24-全局读取)
   - [2.5 多头低秩栈](#25-多头低秩栈)
   - [2.6 Block 集成](#26-block-集成)
3. [配置参数](#3-配置参数)
4. [架构对比图](#4-架构对比图)

---

## 1. FANformer：周期性建模增强

### 1.1 核心思想

**论文动机**（Section 1-2）：
> Transformer 在周期性模式建模上存在缺陷，导致学习效率低下。FANformer 通过将傅里叶分析网络（FAN）集成到注意力机制中，显式建模周期性特征。

**关键洞察**：
- 周期性是人类学习的基本特征之一
- 标准 Transformer 缺乏对周期函数（如 mod 运算）的归纳偏置
- 通过 cos/sin 编码引入傅里叶原理

### 1.2 FANLayer 实现

#### 论文公式 (Eq. 1)

$$\mathbf{X}_F = \text{FANLayer}'(\mathbf{X}) = [\cos(W_p \mathbf{X}) \| \sin(W_p \mathbf{X}) \| (W_{\bar{p}} \mathbf{X} + B_{\bar{p}})]$$

其中：
- $W_p$ 是周期性投影矩阵
- $W_{\bar{p}}, B_{\bar{p}}$ 是非周期性投影参数
- $\|$ 表示拼接操作
- 超参数 $p$ 控制周期性建模的比例（默认 0.25）

#### 代码实现

**文件**: `olmo/model.py:78-128`

```python
class FANLayer(nn.Module):
    """
    FANLayer: The layer used in FAN (https://arxiv.org/abs/2410.02675).
    
    Args:
        input_dim (int): The number of input features.
        output_dim (int): The number of output features.
        p_ratio (float): The ratio of output dimensions used for cosine and sine parts (default: 0.25).
        activation (str or callable): The activation function to apply to the g component.
        use_p_bias (bool): If True, include bias in the linear transformations of p component.
    """
    
    def __init__(self, input_dim, output_dim, p_ratio=0.25, activation=None, use_p_bias=True):
        super(FANLayer, self).__init__()
        
        # 确保 p_ratio 在有效范围内
        assert 0 <= p_ratio <= 0.5, "p_ratio must be between 0 and 0.5"
        
        self.p_ratio = p_ratio
        p_output_dim = int(output_dim * self.p_ratio)        # 用于 cos/sin 的维度
        g_output_dim = output_dim - p_output_dim * 2          # 剩余维度用于线性部分

        # 单个线性层同时计算 p 和 g
        self.input_linear = nn.Linear(input_dim, p_output_dim + g_output_dim, bias=use_p_bias)
        
        self.fused_dims = (p_output_dim, g_output_dim)
        
        # 激活函数（FANformer 中设为 identity）
        if isinstance(activation, str):
            self.activation = getattr(F, activation)
        else:
            self.activation = activation if activation else lambda x: x

    def forward(self, src, norm_g=None):
        """
        Args:
            src (Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, output_dim).
        """
        # 投影得到 p 和 g
        pg = self.input_linear(src)
        p, g = pg.split(self.fused_dims, dim=-1)
        
        # 拼接 cos(p), sin(p), 和激活后的 g
        # 对应论文公式: [cos(W_p·X) || sin(W_p·X) || σ(W_p̄·X + B_p̄)]
        output = torch.cat((torch.cos(p), torch.sin(p), self.activation(g)), dim=-1)
        
        return output
```

#### 代码与公式对照

| 论文符号 | 代码变量 | 说明 |
|---------|---------|------|
| $W_p$ | `input_linear` 的前 `p_output_dim` 列 | 周期性投影权重 |
| $W_{\bar{p}}$ | `input_linear` 的后 `g_output_dim` 列 | 非周期性投影权重 |
| $\cos(W_p \mathbf{X})$ | `torch.cos(p)` | 余弦编码 |
| $\sin(W_p \mathbf{X})$ | `torch.sin(p)` | 正弦编码 |
| $p$ 比例 | `p_ratio = 0.25` | 默认 25% 维度用于周期性编码 |

### 1.3 ATF 模块

#### 论文公式 (Eq. 2-3)

$$[\mathbf{Q}_F, \mathbf{K}_F, \mathbf{V}_F] = \mathbf{X}_F [\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V]$$

$$\text{ATF}(\mathbf{X} | \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V) = \text{softmax}\left(\frac{\mathbf{Q}_F \mathbf{K}_F^\top}{\sqrt{d_h}}\right) \mathbf{V}_F$$

**关键洞察**（论文 Appendix L）：
$$\text{ATF}(\mathbf{X}) = \text{Attention}(\text{FANLayer}'(\mathbf{X}))$$

这意味着 ATF 可以无缝兼容 FlashAttention 等优化。

#### 代码实现

**文件**: `olmo/model.py:130-138`

```python
class FAN(nn.Module):
    """
    FAN = FANLayer + Linear
    等价于: Attention(FANLayer'(X))
    """
    def __init__(self, input_dim, output_dim, config, activation):
        super(FAN, self).__init__()
        
        # 第一步：FANLayer 进行傅里叶编码
        self.fanlayer = FANLayer(input_dim, input_dim, config.p_ratio, activation)
        # 第二步：线性投影到目标维度
        self.linear = nn.Linear(input_dim, output_dim, bias=config.include_bias, device=config.init_device)
        
    def forward(self, src):
        # X_F = FANLayer'(X)
        # output = X_F @ W_qkv
        return self.linear(self.fanlayer(src))
```

### 1.4 Block 集成

#### 论文架构

> FANformer 通过修改特征投影过程，将 FAN 集成到注意力机制中，同时保持与现有框架（如 FlashAttention）的兼容性。

#### 代码实现

**文件**: `olmo/model.py:769-823` (OLMoSequentialBlock)

```python
class OLMoSequentialBlock(OLMoBlock):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        
        # 是否使用 ATF（FANformer 核心开关）
        self.use_ATF = config.use_ATF

        head_dim = config.d_model // config.n_heads
        self.fused_dims = (
            config.d_model,
            config.effective_n_kv_heads * head_dim,
            config.effective_n_kv_heads * head_dim,
        )
        
        # 关键修改：根据配置选择投影方式
        if self.use_ATF:
            # FANformer: 使用 FAN 进行 QKV 投影
            self.att_proj = FAN(config.d_model, sum(self.fused_dims), config, 
                               activation=config.attention_activation)
        else:
            # 标准 Transformer: 使用普通线性层
            self.att_proj = nn.Linear(
                config.d_model, sum(self.fused_dims), 
                bias=config.include_bias, device=config.init_device
            )
```

#### Diff 对比

```diff
# 标准 Transformer
- self.att_proj = nn.Linear(config.d_model, sum(self.fused_dims), ...)

# FANformer
+ if self.use_ATF:
+     self.att_proj = FAN(config.d_model, sum(self.fused_dims), config, 
+                        activation=config.attention_activation)
+ else:
+     self.att_proj = nn.Linear(config.d_model, sum(self.fused_dims), ...)
```

---

## 2. StackTrans：栈记忆增强

### 2.1 核心思想

**论文动机**（Section 1）：
> Transformer 难以捕获 Chomsky 层次结构（如正则表达式、确定性上下文无关文法）。受下推自动机启发，StackTrans 在 Transformer 层之间引入可微分的隐状态栈。

**与 StackAttn 的区别**：
- StackAttn：修改注意力计算
- StackTrans：在层之间插入栈操作，保持 Transformer 层完整性，兼容 FlashAttention

### 2.2 软更新操作

#### 论文公式 (Eq. 1)

操作概率通过 softmax 计算：
$$\mathbf{a}_t = [a_t^{\text{push}}; a_t^{\text{pop}}; a_t^{\text{noop}}] = \text{Softmax}(A \mathbf{h}_t)$$

#### 论文公式 (Eq. 2)

软更新机制：

$$\text{St}_{t+1}[i] = \begin{cases}
a_t^{\text{push}} \cdot \mathbf{h}_t + a_t^{\text{pop}} \cdot \text{St}_t[1] + a_t^{\text{noop}} \cdot \text{St}_t[0], & \text{if } i = 0 \\
a_t^{\text{push}} \cdot \text{St}_t[i-1] + a_t^{\text{pop}} \cdot \mathbf{0} + a_t^{\text{noop}} \cdot \text{St}_t[i], & \text{if } i = S-1 \\
a_t^{\text{push}} \cdot \text{St}_t[i-1] + a_t^{\text{pop}} \cdot \text{St}_t[i+1] + a_t^{\text{noop}} \cdot \text{St}_t[i], & \text{otherwise}
\end{cases}$$

#### 代码实现

**文件**: `olmo/stack_memory.py:44-50` (动作预测)

```python
class StackMemory(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        # ...
        
        # Action 预测头：输出 push/pop/noop 的 logits
        # 对应论文公式: a_t = Softmax(A·h_t)
        self.action_head = nn.Linear(
            config.d_model, 
            3 * self.num_mem_heads,  # 每个头有 3 种操作
            bias=config.include_bias,
            device=config.init_device
        )
```

**文件**: `olmo/stack_memory.py:128-176` (向量化更新)

```python
def _vectorized_update(
    self, 
    stack: torch.Tensor,    # [batch, seq_len, num_heads, stack_slots, head_dim]
    mask: torch.Tensor,     # [batch, seq_len, num_heads, stack_slots]
    actions: torch.Tensor,  # [batch, seq_len, num_heads, 3]
    k_values: torch.Tensor  # [batch, seq_len, num_heads, head_dim]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    向量化的栈更新操作。
    """
    # Push 操作: 新值放栈顶，其他元素下移
    # 对应论文: push 使每个元素 shift down by one，h_t 放在栈顶
    push_stack = torch.cat([
        k_values.unsqueeze(3),  # 新值放在 position 0 (栈顶)
        stack[:, :, :, :-1]     # 现有值右移（下移）
    ], dim=3)
    push_mask = torch.cat([
        torch.ones_like(mask[:, :, :, :1]),  # 新槽位被占用
        mask[:, :, :, :-1]                   # 掩码右移
    ], dim=3)
    
    # Pop 操作: 移除栈顶，其他元素上移
    # 对应论文: pop 移除栈顶，每个元素 shift up by one
    pop_stack = torch.cat([
        stack[:, :, :, 1:],                      # 值左移（上移）
        torch.zeros_like(stack[:, :, :, :1])     # 底部补零
    ], dim=3)
    pop_mask = torch.cat([
        mask[:, :, :, 1:],                       # 掩码左移
        torch.zeros_like(mask[:, :, :, :1])      # 底部空槽
    ], dim=3)
    
    # 软组合三种操作
    # 对应论文公式 (2): St_{t+1}[i] = a^push * ... + a^pop * ... + a^noop * ...
    action_weights = actions.unsqueeze(-1).unsqueeze(-1)  # [batch, seq, heads, 3, 1, 1]
    stacks = torch.stack([push_stack, pop_stack, stack], dim=3)  # [batch, seq, heads, 3, slots, dim]
    masks = torch.stack([push_mask, pop_mask, mask], dim=3)       # [batch, seq, heads, 3, slots]
    
    new_stack = (stacks * action_weights).sum(dim=3)
    new_mask = (masks * action_weights.squeeze(-1)).sum(dim=3)
    
    return new_stack, new_mask
```

#### 代码与公式对照

| 论文符号 | 代码变量 | 说明 |
|---------|---------|------|
| $A$ | `action_head.weight` | 动作预测矩阵 |
| $a_t^{\text{push/pop/noop}}$ | `actions[:,:,:,0/1/2]` | 三种操作的概率 |
| $\text{St}_t[i]$ | `stack[:,:,:,i,:]` | 栈的第 i 个元素 |
| $\mathbf{h}_t$ | `k_values` | 当前时刻的隐状态 |
| $S$ | `stack_slots` | 栈的大小 |

### 2.3 栈掩码机制

#### 论文公式 (Eq. 3)

$$M_{t+1}[i] = \begin{cases}
a_t^{\text{push}} \cdot 1 + a_t^{\text{pop}} \cdot M_t[1] + a_t^{\text{noop}} \cdot M_t[0], & \text{if } i = 0 \\
a_t^{\text{push}} \cdot M_t[i-1] + a_t^{\text{pop}} \cdot 0 + a_t^{\text{noop}} \cdot M_t[i], & \text{if } i = S-1 \\
a_t^{\text{push}} \cdot M_t[i-1] + a_t^{\text{pop}} \cdot M_t[i+1] + a_t^{\text{noop}} \cdot M_t[i], & \text{otherwise}
\end{cases}$$

#### 代码实现

掩码更新与栈更新采用相同的向量化逻辑（见上方 `_vectorized_update`），其中：
- `push_mask`: push 时新槽位为 1，其他右移
- `pop_mask`: pop 时移除栈顶，底部变为 0
- 最终 `new_mask = (masks * action_weights).sum(dim=3)`

### 2.4 全局读取

#### 论文公式 (Eq. 4)

$$R_t = \text{Softmax}(W_g \cdot (\text{St}_t \otimes M_t)) \cdot \text{St}_t$$

最终输出带残差连接：
$$\mathbf{h}'_t = g_h \cdot \mathbf{h}_t + R_t$$

**为什么用全局读取而非标准 peek？**（论文 Section 3.1 & Appendix B.4）
> 标准栈只返回栈顶元素，会导致训练不稳定。全局读取通过可学习的 query-over-stack 注意力收集信息，稳定训练并增强表达能力。

#### 代码实现

**文件**: `olmo/stack_memory.py:52-58` (gate 投影)

```python
# Gate 投影：用于计算对栈的注意力分数
# 对应论文: W_g
self.gate_proj = nn.Linear(
    self.stack_dim,  # 压缩后的维度
    1,
    bias=config.include_bias,
    device=config.init_device
)
```

**文件**: `olmo/stack_memory.py:79-81` (残差权重)

```python
# 残差连接权重
# 对应论文: g_h
self.res_weight = nn.Parameter(
    torch.ones(1, device=config.init_device)
)
```

**文件**: `olmo/stack_memory.py:216-233` (全局读取)

```python
def forward(self, hidden_states, memory_stack, memory_mask):
    # ... 更新栈 ...
    
    # 计算对栈的注意力分数
    # 对应论文: Softmax(W_g · (St_t ⊗ M_t))
    gate_scores = self.gate_proj(new_stack).squeeze(-1)  # [batch, seq, heads, slots]
    gate_weights = F.softmax(
        gate_scores + (1 - new_mask) * -1e9,  # 掩蔽空槽位
        dim=-1
    )
    
    # 加权聚合栈内容
    # 对应论文: R_t = attention_weights · St_t
    memory_output = (new_stack * gate_weights.unsqueeze(-1)).sum(dim=3)
    
    # 解压缩（如果启用了压缩）
    if self.use_compression:
        memory_output = self.up_proj(memory_output)
        
    memory_output = memory_output.view(batch_size, seq_len, -1)
    
    # 残差连接
    # 对应论文: h'_t = g_h · h_t + R_t
    output = memory_output * self.res_weight + hidden_states
    
    return output, new_stack, new_mask
```

#### 代码与公式对照

| 论文符号 | 代码变量 | 说明 |
|---------|---------|------|
| $W_g$ | `gate_proj.weight` | 注意力查询向量 |
| $\text{St}_t \otimes M_t$ | `new_stack` (已被 mask 处理) | 激活的栈元素 |
| $R_t$ | `memory_output` | 读取结果 |
| $g_h$ | `res_weight` | 可学习的残差权重 |

### 2.5 多头低秩栈

#### 论文描述 (Section 3.2)

> 将隐状态 $\mathbf{h}_t \in \mathbb{R}^d$ 下投影并拆分为 $H$ 个子空间：
> $$[\mathbf{h}_t^{(1)}, \mathbf{h}_t^{(2)}, \ldots, \mathbf{h}_t^{(H)}] = \text{Reshape}(W_{\text{down}} \cdot \mathbf{h}_t)$$
> 
> 每个头独立维护栈，最后上投影并拼接：
> $$\mathbf{h}'_t = g_h \cdot \mathbf{h}_t + W_{\text{up}} \cdot \text{Concat}(R_t^{(1)}; \cdots; R_t^{(H)})$$

#### 代码实现

**文件**: `olmo/stack_memory.py:61-76` (压缩层)

```python
# 检查是否启用压缩
self.use_compression = (self.stack_dim != self.head_dim)

# 压缩层（如果启用）
if self.use_compression:
    # 下投影：d_model/H → stack_dim
    # 对应论文: W_down
    self.down_proj = nn.Linear(
        self.head_dim, 
        self.stack_dim,
        bias=config.include_bias,
        device=config.init_device
    )
    # 上投影：stack_dim → d_model/H
    # 对应论文: W_up
    self.up_proj = nn.Linear(
        self.stack_dim,
        self.head_dim, 
        bias=config.include_bias,
        device=config.init_device
    )
```

**文件**: `olmo/stack_memory.py:206-211` (应用压缩)

```python
# 准备 key values
k_values = hidden_states.view(batch_size, seq_len, self.num_mem_heads, self.head_dim)

# 应用下投影压缩
if self.use_compression:
    k_values = self.down_proj(k_values)  # 从 head_dim 压缩到 stack_dim
```

### 2.6 Block 集成

#### 论文描述 (Section 3)

> StackTrans 将栈操作集成在 Transformer 层之间，不修改注意力机制本身。

```
标准 Transformer: Input → Attention → Add → FFN → Add → Output
StackTrans:       Input → Attention → Add → [Stack Memory] → FFN → Add → Output
```

#### 代码实现

**文件**: `olmo/model.py:546-550` (Block 初始化)

```python
class OLMoBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        # ...
        
        # Stack memory 初始化
        if config.use_stack_memory:
            self.stack_memory = StackMemory(config)
        else:
            self.stack_memory = None
```

**文件**: `olmo/model.py:581-590` (应用栈记忆)

```python
def _apply_stack_memory(
    self, 
    x: torch.Tensor, 
    memory_stack: Optional[torch.Tensor], 
    memory_mask: Optional[torch.Tensor]
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Apply stack memory operations if enabled."""
    if self.stack_memory is not None and memory_stack is not None and memory_mask is not None:
        return self.stack_memory(x, memory_stack, memory_mask)
    return x, memory_stack, memory_mask
```

**文件**: `olmo/model.py:893-896` (在 forward 中调用)

```python
def forward(self, x, ...):
    # ... attention 计算 ...
    x = x + self.dropout(att)
    
    # 在 attention 后、FFN 前应用栈记忆
    # 对应论文图 3 的位置
    x, memory_stack, memory_mask = self._apply_stack_memory(x, memory_stack, memory_mask)
    
    # ... FFN 计算 ...
```

#### Diff 对比

```diff
# 标准 Transformer Block forward
  x = x + self.dropout(att)
+ # StackTrans: 在 attention 后插入栈操作
+ x, memory_stack, memory_mask = self._apply_stack_memory(x, memory_stack, memory_mask)
  og_x = x
  x = self.ff_norm(x)
  x = self.ff_proj(x)
```

---

## 3. 配置参数

### FANformer 参数

**文件**: `olmo/config.py:477-482`

```python
@dataclass
class ModelConfig(BaseConfig):
    # ...
    
    # FANformer 开关
    use_ATF: Optional[bool] = False
    
    # 周期性建模比例（默认 25%）
    p_ratio: float = 0.25
    
    # 注意力激活函数（FANformer 中设为 None，即 identity）
    attention_activation: Optional[str] = None
```

### StackTrans 参数

**文件**: `olmo/config.py:484-515`

```python
@dataclass
class ModelConfig(BaseConfig):
    # ...
    
    # StackTrans 开关
    use_stack_memory: bool = False
    """Whether to enable stack memory mechanism."""

    # 栈头数（论文中 H=4 或 8 效果好）
    num_mem_heads: int = 4  
    """Number of memory attention heads for stack operations."""

    # 栈槽位数（论文中 S=24）
    stack_slots: int = 16
    """Number of slots in the memory stack."""

    # 缓存大小
    memory_cache_size: int = 2048
    """Size of the memory cache for efficient operations."""

    # 栈记忆维度（默认 d_model // num_mem_heads）
    stack_memory_dim: Optional[int] = None
    """Dimension of stack memory. If None, uses d_model // num_mem_heads."""

    # 压缩后的维度（用于低秩机制）
    stack_dim: Optional[int] = None
    """Compressed dimension for stack memory. If None, uses head_dim (no compression)."""
```

### 参数对照表

| 论文符号/概念 | 配置参数 | 默认值 | 说明 |
|-------------|---------|-------|------|
| **FANformer** | | | |
| $p$ ratio | `p_ratio` | 0.25 | 周期性编码占比 |
| ATF 开关 | `use_ATF` | False | 是否启用 ATF |
| **StackTrans** | | | |
| $H$ | `num_mem_heads` | 4 | 多头栈数量 |
| $S$ | `stack_slots` | 16 | 栈深度 |
| $d_s$ | `stack_dim` | None | 低秩压缩维度 |
| 栈开关 | `use_stack_memory` | False | 是否启用栈 |

---

## 4. 架构对比图

### 标准 Transformer

```
Input
  │
  ▼
┌──────────────────────────────────────┐
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  Linear(QKV)  ──────────────────┐    │
│      │                          │    │
│      ▼                          │    │
│  Attention                      │    │
│      │                          │    │
│      ▼                          │    │
│  Residual Add  ←────────────────┘    │
│      │                               │
│      ▼                               │
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  FFN                                 │
│      │                               │
│      ▼                               │
│  Residual Add                        │
└──────────────────────────────────────┘
  │
  ▼
Output
```

### FANformer

```
Input
  │
  ▼
┌──────────────────────────────────────┐
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  ┌────────────────────────────┐      │
│  │  FANLayer                  │      │
│  │  X_F = [cos(Wp·X) ║        │      │
│  │        sin(Wp·X) ║         │      │
│  │        Wp̄·X + Bp̄]         │      │
│  └────────────────────────────┘      │
│      │                               │
│      ▼                               │
│  Linear(QKV)  ──────────────────┐    │
│      │                          │    │
│      ▼                          │    │
│  Attention                      │    │
│      │                          │    │
│      ▼                          │    │
│  Residual Add  ←────────────────┘    │
│      │                               │
│      ▼                               │
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  FFN                                 │
│      │                               │
│      ▼                               │
│  Residual Add                        │
└──────────────────────────────────────┘
  │
  ▼
Output
```

### StackTrans

```
Input
  │
  ▼
┌──────────────────────────────────────┐
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  Linear(QKV)  ──────────────────┐    │
│      │                          │    │
│      ▼                          │    │
│  Attention                      │    │
│      │                          │    │
│      ▼                          │    │
│  Residual Add  ←────────────────┘    │
│      │                               │
│      ▼                               │
│  ┌────────────────────────────┐      │
│  │  Stack Memory              │      │
│  │  1. 预测 push/pop/noop     │      │
│  │  2. 软更新栈状态           │      │
│  │  3. 全局读取 (attention)   │      │
│  │  4. 残差连接               │      │
│  └────────────────────────────┘      │
│      │                               │
│      ▼                               │
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  FFN                                 │
│      │                               │
│      ▼                               │
│  Residual Add                        │
└──────────────────────────────────────┘
  │
  ▼
Output
```

### FANformer + StackTrans（可组合）

```
Input
  │
  ▼
┌──────────────────────────────────────┐
│  LayerNorm                           │
│      │                               │
│      ▼                               │
│  ┌────────────────────────────┐      │
│  │  FANLayer (周期性编码)     │ ◄── FANformer 增强  
│  └────────────────────────────┘      │
│      │                               │
│      ▼                               │
│  Linear(QKV) → Attention → Add       │
│      │                               │
│      ▼                               │
│  ┌────────────────────────────┐      │
│  │  Stack Memory (栈操作)     │ ◄── StackTrans 增强
│  └────────────────────────────┘      │
│      │                               │
│      ▼                               │
│  LayerNorm → FFN → Add               │
└──────────────────────────────────────┘
  │
  ▼
Output
```

---

## 附录：关键代码文件索引

| 功能 | 文件 | 行号范围 |
|-----|------|---------|
| FANLayer 定义 | `olmo/model.py` | 78-128 |
| FAN (ATF) 模块 | `olmo/model.py` | 130-138 |
| OLMoSequentialBlock (含 ATF) | `olmo/model.py` | 762-925 |
| OLMoLlamaBlock (含 ATF) | `olmo/model.py` | 928-1092 |
| StackMemory 模块 | `olmo/stack_memory.py` | 18-355 |
| 向量化栈更新 | `olmo/stack_memory.py` | 128-176 |
| 全局读取 | `olmo/stack_memory.py` | 178-239 |
| 配置参数 | `olmo/config.py` | 477-515 |
| OLMo 模型主类 | `olmo/model.py` | 1195-1916 |

---

*报告生成日期: 2026-01-14*
