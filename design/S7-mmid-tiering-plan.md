# S7 — 层内专家分层的 llama.cpp 改造方案（文档级，不在沙箱编译 CUDA）

目标：让**单张 3D 专家张量内部**做到"一部分专家在 VRAM、一部分在 RAM、一部分在 SSD"，
即 §3 里证明现有 `-ncmoe` / `--override-tensor` 做不到的**层内热冷分层**。
这是 Strata 的核心能力，也是本课题"要多大改造代价"的回答对象。

## 0. 事实基础（全部源码实测，commit 35999d1，行号可复核）

| 事实 | 出处 | 对方案的意义 |
|---|---|---|
| kernel 用 `x = (const char*)src0 指针 + channel_x*stride_channel_x` 定位专家权重 | mmq.cuh:1052/1146/1230、mmvq.cu:656/808、mmf.cuh:113/365、mmvf.cu:47/84 | **真寻址点只有 9 行**（不是 123，那是签名传参）；专家基址是单一裸指针 |
| `stride_channel_x` 由 **host 传入**，值 = `src0->nb[2]/ts`（mmf.cu:39,109） | kernel 不重算、不假设它等于紧凑布局 | 只要专家仍在**一段连续 VA**、间隔固定，kernel 无需知道它们物理上分散 |
| ggml-cuda **已在用 CUDA VMM**：`cuMemCreate/cuMemAddressReserve/cuMemMap/cuMemSetAccess/cuMemUnmap` | ggml-cuda.cu:590-599,642-649 | "虚拟地址连续、物理页可换" 的底层能力**现成**，不是从零造 |
| VMM granularity 取 `CU_MEM_ALLOC_GRANULARITY_RECOMMENDED` | ggml-cuda.cu:288 (`vmm_granularity`) | ⚠️ 通常 2 MB；决定换入换出**最小粒度**，是关键约束（见 §1 R-B） |
| decode 路径阈值 `MMVQ_MAX_BATCH_SIZE=8` | mmvq.cuh:3 | 逐 token 生成(ne2≤8)走 **mmvq.cu**（GEMV），分层主战场在 decode，故 mmvq 是必改点 |
| MUL_MAT_ID 会被固化进 CUDA Graph，仅在 fallback 同步路径才禁用 | ggml-cuda.cu:1873-1890,2560-2561 | ⚠️ 分层若改变"哪些专家在哪个设备指针"，必须保证 graph 复用时指针稳定 |
| `-ncmoe` 正则 `\.ffn_(up|down|gate|gate_up)_(ch|)exps` **不含 attn_v_exps** | common.h:1130 | MoVA 的 4.25 GB attn_v 专家现有参数完全碰不到；分层需单独覆盖 |
| `--override-tensor` 可整张搬 CPU，但做不到**张量内只卸部分专家** | 源码级（§3） | 本方案的立身之本：就是要张量内按专家切 |

## 1. 两条可行路线（实现层，编号 R-A/R-B）

> ⚠️ **编号约定**：本文的 R-A/R-B 是**实现层**路线（用什么机制寻址/换页）；
> `MOE-TIERING.md` §4 的 **P1/P2/P3** 是**策略层**路线（哪些专家放哪、何时搬）。两者正交：
> P1/P2 都落在 R-A 上（单专家粒度是硬需求），只有放弃单专家粒度才可能用 R-B。

### R-A：per-expert 指针/偏移表（改 9 行 kernel + 传参）
把 `channel_x*stride_channel_x` 换成 `expert_off[channel_x]`（device 上的 `int64` 偏移表，
或 `void*` 指针表）。专家权重不再要求单一连续 buffer，而是每个专家独立分配，可各自在 VRAM/RAM。

- **改动面**：mmvq.cu(×2)、mmq.cuh(×3)、mmf.cuh(×2)、mmvf.cu(×2) 的 `offset_x`/`kbx_offset` 行；
  加一张 device 端 `expert_off[]` 表；host 侧在 4 处 launch 前上传该表（`ggml_cuda_launch_*` / mmf.cu:109）。
- **优点**：粒度最细（单专家任意换入换出），不受 VMM granularity 约束，热集划分自由（直接吃 H1 直方图）。
- **风险**：
  1. **L2/预取**：连续 buffer 变 gather，可能掉带宽。但专家 GEMV 本就是"每专家读一整块、块内连续"，
     间接层只多一次查表（`expert_off[c]`，被 warp 广播，命中 L1），预计影响小 —— **需 1660 实测**。
  2. **CUDA Graph**：graph 记录的是 kernel 入参（含 `expert_off` 指针）。只要表的**地址**不变、
     内容可就地更新，replay 仍安全 → 比现状更灵活（现状整张指针写死）。这点必须在实现时守住。

### R-B：VMM 虚拟地址重映射（kernel 不动，但受 granularity 约束）
保持 kernel 完全不动（仍 `x + channel_x*stride`）。给整张专家张量 **reserve 一段连续 VA**，
stride 用真实布局；物理页按专家（或专家组）用 `cuMemMap/cuMemUnmap` 换入换出：
- 常驻热专家 → 物理页保持在 VRAM（映射不动）。
- 冷专家 → 需要时 `cuMemMap` 到预分配 VRAM 物理页，用完 `cuMemUnmap`。
- 计算逻辑零改动，纯靠 host 侧管理映射。

- **优点**：**不碰 CUDA kernel**，可移植性最好，风险集中在 host 管理逻辑。
- **致命约束**：`cuMemMap` 的 offset/size 必须是 **granularity（~2 MB）整数倍**。
  而每专家 blob 是 1.1 MB(Q4 down/gate/up)、1.6 MB(Q6 down)、1.47 MB(attn_v)——
  **都小于 2 MB 且非其约数**。⇒ 无法"以单专家为粒度"map/unmap。
- **补救**（把 B 做成立的代价）：以 granularity 为单位**成组**管理，即
  `每 2MB 页塞 floor(2MB/blob) 个专家`、整组一起换。gate/up 1.1MB → 一页 1 个专家(1.82MB 有效)，
  浪费 18%；down Q6 1.6MB → 一页 1 个(浪费 25%)。换来的是"分组粒度"的分层，热集划分精度下降。
  或者 pad 专家到 2MB 对齐（浪费 ~45%，不可接受）。

## 2. 推荐：**R-A 为主线，R-B 作对照**

理由：
- 本课题要"层内按专家分层"，**单专家粒度是硬需求**（H1 直方图给出的热集是按专家排序的）。
  R-B 的 2MB 分组粒度会把相邻专家绑死，削弱热集选择质量，且 padding 浪费 VRAM——而 VRAM 恰恰是 6GB 紧约束。
- R-A 的改动面**可控且集中**：9 行寻址 + 一张偏移表 + 4 处 host 上传。不是"重写 mmid.cu"，
  而是把"乘法"改成"查表"。真正的工作量在 **buffer 分配层**（让每张专家张量按专家独立分配并可跨设备），
  不在 kernel。
- R-B 的价值在于证明"kernel 层面理论上可以不动"。**但注意它守不住硬需求**：
  2 MB 成组换页会把相邻专家绑死，只能做"专家组粒度"分层；若 H1 显示热集在层内高度集中、
  组内命中率差异大，R-B 会明显拖 P1 的命中质量。所以 R-B 只能当**退路**（R-A 的 gather 实测掉带宽严重时），
  不能当默认方案 —— 这也修正了 `MOE-TIERING.md` §4 早期版本"有一条路线不改 kernel 所以最省力"的说法。

## 3. R-A 的实现要点（给 1660 那台机器的改造清单）

1. **buffer 层**（`ggml-cuda.cu` 分配处 + `-ncmoe`/`--override-tensor` 之外的新路径）：
   让 `blk.N.ffn_*_exps` 与 `blk.N.attn_v_exps` 的每个 expert slice 成为可独立寻址单元；
   维护每层 `expert_off_gpu[n_experts]`（在 VRAM 的专家给 VA 偏移，RAM 的给 -1/哨兵）。
2. **缺页路径**：token 选中的专家若 `expert_off==-1`（不在 VRAM）→
   从 RAM staging 用 `ggml_backend_tensor_copy_async`(backend.h:116) 预取到 VRAM，更新表项。
   预取深度按 S6：单层工作集 34–36 MB，PCIe ~3–4 ms，与上一层计算重叠隐藏。
3. **kernel 层**：9 个寻址点把 `channel_x*stride_channel_x` → `expert_off[channel_x]`。
   表放 device，warp 广播命中 L1。
4. **CUDA Graph**：graph 内只固化 `expert_off` **指针**（不固定内容）；每步在 host 更新表内容后
   `cudaGraphExecKernelNodeSetParams` 或直接走"表在 device 常驻、kernel 现读"的方式，避免 replay 读到旧布局。
5. **覆盖 attn_v_exps**：现有 `-ncmoe` 正则不含它；分层管理必须显式把 value 专家纳入
   `expert_off`（64 个/层，HKN=1 确认值）。

## 4. 验证里程碑（1660 独占，对应 TODO 的 H 系列）

| 步 | 验证 | 判据 |
|---|---|---|
| V1 | 先只做 buffer 层 + 偏移表，kernel 用"全常驻"退化路径 | 与现有 `-ncmoe` **数值完全一致**（先证明没算错） |
| V2 | 打开 gather 寻址，全专家常驻 VRAM（小模型或单几层） | 相比 V1 的带宽损失 < ?%（决定 R-A 是否成立） |
| V3 | 接入 H1 直方图，钉热集到 VRAM、冷集到 RAM | 实测 tok/s 落在 S6 模型预测区间内 |
| V4 | 加预取重叠 | 缺页延迟被隐藏的程度 |

## 5. 与 Strata 的差异（回答"改造代价"）

Strata 用一张 28.8 GB 的大表 + 专用调度做并发预取/解码。本方案：**不需要那张大表**
——因为专家 blob 已在 GGUF 里连续可得，只要 9 行寻址改成查表 + buffer 分片管理。
代价集中在**buffer 分配层的专家切片管理**与**缺页预取调度**，kernel 改动极小。
这正是"要多大改造才能达到那种效率"的核心答案：**改的是内存管理，不是算力/kernel**。
