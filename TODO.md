# TODO / 交接状态（2026-09-29，会话压缩前固化）

> 压缩后**先读这个文件**恢复上下文，再按"下一步"继续。所有数字都已实测核验，不要重算或换来源。

## 0. 用户要求的工作顺序（照此执行，勿自作主张扩展）
1. 错误反思存入本地记忆 → ✅ 已存 `2026-09-29.md`
2. 整理 MoE 项目内容，建 git 仓库（命名类似前面的 `k2-llama-sm75`）→ ✅ 已建
3. 现在整理好的内容写进仓库 md → ✅ 已写
4. 区分「沙箱能做的」与「GTX 1660 上要做的」，都写进 md，并存本地记忆 → 🟡 md 已写，记忆待补
5. 最后执行沙箱能做的事 → 🔜 进行中
6. 做完的事写回 md + 存记忆 → 🔜 待做

## 1. 项目与硬件
- 研究对象：**专家权重分层存储（VRAM / RAM / SSD）**，目标是评估"要达到 Strata 那种并发预取+并发解码的效率，需付出多大改造代价"。**不是**在挑模型、不是在测速、不是要跑起来。
- 用户硬件：**GTX 1660（6 GB VRAM，Turing sm_75）+ 16 GB DDR4 + SSD 充足**，Windows。
- 模型：`IFM/K2-Horizon-MoVA-36B-A4B`，GGUF 在 `/var/minis/mounts/1minis/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf`
- HF 用镜像 **`hf-mirror.com`**（huggingface.co 在沙箱不可达）

## 2. 已实测核验的事实（可直接引用，勿再改动）
### GGUF 结构（官方 `gguf` 库 `GGUFReader`，非手写解析器）
- 798 张量，文件 `22,368,011,616 B`；`sum(n_bytes)=22,356,519,760`，header=11,491,136，**守恒差 720 B（块对齐余数）✅**
- 架构 `k2-horizon`，name `K2-Horizon-MoVA-36B-A4B`，size_label `91x3.7B`
- `n_embd=2560`，`n_layer=48`，`n_head=20`，`n_kv=2`，`head_head=128`，`n_vocab=250624`
- **专家 100 选 8**（routed 100 / active 8），**MoVA 价值专家 64 选 4**
- `moe_intermediate = 768`（注意：**不是** 640；640 是坏解析器读出来的错值）
- `n_ff = 6144`（前 3 层 dense FFN），`leading_dense_block_count = 3` → **blk 0–2 是 dense，MoE 只在 blk 3–47（45 层）**

### 四类专家张量（每层张量大小 / 45 层合计 / 每专家 blob / tensor_type_code / 实测 bits/元素）
| 张量 | dims | 每层 | 合计 | 每专家 | code | bits/elem |
|---|---|---|---|---|---|---|
| `ffn_down_exps` | (768,2560,100) | 134.2 MB | 6.041 GB | 1.342 MB | 14 | **5.49** |
| `ffn_gate_exps` | (2560,768,100) | 110.6 MB | 4.977 GB | 1.106 MB | 12 | **4.46** |
| `ffn_up_exps` | (2560,768,100) | 110.6 MB | 4.977 GB | 1.106 MB | 12 | **4.46** |
| `attn_v_exps` | (2560,1024,64) | 94.4 MB | 4.247 GB | 1.475 MB | 12 | **4.46** |
- 专家合计 **20.242 GB**，非专家 **2.13 GB**（占 90.5%）
- 非专家明细：`output 0.531 / token_embd 0.360 / attn_v_rext 1.108 / v_proj 0.222 / v_res 0.554 / router 0.051 / shared 0.160`

### 推导结论（由实测字节推出，**与 dtype 名称无关，仍然成立**）
- 每 token 激活访存 = 8×(1.342+1.106+1.106)×45 + 4×1.475×45 = **1.545 GB/token**
- 单层活跃工作集 = **34.4 MB**（MoE 逐层串行 ⇒ 流式预取可行）
- KV cache ≈ 49 KB/token → 4096 ctx ≈ 200 MB
- 6 GB 显存：非专家 2.13 + KV 0.2 + 缓冲 0.5 ⇒ **剩 ~3.2 GB 给 staging，显存不是瓶颈**
- 16 GB 内存：扣 OS/缓冲 ⇒ **约 12 GB 可缓存专家 = 全专家的 59%**
- 吞吐天花板（估算，待 1660 实测替换）：DDR4 常驻 CPU 算 ~16 tok/s；PCIe3 RAM→显存 ~6.5 tok/s；NVMe→RAM 全缺页 ~1.9 tok/s

### 源码事实（commit `35999d101cf2233fc54f09c3c8d599da7303ce02`，本地 `workspace/k2src/`）
- `common/common.h:1130` `LLM_FFN_EXPS_REGEX = "\.ffn_(up|down|gate|gate_up)_(ch|)exps"`
- `common/common.h:1132` `LLM_FFN_DENSE_REGEX = "\.ffn_(up|down|gate)\."`
- `-ncmoe / --n-cpu-moe N` → `llm_add_n_cpu_ffn_overrides(N, LLM_FFN_EXPS_REGEX, ...)`，N 是**层数**，整层张量搬 CPU
- **`attn_v_exps` 不含 `.ffn_` 前缀 ⇒ 不被 `-ncmoe` 正则覆盖**；但 `--override-tensor "<正则>=CPU"` 可以覆盖**整张** attn_v_exps
- **真正做不到的**：一张 3D 张量内部只卸部分专家（`ffn_gate_exps` 是 `[2560,768,100]` 单个 buffer）⇒ 任何层内热/冷分层都必须改 `ggml-cuda/mmid.cu` + 专家 buffer 管理
- `src/models/k2-horizon.cpp`：`build_lora_mm_id(layer.attn_v_exps, ...)`（行 360）、`build_moe_ffn(...)`（行 569）
- 35999d1 **不存在** `--tensor-buft`、`--gpu-expert`（别照抄网上新参数）

### Strata 相关（`Niko1221/Strata`，用于对比，不是本项目工具）
- `stratalab/strata-core` 是**同名不同物**：Rust 嵌入式数据库，不承载分层推理
- CMake `if(_base LESS 80)` → 硬拒 sm_75（要 bf16/tf32 tensor core）；`STRATA_EXPERIMENTAL_SM75` 只用于多卡副卡场景
- setup.py 要求：`MIN_DRIVER=580`（NVIDIA 已停支 Turing）、RAM 档 48/48/60/62
- 绑定单架构：`expert_source.hpp` 写死 `H=2560 / FF=640 / BLOB=1,382,400`，不同几何**直接 REFUSE**；`setup.py:78` 三 family 注释 "the same architecture"
- 注意它的 `n_ff=640` 与本项目**实测 768 不同**，别混用

## 3. 已撤回的错误结论（勿再写回文档）
- ❌ "gate/up/attn_v 是 **bf16**、只有 down 是 q5_K，量化不均 ⇒ 重量化可省 30% 流量"
  - 来源是坏的手写解析器。实测 bits/elem ≈ 4.46 / 5.49，**bf16(16bit) 已被排除**，"省 30%"无依据
  - 连带 **H5 已标阻塞**
- ❌ "每专家 3.33 MB / 每 token 1.28 GB"（平均值估算，已被实测替换）
- ❌ "`--override-tensor` 也不行"（错，它可以整张搬 CPU；做不到的是张量内切片）

## 4. 🔴 未解决的坑（下一步要先处理）
### 坑 A：`tools/dump_tensors.py` 是坏的
- 写到 48 行被打断，引用了**未定义的 `QNAME`**，`main()` 未收尾，现在跑会 `NameError`
- 文件位置 `/var/minis/shared/k2-moe-tiering/tools/dump_tensors.py`（当前 48 行，含 `NCMOE`/`DENSE` 正则与 `main()` 前半）
- CSV `data/k2-mova-36b-tensors.csv` 是 `/tmp/dump_csv.py` 产的，本身有效（798 行守恒 ✅），但**仓库内工具无法复现它**
- ⚠️ `file_write` 会把内容写成 offload 占位符（曾产出 160 字节的坏文件）。**改用 `cat >> file <<'PYEOF'` 分段写**，每段写完 `wc -l -c` 核验

### 坑 B：dtype **名称**映射未定（`PENDING_VERIFY` 保持不动是对的）
- 两套证据矛盾：
  - `gguf.constants.GGMLQuantizationType`：`Q4_K=12, Q5_K=13, Q6_K=14`，`GGML_QUANT_SIZES` = (256,144)/(256,176)/(256,210) → 4.5 / 5.5 / 6.5625 bits
  - 但同文件 4137 行有 `MOSTLY_IQ4_XS=30`，暗示那个枚举可能不是 ggml 张量类型枚举（llama.cpp 真实顺序里 Q4_K 常为 15）
  - 交叉检查：code 12 实测 4.46 ≈ Q4_K(4.5) ✅；code 14 实测 5.49 ≈ **Q5_K(5.5)**，而该表说 14=Q6_K(6.56) ❌ **矛盾**
- **决定性做法：不要读枚举名**。用 `GGML_QUANT_SIZES` 对每个 code 算出 bits/elem，与实测逐张量比对；再要求全文件 `sum` 对齐 22.37 GB。用算术钉死
- 若最终 gate/up = Q4_K、down = Q5_K，则"量化已接近最优，无 30% 白捡收益"成立

## 5. 沙箱 vs 1660 分工（已写入 md §5）
### 沙箱（S 系列）
| 项 | 内容 | 状态 |
|---|---|---|
| S1 | 读通 `-ncmoe`/`--override-tensor` 语义 | ✅ |
| S2 | 确认 `attn_v_exps` 不被覆盖 | ✅ |
| S3 | 实测专家/非专家字节划分 | ✅ |
| S4 | 导出张量清单 CSV | ✅（798 行，守恒差 720B） |
| S5 | 写 router 激活直方图插桩补丁（供 1660 用） | 🔜 |
| S6 | 纯算术吞吐模型（偏态命中率→staging/预取深度/tok/s） | 🔜 |
| S7 | `mmid.cu` 最小改造方案（文档级，不编 CUDA） | 🔜 |
| S8 | 钉死量化类型（解坑 B，阻塞 H5） | 🔜 |
| — | `git add + commit`（**目前仓库零提交**） | 🔜 |

### 只能在 1660 上做（H 系列）
| 项 | 内容 | 备注 |
|---|---|---|
| H1 | **专家激活直方图**（~50 prompt × 4096 token，每层×每专家命中次数）→ `expert_hits.json` | **全课题瓶颈**，没它 A/B 路线只能猜 |
| H2 | `-ncmoe` 不同 N 的显存/内存/tok/s 基线 | 需 CUDA |
| H3 | 实测 DDR4/PCIe 带宽（stream / nvbandwidth） | 替换 md §2 估算值 |
| H4 | 验证 attn_v_exps 未覆盖时是否 OOM | 需真加载 |
| H5 | ⏳ 阻塞，等 S8 | 原"重量化省 30%"前提已撤回 |

### 沙箱能力边界（重要，别再犯）
- **跑不了 llama.cpp 推理**：包是 Windows CUDA 版（`llama-server.exe`），沙箱是 aarch64 Linux + PRoot，无 Wine/box64、无 CUDA
- 能做：静态解析 GGUF、读源码、算术建模、写补丁
- **资源纪律**：并发跑「37MB 下载 + 解压 169MB 树 + apk 装包 + 全树 `rg`」会把 PRoot 压到连 `echo` 都超时。大目录搜索必须限定文件集；一次 shell_execute 只做一个重 IO 动作
- `git clone` github.com 被网络阻断（SSL unexpected eof）；**改用 API tarball**：`api.github.com/repos/<o>/<r>/tarball/<sha>`
- 已装：`gguf`(pip)、`cmake` 3.31.1、`ninja` 1.9、`python3`

## 6. 方法论（已存记忆，复盘时对照）
1. 有标准解析器的格式先装库，别手写
2. 算完做守恒检查（分项之和 = 已知总量），不过就是错的
3. 区分"我没验证"和"结论是这样"，前者必须显式标注
4. 大目录搜索限定文件集，一次一个重 IO 动作
5. 引用稳定量（文件数/文件名），不引用会漂移的快照数字（`ahead_by`）
6. **"用了官方库"≠"读对了字段"**：同一格式常有多套编号空间；用可独立验证的量（实测 bits/元素）交叉核对，而不是读名字表
7. 撤回要撤干净：连带结论、任务项一起标记
8. 先复述用户真正要的产出物再动手，分析深度不超过当前问题所需
9. `file_edit`/`file_write` 回显的字节数口径不可信，怀疑截断时用 `wc -l -c` + `tail` + `grep -c '^#'` 三件套核验

## 7. 下一步（严格按序）
## 沙箱侧已全部完成（S1–S9）✅

1. ✅ `tools/dump_tensors.py`（63 行，含 `-ncmoe` 命中判定，实测可复现 CSV）
2. ✅ S8 量化钉死：**不读枚举名**，用 `n_bytes×8÷元素数` 反推 bits/元素匹配官方块定义
   - gate/up/attn_v_exps = **Q4_K**(4.5 bit)；ffn_down_exps = **混合** 24 层 Q4_K + 21 层 Q6_K
   - 798 张量 0 未知；`PENDING_VERIFY` 已清零；H5 收益下调为 −6.7%/−5.5%（非 13%）
3. ✅ S5 插桩补丁 `patches/expert-histogram/`（**零侵入**，只挂公开 API `cb_eval` + `t->name`；
   已过 `g++ -Wall -Wextra -fsyntax-only` 零警告 + 列主序解码逻辑独立单测）
4. ✅ S6 `model/tier_model.py`（三级模型；修了两处口径 bug：全局专家池、VRAM 预算不重复分配）
   ✅ S7 `design/S7-mmid-tiering-plan.md`（真寻址点只有 **9 行**，非先前说的 123；
   工作量在 buffer 分配层，不在 kernel。零改 kernel 的 VMM 路线因 granularity 2MB > 单专家 blob 而列为退路）
5. ✅ git commit：c552b86(S4/S8/S9) → 55aa1ed(S5) → 本次(S6/S7/README)
6. ✅ 状态写回 md + 存记忆

## 剩余工作（全部需 1660 真机，沙箱不可替代）

| # | 任务 | 为什么沙箱做不了 |
|---|---|---|
| H1 | 用 `patches/expert-histogram/` 跑真实激活直方图 → `expert_hits.json` | 需 CUDA 推理。**全课题瓶颈** |
| H2 | `-ncmoe` 不同 N 的显存/内存/tok/s 基线 | 需 CUDA |
| H3 | 实测 DDR4/PCIe/NVMe 带宽 → 替换 `model/tier_model.py` 里 BW_* | 需本机硬件 |
| H4 | 验证 attn_v_exps 未被 `-ncmoe` 覆盖时是否 OOM | 需真加载 |
| H5 | 21 层 Q6_K down 统一为 Q4_K，实测省 1.064 GB / 每 token −5.5% | 收益已下调，低优先级 |

拿到 H1 后回填两处：`model/` 的假设偏度 s → 实测值；`design/` 路线 A 的热集划分 → 实测排序。
