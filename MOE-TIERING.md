# K2-Horizon MoVA-36B-A4B 专家分层存储研究

研究对象：在 **GTX 1660（6 GB VRAM）+ 16 GB DDR4 + NVMe** 上，把 MoE 专家权重分层放到显存 / 内存 / SSD，并做到接近 Strata 的"并发预取 + 并发 decode"。

配套仓库：
- [`womenlialtd/k2-llama-sm75`](https://github.com/womenlialtd/k2-llama-sm75) — 我们自建的 sm_75 构建（含 K2-Horizon 支持 + Unicode 补丁）
- [`Niko1221/Strata`](https://github.com/Niko1221/Strata) — 参考实现，**不可直接复用**（原因见下）
- `stratalab/strata-core` — 同名但是另一个项目（Rust 嵌入式数据库），与本课题无关

---

## 1. 模型事实（全部实测，非估算）

数据来源：`/mnt/1minis/K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf`（22.37 GB，798 张量）
工具：`gguf` 官方库 + 按 tensor offset 逐字节差分

| 参数 | 值 |
|---|---|
| `general.architecture` | `k2-horizon` |
| `hidden_size` | 2560 |
| `num_hidden_layers` | 48（**前 3 层 dense**，45 层 MoE） |
| `num_experts` / `num_experts_per_tok` | 100 / 8 |
| `moe_intermediate_size` | **768** |
| `shared_expert_intermediate_size` | 768 |
| `value_expert_count` / `used`（MoVA） | **64 / 4** |
| `value_dim` | 1024 |
| `kv_lora_rank` | 512 |
| `num_key_value_heads` / `head_dim` | 2 / 128 |
| `vocab_size` | 131072 |

### 专家 vs 非专家的体量切分

| 类别 | dims (n_experts 最后) | 每层张量大小 | 45 层合计 | 每专家 blob | 量化（实测反推） |
|---|---|---|---|---|---|
| `blk.N.ffn_gate_exps` | (2560,768,100) | 110.59 MB | 4.977 GB | 1.106 MB | **Q4_K** 4.5 bit/elem |
| `blk.N.ffn_up_exps` | (2560,768,100) | 110.59 MB | 4.977 GB | 1.106 MB | **Q4_K** 4.5 bit/elem |
| `blk.N.ffn_down_exps` | (768,2560,100) | **110.59 / 161.28 MB** | 6.041 GB | 1.106 / 1.613 MB | **混合：24 层 Q4_K + 21 层 Q6_K** |
| `blk.N.attn_v_exps` | (2560,1024,64) | 94.37 MB | 4.247 GB | 1.475 MB | **Q4_K** 4.5 bit/elem |
| **专家合计** | — | — | **20.241 GB** | — | — |
| 非专家（attn / embd / output / router / shared / dense FFN） | — | — | 2.115 GB | — | 混合 |

字节数与维度均为官方 `gguf` 库实测（`n_bytes` / `shape`）。守恒校验：`20.241 + 2.115 = 22.356 GB`，`sum(n_bytes) + header = 文件 − 720 B`（720 B 为块对齐余数）✅。全量 798 张量按此法判定，**未知量化数 = 0**（见 `tools/dump_tensors.py`）。

### 量化类型的判定方法（S8 已完成）

**不读枚举名。** `gguf` Python 库与 `ggml.h` 里存在多套编号空间，直接把 code 12/14 当量化名读会出错（我曾把元数据值类型枚举当张量类型枚举，得出 `bf16` / `q5_K` 的错误标签，已撤回）。可靠做法是纯算术：

```
bits/元素 = n_bytes × 8 ÷ (d0 × d1 × d2)
```

得到的值直接对应官方块定义（`ggml-common.h`，`QK_K = 256`）：Q4_K = 144 B/块 = **4.5 bit**、Q5_K = 176 B = **5.5**、Q6_K = 210 B = **6.5625**、bf16 = **16**。实测四类专家张量分别命中 4.5 与 6.5625，无一落在 16，**故"bf16 专家"这一说法彻底不成立**。

### ⚠️ 新发现：`ffn_down_exps` 层间量化不一致

| 量化 | 层数 | blk 编号 | 每层字节 |
|---|---|---|---|
| Q4_K | 24 | 6,7,9,10,12,13,15,16,18,19,21,22,24,25,27,28,30,31,33,34,36,37,39,40 | 110.59 MB |
| Q6_K | 21 | 3,4,5,8,11,14,17,20,23,26,29,32,35,38,41,42,43,44,45,46,47 | 161.28 MB |

对分层设计的实际影响：**每层工作集不统一**，staging 池必须按**最大层**定容，否则 Q6_K 层会反复溢出。

- Q4_K 层：`8×(1.106+1.106+1.106) + 4×1.475 = 32.4 MB`
- Q6_K 层：`8×(1.106+1.106+1.613) + 4×1.475 = 36.5 MB` ← **定容按这个**

> 已撤回的旧推论"量化不均、把 gate/up 重量化可省 30% 流量"：前提（gate/up 是 bf16）错——gate/up 实测就是 Q4_K，没有优化空间。**"量化不均"方向成立但收益远小于先前所述**：不均只出现在 `ffn_down_exps` 内部（24 层 Q4_K + 21 层 Q6_K），把 21 层 Q6_K 统一成 Q4_K：
>
> | 指标 | 现状 | 统一后 | 变化 |
> |---|---|---|---|
> | FFN 专家（gate+up+down） | 15.994 GB | 14.930 GB | **−1.064 GB（−6.7%）** |
> | 专家总（含 attn_v） | 20.241 GB | 19.177 GB | −5.3% |
> | 每 token 访存 | 1.5450 GB | 1.4598 GB | **−5.5%** |
>
> ⚠️ 上下文压缩前这里写的是"降到 13.87 GB / −13% / 每 token 1.415"，**是错的**——那次估算把 45 层的 down 全当成 Q6_K 了。上表按 CSV 的 `quant` 列逐层实测重算（21 层 × 50.7 MB 差值）。收益只有原来的约一半，优先级应下调。这是现在唯一仍成立的重量化收益。

**不受影响的结论**：下面 §2 的每 token 访存量、工作集均值、吞吐天花板全部由实测字节推出，与量化名称无关，数字不变。

---

## 2. 决定设计的三个数字

### (a) 每 token 必须访问的专家字节 = **1.545 GB**

```
FFN :  8 experts × 3.554 MB × 45 层 = 1.280 GB
MoVA:  4 experts × 1.475 MB × 45 层 = 0.265 GB
                                    = 1.545 GB / token
```

**这才是真正的敌人，不是"22 GB 太大"。** 由此得到的吞吐天花板：

| 数据通路 | 有效带宽 | tok/s 上限 |
|---|---|---|
| VRAM GDDR6（装得下的话） | ~240 GB/s | — 只有 6 GB，不可能 |
| **DDR4 → CPU 直接算专家** | ~25 GB/s | **~16** |
| **DDR4 → PCIe → VRAM**（专家常驻内存） | ~10–12 GB/s (gen3 x16) | **~7** |
| **NVMe → RAM**（专家全缺页，100% miss） | ~2–3 GB/s | **~1.9** |

### (b) 但单层活跃工作集只有 **34.3 MB**

MoE 逐层串行，同一时刻只有本层 top-8(+MoVA top-4) 在被用：`8×3.554 + 4×1.475 = 34.3 MB`。
→ **流式方案不需要 1.545 GB 常驻显存**，几百 MB staging 池 + 提前一层预取即可；预取 34 MB 走 PCIe 仅 ~3 ms，可与本层计算重叠隐藏。

### (c) 16 GB 内存只能缓存专家的 ~56%

`16 − 3(OS) − 0.5(非专家常驻部分) − 0.2(KV) − 1(余量) ≈ 11.3 GB` ÷ 20.24 GB ≈ **56%**。
剩下约 9 GB 专家必须落 SSD → **缺页率不会低**，除非利用偏态做热集选择（见 §4）。

其它实测细节：
- KV cache：`2 × 2(kv_heads) × 128 × 48层 × 2B ≈ 49 KB/token`，4096 ctx ≈ **201 MB**
- 非专家仅 2.115 GB → **6 GB 显存放完非专家+KV+计算缓冲(0.5 GB)后还剩 ~3.2 GB 给 staging**，显存不是瓶颈
- router 权重（`ffn_gate_inp`）仅 **0.05 GB** → 可**全留显存**，因此每层 top-8 在 FFN 之前就算出，预取只需提前一层
- `output.weight` 0.53 GB + `token_embd` 0.36 GB 属非专家，无 expert 维度，不可分层

---

## 3. 为什么 llama.cpp 现成参数做不到（源码级结论）

在 `k2-llama-sm75` @ `35999d1` 上核对：

```c
// common/common.h:1130
const char * const LLM_FFN_EXPS_REGEX = "\\.ffn_(up|down|gate|gate_up)_(ch|)exps";
// common/common.h:1132
const char * const LLM_FFN_DENSE_REGEX = "\\.ffn_(up|down|gate)\\.";
// common/common.h:1138 — --n-cpu-moe 走的就是这条
```

三条限制：

1. **`-ncmoe/--n-cpu-moe N` 的正则不含 `attn_v_exps`**（后者无 `.ffn_` 前缀）→ 那 **4.25 GB MoVA 专家（占专家总量 21%）无法用 `-ncmoe` 卸载**。
2. 卸载粒度是**整张 3D buffer**。一层的 100 个专家打在同一个 `ffn_gate_exps` 张量里，`--override-tensor "<regex>=CPU"` 能把**整张**放 CPU（包括 `attn_v_exps`，这条我早期说错了，已纠正），但**做不到"一张里只卸 92 个、留 8 个热专家"**。
3. `-ncmoe N` 的 N 是**层数**，等于把分层退化成"整层卸载"，热冷混合的层内选择完全丢失。

计算路径：`src/models/k2-horizon.cpp` → `build_moe_ffn(...)`（FFN）与 `build_lora_mm_id(layer.attn_v_exps, ...)`（MoVA）→ CUDA 侧 `ggml-cuda/mmid.cu`。
mmq 配置只有 `mmq-config-ampere.cuh` / `mmq-config-blackwell.cuh`，**sm_75 无专用配置**。

> 结论：§4 的任何路线在 llama.cpp 上都不是调参能实现的，**必须改 `mmid.cu` + 专家 buffer 管理**。这正是"分层卸载参数"与"Strata 级定制调度"之间的距离。

### 关于 Strata 能不能复用

- **代码不能用**：`CMakeLists.txt` 硬拒 `compute capability < 8.0`；且它的几何常量 `H=2560 / FF=640 / BLOB=1,382,400` 写死、`expert_source.hpp` 明写不同几何直接 REFUSED。它的 24,576 experts / `BLOB` 布局与本模型（100 experts）不同构。
- **架构可借鉴**：`STRATA_EXPERIMENTAL_SM75` 不是给 1660 单机用的，是给"多卡里当副卡"（见 `docs/SECOND_GPU.md`）。
- **硬件不构成原则性障碍**：拒绝 sm_75 的理由是缺 bf16/tf32 tensor core，但分层场景做的是 **GEMV（1 token × 1 expert），访存受限而非算力受限**，Turing 有 fp16 HMMA + int8 dp4a，够用。
- Strata 靠 **28.8 GB 的 n-gram PLE 查找表**做预取预测，这是它 RAM 要 48–62 GB 的原因；本课题的对应问题是"16 GB 内存下用什么预测器"。

---

## 4. 三条候选路线

| 路线 | 做法 | 是否需要改代码 | 备注 |
|---|---|---|---|
| **A 热集常驻**（偏态） | MoE 激活近 Zipf；统计每层各专家命中次数，把 top-N 专家 `mlock` 进 RAM，冷的交给 mmap | **要**（同 B，因粒度限制） | 最省力，不改 kernel，只改"哪些专家常驻" |
| **B 层内 LRU + 一层预测预取** | router 全留显存 → 本层 top-8 提前可知；维护每层候选并集，用上一 token 同层选择做预测 | **要**（核心工作量在此） | Strata 路线，去掉它 28.8 GB 的大表 |
| **C GPUDirect Storage** | NVMe 直连 VRAM 绕开 DDR4 | 要，且 Windows 上基本不实用 | 1660 无此能力，**不建议起步** |

A 和 B 可以叠加：先用 A 的热集划分缩小缺页集合，再用 B 的预取隐藏剩余延迟。

**尚未验证、且决定 A/B 成败的唯一关键量：专家激活的实际偏态程度。** 目前只有结构性事实（100 选 8 / 64 选 4），偏态只是推测。没有直方图，A 的"60% 容量覆盖 85% 命中"就是猜的，B 的预测器命中率也无法设计。

---

## 5. 任务分工

### 5.1 沙箱（aarch64 Linux，无 CUDA，无 Wine/box64）能做的

| # | 任务 | 状态 |
|---|---|---|
| S1 | GGUF 全量元数据 + 798 张量清单解析，切分专家/非专家体量 | ✅ 已完成 |
| S2 | 源码核对卸载正则与粒度限制（`common/common.h`、`models/k2-horizon.cpp`） | ✅ 已完成 |
| S3 | 核对 Strata 的架构门槛与几何锁定（README/CMake/头文件/文档/安装包） | ✅ 已完成 |
| S4 | 导出机器可读的张量清单 CSV，作为后续设计的固定输入 | ✅ 已完成（`data/k2-mova-36b-tensors.csv`，798 行，守恒差 720 B） |
| S5 | 编写"专家激活直方图"统计补丁（`build_moe_ffn` router 输出处插桩） | ⬜ 待做 |
| S6 | 纯算术建模：给定命中率/偏度假设，算 staging 池大小、预取深度、预期 tok/s | ⬜ 待做 |
| S7 | 设计 `mmid.cu` / 专家 buffer 管理的最小改造方案（文档级，不在沙箱编译 CUDA） | ⬜ 待做 |
| S8 | 核验四类专家张量的真实量化类型：**不读枚举名**，用 `n_bytes×8÷元素数` 反推 bits/元素匹配官方块定义。结果 gate/up/attn_v = **Q4_K**(4.5)、down = **混合 Q4_K/Q6_K**，798 张量 0 未知 | ✅ 已完成 |
| S9 | 可复现工具 `tools/dump_tensors.py`（63 行，含 `-ncmoe` 命中判定，已验证能重建 CSV） | ✅ 已完成 |

**沙箱明确做不到的**：跑 CUDA 后端（无 GPU）、跑你那个 `llama-cli.exe`（Windows 二进制，无 Wine/box64）、真机显存占用与速度测量。

### 5.2 GTX 1660 那台机器要做的（沙箱无法替代）

| # | 任务 | 为什么必须真机 | 产出物 |
|---|---|---|---|
| H1 | **专家激活直方图**：用 S5 补丁，对 ~50 条真实 prompt 跑 4096 token，统计每层 × 每专家命中次数 | 需要真跑模型 | `expert_hits.json`（决定路线 A/B 全部参数） |
| H2 | 实测基线：`-ncmoe` 不同 N 值下的显存/内存占用与 tok/s | 需要 CUDA | 分层收益的下界参照 |
| H3 | 实测 DDR4 / PCIe 带宽（`stream` + `nvbandwidth` 或 `bandwidthTest`） | 需要本机硬件 | 替掉本文 §2 表里的估算值 |
| H4 | 验证 `attn_v_exps` 未被 `-ncmoe` 覆盖时是否 OOM | 需要真加载 | 证实/证伪 S2 结论 |
| H5 | ⏬ **收益下调，降为低优先级** | 需磁盘+时间 | S8 已定量化：仅 21 层 Q6_K down 可统一为 Q4_K，实测省 1.064 GB / 每 token −6.1%（非原先误估的 13%） |

H1 是整个课题的瓶颈项：**没有它，A/B 的参数都只能猜**。

---

## 6. 方法论备忘（本次踩过的坑，勿重犯）

1. **有标准解析器就别手写** —— 手写 GGUF 解析漏了嵌套数组类型，读出 `moe_intermediate=640`（实际 768）。`pip install gguf` + `GGUFReader()` 一次到位。
2. **算完必做守恒检查** —— 分项之和是否等于已知总量（20.242 + 2.13 = 22.37 ✅）。曾有脚本把跨 45 层总和除以 100，得出每专家 94 MB（真实 1.34 MB），量纲自检能当场抓住。
3. **区分"我没验证"和"结论是这样"** —— 曾断言"`--override-tensor` 也碰不到 `attn_v_exps`"，实际它能（正则任意）；真正做不到的是单张 3D 内部只卸部分专家。
4. **大目录搜索必须限定文件集** —— `grep -rn` 扫 169 MB 源码树把沙箱压到连 `echo` 都超时。
5. **引用稳定量而非快照数字** —— 文档里写文件数/文件名，不写 `ahead_by=N`（会随提交漂移）。
6. **"用了官方库"不等于"读对了字段"** —— 即使数据来自 `GGUFReader`，我仍把**元数据值类型枚举**（12=BFLOAT16、14=BOOL）当成**张量量化类型枚举**去查名字，而 GGML 张量类型 12 其实是 `f16`、14 是 `q6_K` 那一套表。教训：同一份格式里常有多套编号空间，取名字前先确认这张表属于哪个枚举；更稳的做法是**用实测字节 ÷ 元素数反推 bits/元素**（本次得 4.46 / 5.49），它不依赖任何名字表，能独立判断标签是否可信。
7. **撤回要撤干净** —— 发现 `bf16` 标签不可信后，连带由它推出的"重量化可省 30% 流量"与任务 H5 一起标记阻塞/撤回，不能只改表格留着正文。

---

## 7. 结论

- 7B（`num_experts=0`，纯 dense）**不是本课题对象**——没有专家可分层。
- 36B-A4B 是真 MoE 且专家占 90.5%，是合适的研究对象；但它在 1660+16GB 上**不是"能不能装下"的问题，而是 1.545 GB/token 的搬运问题**。
- 显存不是瓶颈（放完非专家还剩 ~3.2 GB），**内存容量和 SSD 缺页才是**。
- llama.cpp 现有卸载机制粒度太粗（整层/整张量），任何有意义的层内热冷分层都必须改 `mmid.cu`。工作量集中在这里，也正是 Strata 论文的核心贡献所在。
- 下一步最高价值的动作：写 H1 的插桩补丁，拿到真实激活直方图。
