# k2-moe-tiering

研究 **MoE 专家权重在 VRAM / RAM / SSD 三级存储之间的分层与并发预取**，研究对象是
K2-Horizon-MoVA-36B-A4B，目标硬件是 GTX 1660（6 GB）+ 16 GB DDR4。

## 目录

- **[MOE-TIERING.md](MOE-TIERING.md)** —— 主文档：实测模型事实、设计结论、沙箱/真机任务分工与状态
- **`data/k2-mova-36b-tensors.csv`** —— 798 张量清单（dims / bytes / 实测 bits-per-elem / 量化 / `-ncmoe` 命中）
- **`tools/dump_tensors.py`** —— 生成上面 CSV 的可复现脚本（官方 `gguf` 库 + 算术反推量化，不读枚举名）
- **`patches/expert-histogram/`** —— 专家激活直方图插桩（**零侵入**独立 example，跑真机 H1 用）
- **`model/tier_model.py`** —— 三级分层算术吞吐模型（给定偏度 s 与容量 → 预期 tok/s）
- **`design/S7-mmid-tiering-plan.md`** —— 层内专家分层的 llama.cpp 改造方案
- **`TODO.md`** —— 会话交接指针（上下文被压缩后先读这个恢复状态）

配套的两个构建仓库（本项目的模型就跑在它们的产物上）：

- [`womenlialtd/k2-llama-sm75`](https://github.com/womenlialtd/k2-llama-sm75) —— K2-Horizon fork + sm_75 Windows CUDA 包（含 CI 期临时打的 `src/unicode.cpp` 补丁）
- [`womenlialtd/llama-sm75-builds`](https://github.com/womenlialtd/llama-sm75-builds) —— 纯 workflow 仓库，编官方上游 b11213 的 sm_75 专用包

## 三个核心数字（均为实测字节推导，与量化名称无关）

- 专家权重 **20.24 GB**（占 90.5%），非专家 **2.115 GB**
- **每 token 访存 1.545 GB** —— 真正的瓶颈，不是"22 GB 装不下"
- 单层活跃工作集仅 **34–36 MB** —— 流式方案不需要常驻显存

## 状态

沙箱侧 **S1–S9 全部完成**（见 [MOE-TIERING.md §5](MOE-TIERING.md)）。
下一步只能在 1660 真机做：用 `patches/expert-histogram/` 跑 H1 激活直方图，
把 `model/` 里假设的偏度换成实测值、把 `design/` 的热集划分定下来。

## 约定

- **不写估算值当结论**。每个数字要么有实测出处，要么显式标注"量级估算 / 假设 / 未验证"。
- **算完做守恒检查**（分项之和 = 已知总量）。不通过就是错的，不进文档。
- **有标准解析器就用标准解析器**（GGUF 用 `pip install gguf`，别手写）。
- **量化类型不读枚举名**，用 `n_bytes×8 ÷ 元素数` 反推 bits/元素匹配官方块定义。
- **引用稳定量**（文件数/文件名/行号），不引用会漂移的快照数字。
