# k2-moe-tiering

研究 **MoE 专家权重在 VRAM / RAM / SSD 三级存储之间的分层与并发预取**，研究对象是
K2-Horizon-MoVA-36B-A4B，目标硬件是 GTX 1660（6 GB）+ 16 GB DDR4。

- 主文档：[MOE-TIERING.md](MOE-TIERING.md) —— 实测模型事实、设计结论、沙箱/真机任务分工与状态
- 数据脚本：`tools/` —— 用官方 `gguf` 库解析真实 GGUF，产出机器可读清单

配套的两个构建仓库（本项目的模型就跑在它们的产物上）：

- [`womenlialtd/k2-llama-sm75`](https://github.com/womenlialtd/k2-llama-sm75) —— K2-Horizon fork + sm_75 Windows CUDA 包（含 CI 期临时打的 `src/unicode.cpp` 补丁）
- [`womenlialtd/llama-sm75-builds`](https://github.com/womenlialtd/llama-sm75-builds) —— 纯 workflow 仓库，编官方上游 b11213 的 sm_75 专用包

## 状态

任务清单和完成状态全部在 [MOE-TIERING.md §5](MOE-TIERING.md)。所有数字标注了来源
（实测 / 推导 / 量级估算 / 未验证），未验证的假设集中在 §5.2，只能在真机上补。

## 约定

- **不写估算值当结论**。每个数字要么有实测出处，要么显式标注"量级估算"。
- **算完做守恒检查**（分项之和 = 已知总量）。不通过就是错的，不进文档。
- **有标准解析器就用标准解析器**（GGUF 用 `pip install gguf`，别手写）。
