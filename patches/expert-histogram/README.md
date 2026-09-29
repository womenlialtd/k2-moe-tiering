# S5 — 专家激活直方图插桩（零侵入）

## 用途
统计真实推理里 **每层 × 每专家** 的命中次数与权重和 → `expert_hits.json`。
这是整个分层课题的瓶颈输入（H1）：没有它，热冷分层 / LRU / staging 池大小全只能猜。

## 为什么零侵入
不改 `llama-graph.cpp`、`llama-context.cpp`、`common/` 任何一行。
只用两个上游公开接口：
1. `llama_context_params.cb_eval`（`llama.h:386`，类型 `ggml_backend_sched_eval_callback`）——
   每个计算节点执行前后回调，`ask=false` 时张量数据已在设备上算好并同步（见 `ggml-backend.cpp:1775`）。
2. 图节点的 `t->name`。`llama-context.cpp:2483` 无条件挂 `graph_get_cb()`，
   它用 `ggml_format_name` 给张量命名为 `<name>-<层号>`。
   K2 里两个 router topk 节点被命名为 `ffn_moe_argsort-<il>`（`llama-graph.cpp:2057/2060`）
   与 `v_moe_argsort-<il>`（`k2-horizon.cpp:329/353`），且这两处 `cb()` 的是**完整 argsort 张量**（非 top-k view），
   dtype=I32、内存连续、无步长坑 —— 前 k 个元素即降序 top-k 专家编号。

## 拦截协议（照抄 examples/eval-callback + common/debug.cpp）
- `ask==true`：只有匹配 `*_moe_argsort` 才 `return true`（要求单独停算此节点）；
  其余 `return false`（被批量计算、不回调）。→ 只有 argsort 节点触发同步，开销可控。
- `ask==false`：`ggml_backend_tensor_get` 读回整段 I32 + 对应 probs（见下），累进直方图。

## top-k 大小与专家数
- **top-k 由 CLI 传**：`--moe-topk`（默认 8）、`--value-topk`（默认 4）。llama.cpp 没有公开 API 能从
  `llama_model` 读 `n_expert_used`，所以这两个数只能显式给，K2-Horizon 的正确值是 8 / 4。
- **专家数不靠 CLI**：直方图桶宽直接取张量自身的 `t->ne[0]`（FFN=100 / MoVA=64），
  这样即使传错 `--n-expert` 也不会越界。`--*topk > ne[0]` 时该节点被跳过而不是崩。
- 读回量按 `ggml_nbytes(t)` 算，不是按 `topk*n_tok` —— argsort 是**完整排序**输出，
  列主序，第 j 个 token 的第 c 名在 `flat[c + j*ne0]`。这个 stride 用错是这类插桩最常见的 bug，
  已用独立单测覆盖（见"自检"）。

## 只统计命中次数，不含权重
本补丁只产出**命中次数**（足以决定路由偏好与热冷分层）。gate 权重（softmax 后的值）
来自紧随其后的 `get_rows(probs, selected_experts)` 节点，名为 `ffn_moe_weights-<il>` /
`v_moe_weights-<il>`，形状 `[1, k, n_tok]` f32。若后续要加权热度，在 `cb_eval` 里
按同样的列主序多拦这两类节点即可（当前故意不做，少一次同步开销）。

## 构建（需在 1660 那台 Windows+CUDA 机上，或任何能编 llama.cpp 的环境）
```
# 把本目录放到 src 树 examples/expert-histogram/ 下，
# 并在 examples/CMakeLists.txt 的 add_subdirectory(eval-callback) 后面加一行：
#     add_subdirectory(expert-histogram)
cmake -B build -DLLAMA_CUBLAS=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build --config Release -t llama-expert-histogram
```

## 运行
```
llama-expert-histogram -m K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf \
    --prompts-file prompts.txt --moe-topk 8 --value-topk 4 \
    -ngl 99 -c 4096 --temp 0 -o out/expert_hits.json
```
`--prompts-file` 每行一条 prompt（目标 ~50 条，覆盖代码/数学/长文/对话）。
prefill 阶段整批 token 都计入（每个 position 独立统计），这是真实的专家访问分布。

## 输出 schema
```json
{
  "meta": {"model":"", "n_prompts":50, "n_tokens":12345,
           "moe_layers":[3,47], "moe_topk":8, "n_expert":100,
           "value_topk":4, "n_value_expert":64},
  "ffn":  {"3": {"counts":[c0,...,c99], "total":N, "expect":N}, ...},
  "value":{"3": {"counts":[c0,...,c63],  "total":N}, ...}
}
```
每层 counts 数组长度 = 该层专家数；据此可算 Gini/熵、top-p 覆盖率 → 直接喂给 S6 吞吐模型。

## 正确性自检（跑完必查，别只信产物）
1. 每层 `total == n_tokens × topk`（FFN total=n_tokens×8，MoVA ×4）。不等说明漏 token 或误命中。
2. `counts` 数组非全零（若某层全零，多半是该层是 dense blk0-2 被正确排除，或匹配串错）。
4. `samples` 非空且形如 `ffn_moe_argsort-3`；为空说明模型不是 MoE/MoVA 或上游改了节点名。
5. `ffn_layers` 应为 `[3,47]`（blk 0-2 是 dense），`value_layers` 同理。若不是，先查
   `leading_dense_block_count` 与 `-ngl` 是否把所有层都放进了计算图。
