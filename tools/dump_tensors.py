#!/usr/bin/env python3
"""导出 K2-Horizon-MoVA-36B-A4B GGUF 张量清单 + -ncmoe 正则命中判定。
方法：量化类型不读枚举名（多套编号空间易错），改用 实测 n_bytes*8/元素数
      匹配 ggml 官方块 bits/元素。依赖 pip install gguf。
用法：python3 tools/dump_tensors.py [gguf路径]  (输出 data/k2-mova-36b-tensors.csv)
      也可用环境变量 K2_GGUF 指定；两者都缺省时按下方 DEFAULTS 顺序探测。
"""
import csv, os, re, sys
from collections import defaultdict
from gguf import GGUFReader

# 候选路径：K2_GGUF 环境变量优先，其次常见放置位置。找不到会明确报错，不猜。
DEFAULTS = [os.environ.get("K2_GGUF", ""),
            "K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf",
            "../K2-Horizon-MoVA-36B-A4B-Q4_K_M.gguf"]
# (bits/元素, 名称) 来自 ggml-common.h：QK_K=256, 每块字节/256
THEO = [(4.5,'Q4_K'),(5.5,'Q5_K'),(6.5625,'Q6_K'),(8.0,'Q8_0'),(16.0,'F16/BF16'),
        (32.0,'F32'),(2.625,'Q2_K'),(3.4375,'Q3_K'),(4.25,'IQ4_XS'),(5.25,'IQ3'),
        (2.25,'IQ2_XXS'),(2.5,'IQ2_XS'),(3.0625,'IQ3_XXS')]
NCMOE = re.compile(r"\.ffn_(up|down|gate|gate_up)_(ch|)exps")

def quant_of(bpp):
    for v, name in THEO:
        if abs(bpp - v) / v < 0.005:
            return name
    return f"UNKNOWN({bpp:.4f})"

def main():
    path = next((p for p in ([sys.argv[1]] if len(sys.argv) > 1 else []) + DEFAULTS if p and os.path.exists(p)), "")
    if not path:
        sys.exit("找不到 GGUF：把路径作为参数传入，或设 K2_GGUF 环境变量。候选： %s" % DEFAULTS[1:])
    fs = os.path.getsize(path)
    r = GGUFReader(path)
    rows, agg = [], defaultdict(lambda: [0, 0, set()])
    for t in r.tensors:
        dims = [int(x) for x in t.shape]
        el = 1
        for d in dims: el *= d
        nb = int(t.n_bytes); bpp = nb * 8 / el
        q = quant_of(bpp)
        base = re.sub(r"^blk\.\d+\.", "", t.name)
        agg[base][0] += 1; agg[base][1] += nb; agg[base][2].add(q)
        rows.append({"name": t.name, "dims": "x".join(map(str, dims)),
                     "n_elements": el, "n_bytes": nb,
                     "bits_per_element": round(bpp, 4), "quant": q,
                     "ncmoe_match": "YES" if NCMOE.search(t.name) else "no",
                     "data_offset": int(t.data_offset)})
    rows.sort(key=lambda x: x["data_offset"])
    tot = sum(x["n_bytes"] for x in rows)
    hdr = min(x["data_offset"] for x in rows)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..",
                                       "data", "k2-mova-36b-tensors.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(f"张量={len(rows)}  量化未知={sum(1 for x in rows if x['quant'].startswith('UNKNOWN'))}")
    print(f"sum(n_bytes)+hdr={tot+hdr:,}  file={fs:,}  守恒差={fs-(tot+hdr):,} (应为对齐余数<~1KB)")
    EX = ['ffn_gate_exps.weight','ffn_up_exps.weight','ffn_down_exps.weight','attn_v_exps.weight']
    gx = sum(agg[b][1] for b in EX)
    ne = sum(v[1] for k, v in agg.items() if k not in EX)
    print(f"专家={gx/1e9:.3f} GB  非专家={ne/1e9:.3f} GB  占比={100*gx/(gx+ne):.1f}%")
    hit = sum(1 for x in rows if x["ncmoe_match"] == "YES")
    hb = sum(x["n_bytes"] for x in rows if x["ncmoe_match"] == "YES")
    mb = sum(x["n_bytes"] for x in rows if x["ncmoe_match"] == "no" and "_exps" in x["name"])
    print(f"-ncmoe 命中={hit} 张量/{hb/1e9:.2f} GB   漏掉的专家(attn_v_exps)={mb/1e9:.2f} GB")

if __name__ == "__main__":
    main()
