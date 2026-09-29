#!/usr/bin/env python3
"""S6: 专家分层（SSD→RAM→VRAM 三级）算术吞吐模型。
纯解析式，不依赖真机。结论看拐点与单调性，不看末位小数。

口径关键：分层决策作用于【全局专家池】，不是每层。
  FFN 全局 = 45 层 × 100 = 4500 个专家；value 全局 = 45 × 64 = 2880 个。
  一个 token 命中某专家 = 需要它此刻在 VRAM。
  - 常驻 VRAM  → 0 传输
  - 在 RAM 不在 VRAM → 1× PCIe
  - 在 SSD → NVMe(→RAM) + PCIe(→VRAM) + 1 次缺页时延
容量分配按各自每 token 字节占比（FFN 82.8% / value 17.2%）。

常数来源：blob/专家数/层数 = CSV 实测；BW_*, L_FAULT = 保守估计待 H3 替换；s = 假设待 H1 替换。
"""
import math

NL=45; NE=100; K=8; NVE=64; KV=4
HKN=1   # 确认值：元数据无 hkn 键，且 build_routed_value 的 router 形状 [64,n_tokens] ⇒ 按 token 选而非按 head 选
B_GATE=B_UP=1.1059e6
B_DOWN=(24*1.1059e6+21*1.6128e6)/45       # 混合 down 每专家均值
B_FFN=B_GATE+B_UP+B_DOWN                  # 每 FFN 专家(gate+up+down) ≈ 3.55 MB
B_VAL=1.4746e6                            # 每 value 专家

BW_RAM,BW_PCIE,BW_NVME=25e9,10e9,2.5e9
BW_VRAM=240e9                           # 1660 GDDR6 实测约 240 GB/s，待 H3 核
L_FAULT=1.2e-4                            # 单次 SSD→RAM 缺页时延

VRAM=3.2e9; RAM=11.3e9
NFFN_TOT=NE*NL; NVAL_TOT=NVE*NL
# 每 token 字节切分（决定容量如何分配）
tok_ffn=K*B_FFN*NL; tok_val=HKN*KV*B_VAL*NL; B_TOK=tok_ffn+tok_val
f_ffn=tok_ffn/B_TOK                       # ≈0.828

def zipf_hit(n,s,cap):
    if cap<=0: return 0.0
    cap=min(int(cap),n)
    w=[1.0/(i+1)**s for i in range(n)]
    return sum(w[:cap])/sum(w)

def tok_s(s,vr,rm,hkn=HKN):
    """vr,rm: VRAM/RAM 分给 FFN 的专家数；value 侧按各自剩余字节预算算（不重复计池）。
       返回 (tok/s, FFN_VRAM命中, FFN_RAM命中)。"""
    tok_val_=hkn*KV*B_VAL*NL; bt=tok_ffn+tok_val_
    ff=tok_ffn/bt; fv=tok_val_/bt
    vrv=int((VRAM - vr*B_FFN)/B_VAL); rmv=int((RAM - rm*B_FFN)/B_VAL)
    pvf=zipf_hit(NFFN_TOT,s,vr); prv=zipf_hit(NVAL_TOT,s,vrv)
    prf=zipf_hit(NFFN_TOT,s,rm); prv2=zipf_hit(NVAL_TOT,s,max(0,rmv))
    # 各链路承载字节比例
    fp=(1-pvf)*ff + (1-prv)*fv          # 需跨 PCIe（不在 VRAM）
    fn=(1-prf)*ff + (1-prv2)*fv         # 需跨 NVMe（不在 RAM）
    acc=K*NL + hkn*KV*NL
    T = pvf*ff*bt/BW_VRAM + fv*prv*bt/BW_VRAM   # 常驻 VRAM 的读取也吃 VRAM 带宽
    T += fp*bt/BW_PCIE + fn*bt/BW_NVME + fn*acc*L_FAULT
    return (1.0/T if T>0 else 0.0), pvf, prf

def main():
    print(f"每 token 访存={B_TOK/1e9:.4f} GB (FFN {tok_ffn/1e9:.3f} + value {tok_val/1e9:.3f})")
    print(f"每 FFN 专家={B_FFN/1e6:.2f}MB  每 value 专家={B_VAL/1e6:.2f}MB")
    vr=int(VRAM*f_ffn/B_FFN); rm=int(RAM*f_ffn/B_FFN)
    vrv=int(VRAM*(1-f_ffn)/B_VAL); rmv=int(RAM*(1-f_ffn)/B_VAL)
    print(f"VRAM 可钉 FFN≈{vr}/{NFFN_TOT} (value≈{vrv}/{NVAL_TOT})")
    print(f"RAM  可钉 FFN≈{rm}/{NFFN_TOT} (value≈{rmv}/{NVAL_TOT})")
    print("\n命中率(FFN 全局池 4500, VRAM cap %d, RAM cap %d):"%(vr,rm))
    print(f"{'s':>5}{'VRAM命中':>10}{'RAM命中':>10}")
    for s in(0.6,0.8,1.0,1.2):
        print(f"{s:>5.2f}{zipf_hit(NFFN_TOT,s,vr)*100:>9.1f}%{zipf_hit(NFFN_TOT,s,rm)*100:>9.1f}%")
    print("\ntok/s（VRAM 钉最热 %d 个 FFN 专家，RAM 钉 %d 个）:"%(vr,rm))
    for s in(0.6,0.8,1.0,1.2):
        t,_,_=tok_s(s,vr,rm); print(f"  s={s:.2f}: {t:5.2f} tok/s")
    print("\n对照：VRAM 不钉热集(纯流式)，同 s、RAM 钉 %d 个:"%rm)
    for s in(0.8,1.0,1.2):
        t,_,_=tok_s(s,0,rm); print(f"  s={s:.2f}: {t:5.2f} tok/s")
    print("\n天花板（每 token 全走单一路径 1.545 GB）:")
    print(f"  CPU 直算专家走 DDR4 = {BW_RAM/B_TOK:5.2f} tok/s")
    print(f"  全驻 RAM→PCIe→GPU   = {BW_PCIE/B_TOK:5.2f} tok/s   ← GPU 分层硬上限")
    print(f"  全缺页 SSD→RAM      = {BW_NVME/B_TOK:5.2f} tok/s")
    print(f"  全常驻 VRAM(装不下) = {240e9/B_TOK:5.2f} tok/s   ← 仅参照")
    print("\n敏感性 hkn(value专家数倍):",end=" ")
    for hkn in(1,2,3): print(f"hkn{hkn}:{tok_s(1.0,vr,rm,hkn)[0]:.2f}",end="  ")
    print("\n敏感性 L_FAULT (s=1.0):",end=" ")
    global L_FAULT
    for lf in(2e-5,1.2e-4,5e-4):
        L_FAULT=lf; print(f"{lf*1e6:.0f}us:{tok_s(1.0,vr,rm)[0]:.2f}",end="  ")
    print()

if __name__=="__main__":
    main()
