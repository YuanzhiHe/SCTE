# SCTE-R · 第一阶段交接:公开数据集预训练

**目标(这一阶段只做这一件事):** 在**公开薄层 CT** 上把 SCTE-R 管线跑通并训出一个可用的初始权重 `public.pt`。厚层不需要真实采集——由 SSP 前向算子从薄层**在线合成**,得到完美配对监督。私有数据(河北 5mm、安阳 8mm)是**下一阶段**的事,本包不涉及。

> 单次前向变体(`--iters 1`)即可完成本阶段。迭代/多智能体变体、基线对比(BIS/DLS)、出图、私有 DICOM 审计**不在本包内**,是后续阶段的模块。

---

## 0. 包内文件清单(全部已验证可跑)

```
scte_r/
  __init__.py
  forward_operator.py   # SSP 前向算子:合成厚层 + 数据一致性共用(全管线核心)
  modules.py            # M1 算子/噪声辨识 · FiLM · M2 3D 残差 backbone · M3 数据一致 · M5 可靠性
  model.py              # build_model() / SCTE_R 组装(单次前向)
  metrics.py            # LAA-950 / Perc15 / PSNR / SSIM / CCC / NPS
  losses.py             # SCTERLoss:体素 + M3 数据一致 + M4 泛函轨迹护栏 + M4 NPS 噪声校准
  datasets.py           # SimulatedThickDataset:读公开薄层 .npy,在线合成厚层
  train.py              # 训练入口
  evaluate.py           # 评估:PSNR + LAA-950/Perc15 MAE + CCC
scripts/
  smoke_test.py         # 无数据无GPU 端到端自检(先跑这个)
  prep_public.py        # 公开 DICOM / NIfTI / .mhd  ->  .npy(HU, D,H,W)  ← 本阶段第一步
configs/default.yaml    # 超参(层厚、patch、loss 权重、ladder)
requirements.txt
```

---

## 1. 环境

```bash
pip install -r requirements.txt
python scripts/smoke_test.py          # 期望最后一行打印 [smoke] PASS
```

`smoke_test.py` 不需要数据/GPU,验证 M1–M5 前向 + 4 项 loss + 反向传播全部跑通。**没过就先别往下走。**

---

## 2. 数据准备(本阶段的第一段活)

### 用哪个公开集
| 数据集 | 格式 | 说明 | 建议 |
|---|---|---|---|
| **RPLHR-CT** | NIfTI | 唯一真实 5mm↔1mm 配对,250 例;**取其 1mm 薄层** | **先用它起步**(直接相关、量小、快) |
| LIDC-IDRI | DICOM 序列 | 大规模胸部薄层,一例一文件夹 | 扩规模用 |
| LUNA16 | .mhd/.raw | LIDC 的整理子集 | 扩规模用 |

> **只喂薄层(1mm 级)。** 厚层由代码合成,**不要**把真实 5mm 喂进来。RPLHR 的真实 5mm 留到下一阶段做 sim-vs-real 对照。

### 转成训练要的 .npy
`SimulatedThickDataset` 读 `--root` 下的 `<case>.npy`(`float32`,HU,形状 `(D,H,W)`)。用 `prep_public.py` 一键转:

```bash
# NIfTI 或 .mhd 直接放在一个目录里:
python scripts/prep_public.py --in /data/RPLHR/thin --out DATA/public_thin
# DICOM:每个病例是 --in 下的一个子文件夹:
python scripts/prep_public.py --in /data/LIDC --out DATA/public_thin --dicom
```

脚本会:走 SimpleITK 读取(自动套 CT rescale slope/intercept → 输出即 HU)、按 z 排好、HU 裁剪到 `[-1000,200]`、跳过非 CT/过薄的体、逐例存 `.npy`。产物直接就是 `--root`。

---

## 3. 训练

```bash
python -m scte_r.train --data simulated --root DATA/public_thin \
       --epochs 50 --bs 2 --lr 1e-3 --ckpt public.pt
# 冒烟(几步)确认能收敛趋势:加 --steps 5 --device cpu
```

关键参数(见 `configs/default.yaml`,命令行同名 flag 覆盖):
- `--downsample 5`:厚/薄层间距比(5mm/1mm)。**默认先训 5mm。**
- `--iters 1`:单次前向。>1 走迭代变体,但那需要 `agentic.py`(不在本包)。
- patch `(40,64,64)`、`base_ch 32`、`n_blocks 6`、loss 权重 `w_dc/w_traj/w_nps`。

> **跨层厚(为下阶段安阳 8mm 铺路,可选加做)**:再训一个覆盖 5–8mm 的版本——对同一批薄层分别用 `--downsample 5`、`--downsample 8` 各训一遍或混合,让模型见过 8mm 退化。这样下阶段真实 8mm 才在算子分布内。本阶段可先只做 5mm 打通。

产物:`public.pt`,作为下阶段私有微调的 `--init`。

---

## 4. 评估与验收标准

```bash
python -m scte_r.evaluate --data simulated --root DATA/public_thin --ckpt public.pt
```

打印 `n / PSNR / LAA-950 MAE(thick) / LAA-950 MAE(SCTE-R) / Perc15 MAE / CCC`。

**这一阶段算成功的判据(合成配对上):**
1. **`LAA-950 MAE(SCTE-R)` 明显小于 `LAA-950 MAE(thick)`** —— 即重建确实把厚层的低估拉回来了(这是最核心的一条)。
2. `LAA-950 CCC(recon vs ref)` 趋近 1(合成集上应能到 0.9+)。
3. `PSNR` 为正且随训练上升,`Perc15 MAE` 下降。

> 合成集上的漂亮数字**只证明管线和优化正确**,不是临床声明;临床结论在下一阶段私有真实配对上做。

---

## 5. 交接边界(哪些故意没给 / 下阶段再接)

- **私有数据管线**(河北 5mm、安阳 8mm 的 DICOM 扫描/同核配对审计):`scan_format.py` / `audit_pairs.py`,下阶段随私有数据一起给。
- **基线对比**(BIS 双三次 / DLS 即 Yu 类纯图像保真):`baselines.py`,出结果阶段用。
- **出图**(Fig.2/3 渲染、Fig.4 阅片、Fig.6 肺气肿):`render_cases.py` / `reader_study.py` / `plot_fig6.py`。
- **迭代/多智能体变体**:`agentic.py` / `orchestration.py`(`--iters>1` 才需要)。

需要哪块我再从主包里拆给你。

---

## 6. 一句话给接手的 agent

先 `smoke_test.py` 过关 → `prep_public.py` 把 RPLHR 的 **1mm 薄层**转成 `DATA/public_thin/*.npy` → `train.py --data simulated` 训出 `public.pt` → `evaluate.py` 确认 **SCTE-R 的 LAA-950 MAE < thick 的 LAA-950 MAE**。达成即本阶段完成。
