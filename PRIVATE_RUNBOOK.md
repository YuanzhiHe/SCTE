# 私有数据上机手册（河北大学附属医院 203 例 / 河南省人民医院 491 例，均 1mm + 5mm）

> **队列就这两家**，没有第三家，也没有 8mm 队列。河北做内部训练，河南做外部验证，
> 命令是 `bash scripts/run_private.sh full /data/hebei hebei /data/henan`。

> 前提：设备在别人手里、接触时间有限。这份手册的目标是**上机当天不写代码、不调试**，
> 所有可能出错的地方都在公开数据上预先跑通过了。

---

## 0. 上机前一天（在自己机器上完成）

拷贝到一个 U 盘 / 目录，**不需要联网**：

```
SCTE/                          代码（scte_r/ scripts/ configs/）
public.pt                      确定性预训练权重（5mm/1mm，1.4 MB）
public_flow.pt                 流匹配解码器权重（2.8 MB）—— 零样本探针用
baselines/                     TVSRN / CTHNet / I3Net 官方模型定义（已在仓库内）
public_forward_op.pt           公开队列学出来的前向算子（35 KB，做对照/热启动）
runs/protocol_rplhr_5mm.json   公开队列的 protocol 常数（做对照用）
totalsegmentator_weights/      TS 权重（~320 MB，见下，**必带**）
env/                           离线依赖（见下）
```

离线依赖（对方机器多半不能 pip install）：

```bash
# 在自己机器上先下好轮子
pip download torch torchvision --index-url https://download.pytorch.org/whl/cu128 -d env/
pip download numpy scipy SimpleITK pydicom PyYAML TotalSegmentator nnunetv2 -d env/
# 上机后
pip install --no-index --find-links env/ torch torchvision numpy scipy SimpleITK pydicom PyYAML
pip install --no-index --find-links env/ TotalSegmentator
```

**TotalSegmentator 的权重必须提前拷贝**（它第一次运行会联网下载，对方机器多半没网，
这是整套流程最容易当场卡死的一步）：

```bash
# 自己机器上（权重已在本机 ~/.totalsegmentator，320 MB）
cp -r ~/.totalsegmentator totalsegmentator_weights
# 上机后
cp -r totalsegmentator_weights ~/.totalsegmentator
python -c "import totalsegmentator, torchvision; print('TS ok')"   # 必须不报错
```

`torchvision` 版本要和 `torch` 同一个 CUDA 通道，否则 TS 会在
`torchvision::nms does not exist` 上失败（公开数据阶段踩过这个坑）。

**没有 GPU 也能跑，但会慢很多**：所有脚本 `--device cpu` 可用；配对、算子拟合、证书是
CPU 主导；微调 80 epoch 在 CPU 上约 2–3 小时（可减到 20）。**阶段 2（TS 分割）和阶段 11
（整卷 × 64 步采样）在纯 CPU 上是数小时级**——没有 GPU 时阶段 2 加 `--fast`，
阶段 11 用 `--n 15 --flow_steps 32`，或干脆跳过阶段 11 只报 patch 级结果。

**数据目录要求**：一个病例一个子目录，里面放 1mm 和 5mm 两个序列（DICOM 文件夹或 NIfTI 都行）：

```
/data/hebei/
  CASE001/  1mm/*.dcm   5mm/*.dcm
  CASE002/  thin.nii.gz thick.nii.gz
```

薄/厚角色由 z 间距自动判定（读不到间距就按层数），也可以用 `--thin_hint 1mm --thick_hint 5mm` 强制。

---

## 1. 上机第一件事：15 分钟 dry run（**必做**）

```bash
bash scripts/run_private.sh dryrun /data/hebei hebei
# 数据若是归一化值而不是 HU：
# RESCALE="3072 -1024" bash scripts/run_private.sh dryrun /data/hebei hebei
```

只处理 6 例（划分成 3/3）、训练 3 epoch、采样 16 步、整卷只推 2 例。它会一次性暴露所有格式问题：
DICOM 读不出、薄厚角色判反、HU 不对、z 对不齐、TotalSegmentator 起不来。
**dry run 不过就不要往下走**，先看 `PRIVATE/hebei/results/01_pairing.log`。

> 这条路径已经在公开真实配对上完整排练过一遍（`PRIVATE/rehearsal/`，8 例走完 2–9 阶段，
> 每个阶段的输出文件都生成且格式正确）。dry run 检查的是**管路**不是**结论**：
> 3 epoch 微调只把 δ̂ 挪动约 0.007 HU，各臂在 dry run 里不会分开，这是正常的。

脚本**每一阶段都可断点续跑**：重跑时已完成的阶段自动跳过（`FORCE=1` 强制重做）。
时间不够时可以先跑到第 6 阶段（最重要的那个数就在那里）再说。

自检清单（dry run 输出里逐条对）：

| 检查项 | 期望 | 不对的话 |
|---|---|---|
| `[pair]` 行的层数/间距 | thin ≈ 5×thick 层数；间距 1.0 / 5.0 mm | 用 `--thin_hint/--thick_hint` 强制 |
| `[ok]` 行的 `thin (D,H,W)` | D 是 thick 的正好 5 倍 | 层数不整除会被自动裁剪，正常 |
| `HU[...]` 范围 | 约 [−1000, 200] | 若是 [0,1] 说明是归一化数据，加 `--rescale_hu 3072 -1024` |
| `off=` 偏移 | 多数为 0 或 ±1 | 大量 ±3/±4 说明两个序列不是同一次采集 |
| `lung=` 肺占比 | 0.2–0.5 | <0.05 说明肺分割失败（对比剂/严重病变） |
| `[2/9]` 阶段 | 每例写出 `<case>_lung.npy` | TS 失败则整套 LAA **不是临床口径**，见 §6 |

---

## 2. 正式跑（河北训练 + 河南外部验证，约 10–14 小时，可分两天）

```bash
bash scripts/run_private.sh full /data/hebei hebei
bash scripts/run_private.sh full /data/henan henan
```

**两家医院必须分别拟合各自的 protocol 常数**（有效层厚、纹理标定、阈值）——
设备和重建核不同，公开队列的常数不能直接套。脚本已经这么做了。

流程与耗时（50 例、单卡）：

| 阶段 | 内容 | 拟合/评测 | 耗时 |
|---|---|---|---|
| 1 | 配对 + z 对齐（**薄层分辨率**）+ 匿名化 | — | 20–30 min |
| 2 | TotalSegmentator 肺叶掩膜（**临床 LAA 口径的前提**） | — | 40–60 min |
| 3 | TRAIN/TEST 划分（给了第 4 个参数则是跨医院） | — | 秒级 |
| 4 | 该队列的 protocol 常数（有效层厚 + 纹理关系） | TRAIN | 5 min |
| 5 | 学该扫描仪的前向算子 `A_θ` | TRAIN | 10 min |
| 6 | 完美重建零分布 → 阈值（TRAIN 定，TEST 验样本外） | TRAIN→TEST | 10 min |
| 7 | **头条**：公开权重零样本 δ̂ + Landweber / 仅偏置对照 | TEST | 25 min |
| 8 | **骨干**：本队列训 CTHNet，然后缓存其重建（冻结） | TRAIN→两边 | 2–3 h |
| 9 | **生成级**：在冻结骨干上训流匹配残差 | TRAIN | 1 h |
| 10 | 证书 + 确定性回归对照（该设备是否也 ~20 HU 位移） | TRAIN→TEST | 1 h |
| 11 | 整卷：同一模型两个输出 + Lanczos / CTHNet 对照 | TEST | 3–5 h |

**为什么骨干要在他们的数据上重训**：CTHNet 决定图像质量，而图像质量高度依赖设备与重建核；
公开权重直接零样本用会低估它。第 8 阶段之后骨干**冻结**，生成级只学它剩下的残差。

**两个输出，一次推理**（第 11 阶段）：
- `volume_ours_s4.csv` —— 4 采样均值，**看图用**，肺 PSNR 与骨干差 ~0.25 dB
- `volume_ours_s1.csv` —— 单采样，**报指标用**，密度学是骨干的 1.5–2.2 倍准

**划分纪律**：凡是被"拟合"出来的东西（protocol 常数、前向算子、证书阈值、微调权重）
一律只用 TRAIN；凡是要"上报"的数字一律只在 TEST 上测。这和公开阶段的 85/15/50 是同一套规矩。
划分按匿名化 case id 排序后隔一取一，可复现、与目录顺序和运行日期无关，写在 `results/03_split.txt`。

> 注意：**零样本那一臂（阶段 7 的头条数字）本来不需要训练**，理论上可以在全部病例上测。
> 这里仍然只报 TEST，是因为证书阈值来自 TRAIN——只有这样 δ̂ 和 verdict 才是同一套口径下的样本外结论。

跑完自动打印并写出 `results/SUMMARY.txt`，两张表：逐例证书（patch 级）与全肺偏差（整卷）。

**队列最少 6 例**（阶段 4 至少要 3 例可用，而它只看得到一半）。少于 6 例就用 `NOSPLIT=1 bash scripts/run_private.sh full ...`
全部用同一批（此时微调臂是样本内结果，必须在论文里注明；零样本臂不受影响）。

---

## 3. 上机要拿到的数（按重要性排序）

1. **`public.pt` 在你们设备上的 δ̂**（阶段 7）。公开数据上是 −15.4 HU。
   如果这里也是十几 HU 的负偏移，就证明「平移换指标」不是某一台设备的偶然，
   而是这类目标函数的系统性行为——**这是论文最强的一句话**。
2. **流匹配 vs 确定性微调的全肺偏差符号**（阶段 11）。公开数据上，厚层四个指标一致地
   低估肺气肿（−0.49 / −7.58 pp，+19.4 / +21.5 HU），确定性微调把每一个都**翻到另一侧**
   （+0.22 / +5.52 / −8.49 / −6.92），流匹配则**同侧收窄**（−0.19 / −2.25 / +5.68 / +5.98）。
   在真实临床设备上复现这个"翻号 vs 收窄"的对照，是"真的恢复了分布"最直观的证据。
3. **该队列的有效层厚与前向算子残差**（阶段 4–5）。公开数据 5 mm 标称 → 实测 6.25 mm，
   学出算子后未解释残差 0.356 → 0.161。两家医院各自的值直接进论文表格，
   也是"算子必须实测不能假设"的证据。
4. **仅偏置对照的 N 曲线**（阶段 7）。公开数据上 N=3 就能追平完整微调。
   在真实临床队列上重复出来，这条对照就无法被审稿人质疑成"数据集特例"。
5. **证书零分布的宽度**（阶段 6）。公开真实数据 τ_δ = 4.33 HU，合成数据 0.29 HU。
   你们的设备落在哪，决定了证书在临床数据上的可用分辨率。
   ⚠️ 阈值 τ = |均值| + 2×标准差 是从零分布**估**出来的，例数少时它本身就抖：
   排练里同一个流匹配模型，在 4 例零分布下 4/4 通过，在 8 例零分布下 0/8——不是模型变了，
   是阈值收紧了。**通过率只有在几十例以上的零分布上才有意义**，dry run 的通过率不要解读。
6. **逐肺叶 LAA 偏差**（阶段 11）。公开数据上下叶与右中叶是厚层最吃亏的地方
   （−0.50 / −0.77 / −0.80 pp）。这条在真实数据上是否同样成立，决定能不能讲"层厚损失有解剖结构"。

**时间不够时的取舍顺序**：阶段 1→2→3→4→5→6→7 是必须的（7 依赖 5 学出的算子，最重要的那个数就在 7）；
阶段 8–9 可以砍预算（`BSTEPS=8000`、`--epochs 100`）；阶段 11 可以只推 15 例（`--n 15`）。

## 4. 能带出来什么（合规）

只带 `PRIVATE/<tag>/results/*.csv` 和 `*.log`：**逐例的数字，不含任何图像、不含病人标识**
（case 列是 sha1 前 12 位）。`pairs/` 目录（几十 GB 的 .npy）和 `ID_MAP_DO_NOT_EXPORT.csv`
留在对方机器上。带出来的 CSV 每行长这样：

```
case,w_hat,rho,delta_HU,rho_struct,s,verdict,violated,laa_ref,laa_rec,laa_thick,...
3f9a1c2b7e04,6.25,0.31,-15.02,0.28,+5.8,flagged,displacement,4.21,3.10,0.02,...
```

---

## 5. 出问题怎么办（现场排障）

| 症状 | 原因 | 处理 |
|---|---|---|
| `no readable DICOM series` | 病例目录下不是 DICOM 或有嵌套层级 | 把序列目录直接放在病例目录下一层 |
| 全部 `[skip] ... not CT-like` | 数据是归一化值不是 HU | 加 `--rescale_hu 3072 -1024`（先用一例确认空气峰落在 −1000） |
| 大量 `off=±4` | 两个序列不是同一次扫描 | 记下来并把这些例排除；这本身是个可报告的发现 |
| 阶段 4 报 `informative=False` | 纹理关系没有信息量 | 正常，证书会把该项报 UNAVAILABLE，其余照跑 |
| 显存不够 | patch 太大 | 全流程加 `--patch 32 48 48`；阶段 11 加 `--slab 30 --overlap 15` |
| 完全没有 GPU | — | 所有命令加 `--device cpu`，阶段 8–9 大幅砍预算，阶段 2 加 `--fast`；**阶段 8 的 CTHNet 在纯 CPU 上不可行** |
| TS 报 `torchvision::nms does not exist` | torch/torchvision CUDA 通道不一致 | 用同一 index-url 重装 torchvision |
| TS 一启动就卡在下载 | 权重没带 | `cp -r totalsegmentator_weights ~/.totalsegmentator`（§0） |
| 没有 `_lung.npy`，脚本继续跑了 | TS 失败 | 结果里所有 LAA 都是 HU 窗口径，**不能当临床值报**，见 §6 |
| 阶段 11 特别慢 | 整卷 × 64 步 × 两个输出 | `--n 15`；4 采样那一路可最后再补 |
| 阶段 8 显存不足 | CTHNet 在 256 裁剪下很重 | 已默认 `--amp --grad_ckpt`；仍不够则减 `--crop`（注意权重与裁剪绑定） |
| 阶段 4 报 `only N usable case(s) ... need at least 3` | 队列（的 TRAIN 一半）太小 | 报错会区分"单位不对"和"例数不够"两种原因；后者用 `NOSPLIT=1` 重跑 |
| 想中途停下改天再跑 | — | 直接 Ctrl-C，重跑同一条命令会跳过已完成阶段 |

---

## 6. 关于「先验证算法没问题并且有创新」

上机前公开数据上已经坐实的部分（`STAGE1_执行报告.md` §7–§9）：

- **正确性**：证书对「完美重建」放行 45/50，零分布 δ̂ = +1.04 ± 1.44 HU；估计量在
  4–10 mm 层厚范围内的漂移只有 0.04 HU。
- **创新性**：现有方法（本项目权重、真实配对微调、经典 Landweber 反卷积、
  agentic 影像系统）都没有「逐例、无参考、可判定」的密度学证据；检索到的最近邻
  （operator mismatch benchmark、双盲成像、residual-bounded restoration、kernel conversion）
  各自缺的那一步已经逐条写在 idea card 的 `differentiation_from_lit` 里。
- **发现**：这条技术路线现有的 LAA-950 收益 = 直方图整体平移，跨 4 个训练配置呈单调剂量-反应，
  且在**合成域**同样成立（δ̂ = −15.56 HU，零分布 ±0.13 HU）。
- **解法**：流匹配解码器在 δ̂ 只有 −0.55 HU 的前提下把 LAA MAE 做到 0.31 pp（优于靠平移的 0.35），
  证书 41/50 对 0/50；收益来自采样步数不是参数量；换上学出来的前向算子后再降一档。

**一条必须一起带过去的口径纪律**：LAA 的分母必须是 TotalSegmentator 解剖肺实质。
用 HU 窗 `(-990,-500)` 当分母会把同一例的 LAA-950 从 0.34–1.57 % 抬到 9.9–13.6 %
（`STAGE1_执行报告.md` §8.1，这是我在公开阶段犯过并更正的一个错）。阶段 2 失败而没补救的话，
带回来的所有 LAA 绝对值都不能当临床值报，只能做臂间相对比较。

私有数据要回答的是最后一步：**这两条结论（平移的普遍性、流匹配的收窄）在真实临床设备上是否复现**。
