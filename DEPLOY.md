# 部署与上机执行

对象：河北大学附属医院 203 例（内部训练）+ 河南省人民医院 491 例（外部验证），均 1 mm + 5 mm。
设备在第三方手里、接触时间有限，所以这份文件的目标是**当天不写代码、不调试**。

---

## 0. 装（约 20 分钟）

```bash
git clone git@github.com:YuanzhiHe/SCTE.git
cd SCTE
pip install -r requirements.txt
pip install TotalSegmentator einops timm scikit-image
bash scripts/fetch_baselines.sh          # I3Net 按许可证要求是用时拉取，不随仓库分发
```

`torch` 和 `torchvision` **必须来自同一个 CUDA 通道**：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

不一致时 `torchvision` 能正常 import，却会在 TotalSegmentator 内部炸在
`torchvision::nms does not exist`。下一步的自检会当场抓出来。

**TotalSegmentator 权重（320 MB）**：第一次运行会联网下载。医院能上网就不用管；
不确定就在自己机器上先跑一次，然后带过去：

```bash
tar czf ts_weights.tgz -C ~ .totalsegmentator     # 自己机器
tar xzf ts_weights.tgz -C ~                        # 医院机器
```

没有外网的机器：在有网的机器上跑 `bash scripts/make_bundle.sh`，得到含代码、权重、
TS 权重、离线 wheels 的自包含包。

---

## 1. 上机前必做：自检（约 3 分钟）

```bash
python scripts/preflight.py
```

它不是"import 一下看看"。它会**造一对已知亚 slab 偏移的合成 CT，跑真正的阶段 1 代码，
验证偏移被精确还原**，然后跑算子拟合和证书。除了 DICOM 读取本身，上机要用的每一环都验到了。

```
PRE-FLIGHT PASSED - environment and pipeline verified end to end.
```

**没看到这一行就不要往下走。** 退出码非 0 时它会明确指出是哪一项。

TotalSegmentator 那项报警告（而不是失败）时要特别注意：没有它，LAA 的分母会退化成 HU 窗，
**那不是临床 LAA-950**——同一例会从 0.34–1.57% 被抬到 9.9–13.6%。

---

## 2. 数据目录要求

一个病例一个子目录，里面放 1 mm 和 5 mm 两个序列（DICOM 文件夹或 NIfTI 均可）：

```
/data/hebei/
  CASE001/  1mm/*.dcm   5mm/*.dcm
  CASE002/  thin.nii.gz thick.nii.gz
```

薄/厚由 z 间距自动判定，读不到就按层数；也可用 `--thin_hint 1mm --thick_hint 5mm` 强制
（脚本已默认带上）。

---

## 3. dry run（约 15 分钟，**必做**）

```bash
bash scripts/run_private.sh dryrun /data/hebei hebei
```

6 例、3 epoch，只验管路，**不要解读它的任何数字**。它专门用来暴露 preflight 验不到的那一环：
DICOM 读不出、薄厚判反、HU 不是 HU、序列不是同一次采集。

数据是归一化值而不是 HU 时（HU 范围显示成 `[0, 4095]` 之类）：

```bash
RESCALE="3072 -1024" bash scripts/run_private.sh dryrun /data/hebei hebei
```

**紧接着阶段 1 会自动跑对齐自检**，抽 20 例检查亚 slab 偏移。这一项失败会直接
`exit 1` 停住。它值得单列出来说：亚 slab 错位在所有日志里都看不见，会把插值基线压低约 3 dB，
并且悄悄错移整个证书赖以成立的前向算子。这个项目为它整体重跑过一次。

---

## 4. 正式跑（约 10–14 小时，可分两天）

```bash
EXT_CAL=50 bash scripts/run_private.sh full /data/hebei hebei /data/henan
```

- 第 2 个参数 = 内部训练队列（河北），第 4 个 = **外部验证队列**（河南）
- `EXT_CAL=50` 从河南划 50 例做**扫描仪物理标定**（有效层厚、证书阈值），其余 441 例只上报。
  去掉它就是最严格的外部验证：河北的标定常数直接套到河南，一个数都不在河南拟合
- **每个阶段断点续跑**。Ctrl-C 之后重跑同一条命令会跳过已完成的阶段，`FORCE=1` 强制重做

### 11 个阶段

| # | 内容 | 拟合/评测 | 耗时 |
|---|---|---|---|
| 1 | 配对 + z 对齐（薄层分辨率）+ 匿名化 + **对齐自检** | — | 20–30 min |
| 2 | TotalSegmentator 肺叶掩膜 | — | 40–60 min |
| 3 | TRAIN/TEST（+ 标定）划分 | — | 秒级 |
| 4 | 该队列的 protocol 常数 | 标定集 | 5 min |
| 5 | 学该扫描仪的前向算子 `A_θ` | 标定集 | 10 min |
| 6 | 完美重建零分布 → 证书阈值 | 标定集→TEST | 10 min |
| 7 | **头条**：公开权重零样本 δ̂ + Landweber / 仅偏置对照 | TEST | 25 min |
| 8 | 骨干：本队列训 CTHNet，缓存重建后冻结 | TRAIN | 2–3 h |
| 9 | 生成级：冻结骨干上训流匹配残差 | TRAIN | 1 h |
| 10 | 证书 + 确定性回归对照 | TRAIN→TEST | 1 h |
| 11 | 整卷：同一模型两个输出 + Lanczos / CTHNet 对照 | TEST | **6–10 h** |

> 阶段 11 的耗时按**面内尺寸**平方增长。上表按 512×512 估（你们的数据），
> 公开数据（约 275×382）只要一半。实测：512×512 单采样约 6 min/例，
> 4 采样约 24 min/例。时间紧时先出单采样（报指标用），4 采样那一路可以事后补。

跑完自动写出 `results/SUMMARY.txt`。

### 时间不够时的取舍

阶段 **1→7 必须做**——最重要的那个数（公开权重在他们设备上的 δ̂）就在阶段 7，而且它
**不需要训练**。阶段 8–9 可砍预算（`BSTEPS=8000`，`--epochs 100`），阶段 11 可 `--n 15`。

---

## 5. 要拿到的四个数

1. **公开权重在他们设备上的 δ̂**（阶段 7）。公开数据上是 −15.4 HU。若这里也是十几 HU 的负偏移，
   就证明"平移换指标"不是某台设备的偶然。
2. **确定性回归的 δ̂**（阶段 10）。公开数据上所有确定性臂都落在 19–21 HU，与损失配方无关。
   在真实临床设备上复现这一点，是本文机制主张最硬的一块。
3. **看图输出 vs 骨干的肺 PSNR 差距**（阶段 11）。公开数据上是 −0.25 dB。
4. **指标输出的密度学优势**（阶段 11）。公开数据上 LAA-950 好 1.5×、LAA-910 好 2.2×、CCC 好 3.9×。

**一个口径陷阱**：河南队列病更重、更离散，**所有臂的 CCC 都会跳上去**。CCC 衡量的是
"能不能区分病人"，不是"数准不准"——本项目实测过：误差最小的轻度层 CCC 最低（0.519），
误差最大的重度层 CCC 反而更高（0.532）。**不能拿 CCC 当进步的证据，必须同时报 MAE 和偏差。**

---

## 6. 能带走什么

只带 `PRIVATE/<tag>/results/` 下的 `*.csv`、`*.log`、`SUMMARY.txt`：逐例数字，
**不含图像、不含病人标识**（case 列是 sha1 前 12 位）。

`PRIVATE/`、`ID_MAP_DO_NOT_EXPORT.csv`、所有 `*_thin/_thick/_lung/_base.npy` 都在
`.gitignore` 里，即使在医院机器上 `git add -A` 也不会被提交。要推结果回来必须显式
`git add -f`，这个摩擦是故意的——推之前自己看一眼。

---

## 7. 现场排障

| 症状 | 原因 | 处理 |
|---|---|---|
| `preflight` 在 torchvision 那项失败 | torch/torchvision CUDA 通道不一致 | 用同一 `--index-url` 重装 |
| `no readable DICOM series` | 病例目录下不是 DICOM 或有嵌套 | 把序列目录放到病例目录下一层 |
| 全部 `[skip] ... not CT-like` | 数据是归一化值不是 HU | 加 `RESCALE="3072 -1024"` |
| **`ALIGNMENT CHECK FAILED`** | 阶段 1 没把配对落到算子网格上 | **不要往下走**。先查 `--thin_hint/--thick_hint` 和 HU 范围 |
| 大量 `off=±4` | 两个序列不是同一次扫描 | 排除这些例；这本身是可报告的发现 |
| 阶段 4 报 `only N usable case(s)` | 队列（的 TRAIN 一半）太小 | 报错会区分"单位不对"和"例数不够"；后者用 `NOSPLIT=1` |
| 阶段 8 显存不足 | CTHNet 在 256 裁剪下很重 | 已默认 `--amp --grad_ckpt`；权重与裁剪尺寸绑定，减 crop 需重训 |
| 阶段 11 特别慢 | 整卷 × 64 步 × 两个输出 | `--n 15`；4 采样那一路最后再补 |
| 想中途停下 | — | Ctrl-C，重跑同一命令自动跳过已完成阶段 |
| 完全没有 GPU | — | 阶段 8 的 CTHNet 在纯 CPU 上不可行，只能跳过骨干、退回纯流匹配 |

---

## 8. 一句话版本

```bash
git clone git@github.com:YuanzhiHe/SCTE.git && cd SCTE
pip install -r requirements.txt && bash scripts/fetch_baselines.sh
python scripts/preflight.py                                    # 必须 PASSED
bash scripts/run_private.sh dryrun /data/hebei hebei           # 必须过
EXT_CAL=50 bash scripts/run_private.sh full /data/hebei hebei /data/henan
```
