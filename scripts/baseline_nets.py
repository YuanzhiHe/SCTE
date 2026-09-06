"""Train and evaluate the two published thick-to-thin CT baselines on OUR split.

  TVSRN   - Yu et al., MICCAI 2022 (the RPLHR-CT benchmark network)
  CTHNet  - Yu et al., npj Digital Medicine 7:335 (2024)

The network definitions in `baselines/` are the authors' unmodified files; this
script only supplies data, optimiser and evaluation. Three things are copied from
upstream on purpose, because getting any of them wrong would understate a baseline:

  * z alignment: thick slice k pairs with thin slice 5k, and a window of c_z thick
    slices supervises thin slices [5*z_s+3, 5*(z_s+c_z-1)-3) - upstream get_train_img.
  * hyper-parameters: the upstream config defaults (c_z=4 for TVSRN, 8 for CTHNet,
    256x256 in-plane crops, AdamW lr 3e-4 wd 1e-4, batch 1, L1 loss).
  * units: raw Hounsfield units, no normalisation - upstream feeds SimpleITK arrays
    straight in.

Usage
  python scripts/baseline_nets.py train --net tvsrn  --root DATA/public_pairs --steps 20000
  python scripts/baseline_nets.py infer --net tvsrn  --root DATA/public_pairs_test \
      --ckpt runs/BL_tvsrn.pt --csv results/vol_tvsrn.csv
"""
import argparse, csv, importlib, os, sys, types
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r import metrics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Upstream config defaults, verbatim from RPLHR-CT/config/default.txt and
# CTHNet-for-CT-Slice-Thickness-Reduction/config/all.txt.
DEFAULTS = dict(
    tvsrn=dict(ratio=5, c_z=4, c_y=256, c_x=256, T_mlp=4, T_pos=True,
               TE_c=8, TE_l=1, TE_d=4, TE_n=8, TE_w=8, TE_p=8,
               TD_p=8, TD_s=1, TD_Tw=4, TD_Tl=1, TD_Td=4,
               TD_Iw=8, TD_Il=2, TD_Id=4, TD_n=8),
    cthnet=dict(ratio=5, c_z=8, c_y=256, c_x=256, T_mask=False, T3d_mlp=4, T_rc='1conv',
                T3dE_w=[2, 2, 2], T3dE_l=1, T3dE_d=4, T3dE_n=8, T3dE_c=32,
                TD_s=1, TD_p=8, TD_n=8, TD_Td=4, TD_Tl=1, TD_Tw=4,
                TD_Id=4, TD_Il=2, TD_Iw=8),
    # upstream opt/i3net.json + config.py defaults, with upscale set to our ratio
    i3net=dict(kernel_size=3, n_feats=64, num_blocks=16, res_scale=1, head_num=1,
               window_size=16, win_num_sqrt=16, upscale=5, lr_slice_patch=4),
)
# thick slices consumed per forward, and thin slices produced
WINDOW = dict(tvsrn=(4, 10), cthnet=(8, 30), i3net=(4, 16))


def load_net(which, c_y, c_x):
    """Import the vendored network in isolation.

    Both directories ship a module literally called `swin_utils` and a `config`
    that the networks import by bare name, so they cannot both live in sys.modules
    at once. Build the config shim, import inside a scoped sys.path, then evict the
    names again so the other network can be loaded later in the same process.
    """
    cfg = dict(DEFAULTS[which]); cfg.update(c_y=c_y, c_x=c_x)
    if which == 'i3net':
        cfg['hr_slice_patch'] = cfg['upscale'] * (cfg['lr_slice_patch'] - 1) + 1
        cfg['c_z'] = cfg['lr_slice_patch']
    opt = types.SimpleNamespace(**cfg)
    shim = types.ModuleType('config'); shim.opt = opt

    # The vendored files use package-relative imports between themselves but a bare
    # `from config import opt`; so import them as a package, with the shim installed
    # under the bare name they expect.
    saved = sys.modules.pop('config', None)
    sys.modules['config'] = shim
    try:
        if which == 'i3net':
            # I3Net takes its hyper-parameters as an args object, not a global
            m = importlib.import_module('baselines.i3net.basic_model')
            net = m.make_model(opt)
        else:
            for sub in (('swin_utils', 'swin_utils_3d') if which == 'cthnet' else ('swin_utils',)):
                importlib.import_module('baselines.%s.%s' % (which, sub))
            m = importlib.import_module('baselines.%s.model_TransSR' % which)
            net = (m.TVSRN() if which == 'tvsrn' else m.TVSRN3d_v1())
    finally:
        sys.modules.pop('config', None)
        if saved is not None: sys.modules['config'] = saved
    return net, opt


def net_io(which, z_shift):
    """Per-network tensor layout and supervised z window.

    tvsrn/cthnet take (1,1,c_z,H,W) and emit the slices strictly between the outer
    thick slices; I3Net takes (1,H,W,c_z) and emits the whole span from the first
    slab centre to the last, copying the input into every `upscale`-th output.
    Both windows are expressed against our aligned convention (thick k sits at thin
    5k+z_shift), so changing the data convention changes one number, not the code.
    """
    c_z, n_out = WINDOW[which]
    if which == 'i3net':
        def wrap(t): return t.permute(1, 2, 0)[None]          # (S,H,W)->(1,H,W,S)
        def unwrap(o): return o[0].permute(2, 0, 1)           # (1,H,W,S)->(S,H,W)
        def window(z_s): return 5 * z_s + z_shift, 5 * z_s + z_shift + n_out
    else:
        def wrap(t): return t[None, None]
        def unwrap(o): return o[0, 0]
        def window(z_s): return (5 * z_s + 3 + z_shift,
                                 5 * (z_s + c_z - 1) - 2 + z_shift)
    return c_z, wrap, unwrap, window


def enable_grad_checkpointing(net):
    """Turn on the upstream blocks' own gradient checkpointing.

    The vendored code implements `use_checkpoint` but never wires it to a config
    flag. Setting the attribute post-construction leaves the network's mathematics
    untouched (checkpointing only trades compute for activation memory) and avoids
    editing the authors' file. Without it CTHNet does not fit at the upstream 256
    crop on a 16 GB card, and shrinking the crop is NOT an option - the patch
    embedding size is baked into the parameters.
    """
    n = 0
    for mod in net.modules():
        if hasattr(mod, 'use_checkpoint'):
            mod.use_checkpoint = True; n += 1
    print('gradient checkpointing enabled on %d blocks' % n)
    return net


def cases_in(root):
    return sorted(f for f in os.listdir(root) if f.endswith('_thin.npy'))


def train(a):
    net, opt = load_net(a.net, a.crop, a.crop)
    net = net.cuda().train()
    if a.grad_ckpt: enable_grad_checkpointing(net)
    print('%s: %.2fM parameters' % (a.net, sum(p.numel() for p in net.parameters()) / 1e6))
    o = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    files = cases_in(a.root)
    vols = [(np.load(os.path.join(a.root, f), mmap_mode='r'),
             np.load(os.path.join(a.root, f.replace('_thin', '_thick')), mmap_mode='r'))
            for f in files]
    print('%d training cases' % len(vols))
    c_z, wrap, unwrap, window = net_io(a.net, a.z_shift)
    rng = np.random.default_rng(0)
    run, t0 = 0.0, 0
    for it in range(1, a.steps + 1):
        thin, thick = vols[rng.integers(len(vols))]
        Dt, H, W = thick.shape
        if Dt <= c_z: continue
        if H < a.crop or W < a.crop: continue      # 85 training volumes are all >= 256
        z_s = int(rng.integers(0, Dt - c_z))              # upstream: randint(0, z-1-c_z)
        y_s = int(rng.integers(0, H - a.crop + 1))
        x_s = int(rng.integers(0, W - a.crop + 1))
        z_e = z_s + c_z
        m_s, m_e = window(z_s)
        if m_e > thin.shape[0]: continue
        x = torch.from_numpy(np.ascontiguousarray(
            thick[z_s:z_e, y_s:y_s + a.crop, x_s:x_s + a.crop]).astype(np.float32))
        y = torch.from_numpy(np.ascontiguousarray(
            thin[m_s:m_e, y_s:y_s + a.crop, x_s:x_s + a.crop]).astype(np.float32))
        x = wrap(x).cuda(); y = y.cuda()
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
            p = unwrap(net(x))
            loss = torch.nn.functional.l1_loss(p.float(), y)
        o.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        o.step()
        run += loss.item(); t0 += 1
        if it % 200 == 0:
            print('it %6d  L1 %.2f HU' % (it, run / max(t0, 1)), flush=True); run, t0 = 0.0, 0
        if it % 2000 == 0:
            torch.save(net.state_dict(), a.ckpt)
    torch.save(net.state_dict(), a.ckpt); print('saved', a.ckpt)


@torch.no_grad()
def infer(a):
    from skimage.metrics import structural_similarity
    net, opt = load_net(a.net, a.crop, a.crop)
    net = net.cuda().eval()
    net.load_state_dict(torch.load(a.ckpt, map_location='cuda'))
    c_z, wrap, unwrap, window = net_io(a.net, a.z_shift)
    files = cases_in(a.root)
    if a.n: files = files[:a.n]
    rows = []
    for ci, f in enumerate(files, 1):
        thin = np.load(os.path.join(a.root, f)).astype(np.float32)
        thick = np.load(os.path.join(a.root, f.replace('_thin', '_thick'))).astype(np.float32)
        mp = os.path.join(a.root, f.replace('_thin', '_lung'))
        has_mask = os.path.exists(mp)
        if not has_mask and not a.save_recon:
            print('[skip] %s: no mask' % f); continue
        # Caching the base reconstruction needs no lung mask - the mask only enters
        # the metrics. The training split has no TotalSegmentator masks (they were
        # only ever built for val/test), and requiring one here would silently cache
        # nothing at all.
        lung = np.load(mp) if has_mask else np.zeros(0, np.uint8)
        D, H, W = thin.shape
        # 6/50 test volumes are narrower than the 256 crop these networks were built
        # for (their patch-embedding size is baked into the weights, so the crop
        # cannot shrink). Reflect-pad, run, then crop back - truncating the tile
        # instead would silently feed the network a wrong-shaped input.
        py, px = max(a.crop - H, 0), max(a.crop - W, 0)
        if py or px:
            thin = np.pad(thin, ((0, 0), (0, py), (0, px)), mode='reflect')
            thick = np.pad(thick, ((0, 0), (0, py), (0, px)), mode='reflect')
            if has_mask: lung = np.pad(lung, ((0, 0), (0, py), (0, px)), mode='constant')
        Hp, Wp = thin.shape[1:]
        # Seed with cubic interpolation so the slices the network cannot predict
        # (the first/last 3 of every window, and the volume ends) are never empty;
        # every voxel the network DOES predict overwrites the seed.
        from scipy.interpolate import CubicSpline
        pz = 5 * np.arange(thick.shape[0], dtype=float) + a.z_shift
        rec = CubicSpline(pz, thick, axis=0, bc_type='natural')(np.arange(D, dtype=float)
                                                                ).astype(np.float32)
        acc = np.zeros_like(rec); cnt = np.zeros_like(rec)   # per-VOXEL weight: the
        # y/x tiles overlap in the final strip, so a per-slice counter would
        # double-count exactly there and darken a band down the edge of the volume.
        ys = list(range(0, max(Hp - a.crop, 0) + 1, a.crop)) or [0]
        xs = list(range(0, max(Wp - a.crop, 0) + 1, a.crop)) or [0]
        if ys[-1] + a.crop < Hp: ys.append(Hp - a.crop)
        if xs[-1] + a.crop < Wp: xs.append(Wp - a.crop)
        # Stride 1 recomputes every output slice about n_out/r times - six for CTHNet,
        # which is where 200 forward passes per case came from. A window emits n_out thin
        # slices and advances r per thick step, so stride n_out//r tiles exactly and
        # n_out//(2r) leaves half-overlap for blending. Pure efficiency: the model, the
        # windows and the outputs are unchanged, only the redundancy is removed.
        n_out = WINDOW[a.net][1]
        # Measured on four public volumes: stride 1 -> 3 -> 6 costs 34.10 -> 33.97 ->
        # 33.72 dB while wall clock goes 116 -> 50 -> 35 s. Half-overlap keeps the loss
        # inside a rounding error of the differences this paper argues about, and turns
        # a 24 h cohort pass into 9 h.
        stride = a.z_stride or max(n_out // (2 * 5), 1)
        zs = list(range(0, max(thick.shape[0] - c_z, 0) + 1, stride))
        if zs and zs[-1] != thick.shape[0] - c_z and thick.shape[0] - c_z > 0:
            zs.append(thick.shape[0] - c_z)          # never leave the top of the volume out
        for z_s in zs:
            z_e = z_s + c_z
            m_s, m_e = window(z_s)
            if m_e > D: continue
            for y_s in ys:
                for x_s in xs:
                    xb = wrap(torch.from_numpy(
                        thick[z_s:z_e, y_s:y_s + a.crop, x_s:x_s + a.crop])).cuda()
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.amp):
                        p = net(xb)
                    p = unwrap(p.float()).cpu().numpy()
                    acc[m_s:m_e, y_s:y_s + a.crop, x_s:x_s + a.crop] += p
                    cnt[m_s:m_e, y_s:y_s + a.crop, x_s:x_s + a.crop] += 1.0
        hit = cnt > 0
        rec[hit] = acc[hit] / cnt[hit]
        if py or px:                                  # back to the true field of view
            rec = rec[:, :H, :W]; thin = thin[:, :H, :W]
            if has_mask: lung = lung[:, :H, :W]
        if a.save_recon:
            os.makedirs(a.save_recon, exist_ok=True)
            np.save(os.path.join(a.save_recon, f.replace('_thin', '_base')),
                    rec.astype(np.float16))
        if not has_mask:
            print('[%2d/%d] %-14s cached (no mask -> no metrics)'
                  % (ci, len(files), f.replace('_thin.npy', '')), flush=True)
            continue
        m = lung > 0
        if m.sum() < 1000: print('[skip] %s: empty lung' % f); continue
        mse = float(np.mean((rec - thin) ** 2)); mse_l = float(np.mean((rec[m] - thin[m]) ** 2))
        dr = a.data_range
        _, smap = structural_similarity(thin, rec, data_range=dr, full=True)
        t = lambda v: torch.from_numpy(v.astype(np.float32) / 1000.)
        mt = torch.from_numpy(m)
        row = dict(case=f.replace('_thin.npy', ''), lung_voxels=int(m.sum()),
                   psnr=10 * np.log10(dr ** 2 / max(mse, 1e-8)), ssim=float(smap.mean()),
                   lung_psnr=10 * np.log10(dr ** 2 / max(mse_l, 1e-8)),
                   lung_ssim=float(smap[m].mean()),
                   laa950_ref=metrics.laa950(t(thin), mt).item(),
                   laa950_rec=metrics.laa950(t(rec), mt).item(),
                   laa910_ref=metrics.laa910(t(thin), mt).item(),
                   laa910_rec=metrics.laa910(t(rec), mt).item(),
                   p15_ref=1000 * metrics.percN(t(thin), .15, mt).item(),
                   p15_rec=1000 * metrics.percN(t(rec), .15, mt).item(),
                   p10_ref=1000 * metrics.percN(t(thin), .10, mt).item(),
                   p10_rec=1000 * metrics.percN(t(rec), .10, mt).item())
        for li, ln in enumerate(['UL_L', 'LL_L', 'UL_R', 'ML_R', 'LL_R'], 1):
            sel = torch.from_numpy(lung == li)
            ok = sel.sum() > 1000
            row['laa950_ref_' + ln] = metrics.laa950(t(thin), sel).item() if ok else float('nan')
            row['laa950_rec_' + ln] = metrics.laa950(t(rec), sel).item() if ok else float('nan')
        rows.append(row)
        print('[%2d/%d] %-14s LAA950 ref %6.2f%% rec %6.2f%%  PSNR %5.2f  lungPSNR %5.2f  SSIM %.4f'
              % (ci, len(files), row['case'], row['laa950_ref'], row['laa950_rec'],
                 row['psnr'], row['lung_psnr'], row['ssim']), flush=True)
    if not rows:
        print('\ncached %d volumes, no masks -> no metrics' % len(files)); return
    A = lambda k: np.array([x[k] for x in rows], dtype=float)
    print('\nn=%d  %s' % (len(rows), a.net))
    for k in ('psnr', 'lung_psnr', 'ssim', 'lung_ssim'):
        print('  %-10s %8.4f' % (k, A(k).mean()))
    for k, u in (('laa950', 'pp'), ('laa910', 'pp'), ('p15', 'HU'), ('p10', 'HU')):
        ref, rec = A(k + '_ref'), A(k + '_rec')
        print('  %-7s bias %+6.2f  MAE %5.2f %s  CCC %.3f'
              % (k, (rec - ref).mean(), np.abs(rec - ref).mean(), u,
                 metrics.ccc(torch.tensor(rec), torch.tensor(ref)).item()))
    if a.csv and rows:
        with open(a.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
        print('wrote', a.csv)


ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['train', 'infer'])
ap.add_argument('--net', required=True, choices=['tvsrn', 'cthnet', 'i3net'])
ap.add_argument('--root', required=True)
ap.add_argument('--ckpt', default=None)
ap.add_argument('--csv', default=None)
ap.add_argument('--steps', type=int, default=20000)
ap.add_argument('--lr', type=float, default=3e-4)
ap.add_argument('--crop', type=int, default=256)
ap.add_argument('--n', type=int, default=0)
ap.add_argument('--data_range', type=float, default=2000.0)
ap.add_argument('--save_recon', default=None,
                help='directory to cache the whole-volume reconstruction as <case>_base.npy')
ap.add_argument('--z_stride', type=int, default=0,
                help='thick slices to advance per window. 0 = auto (half-overlap). '
                     '1 was the old behaviour and recomputes each output ~6x.')
ap.add_argument('--z_shift', type=int, default=2,
                help='thin-slice offset of the thick series (2 for our re-aligned '
                     'pairs, 0 for data in the upstream convention)')
ap.add_argument('--grad_ckpt', action='store_true',
                help='enable the upstream blocks\' own gradient checkpointing')
ap.add_argument('--amp', action='store_true',
                help='bf16 autocast - needed to fit CTHNet at the upstream 256 crop '
                     'on a 16 GB card; the loss itself is still computed in fp32')
a = ap.parse_args()
if a.ckpt is None: a.ckpt = 'runs/BL_%s.pt' % a.net
torch.manual_seed(0)
(train if a.mode == 'train' else infer)(a)
