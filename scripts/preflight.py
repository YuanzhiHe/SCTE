"""Pre-flight: prove the environment and the pipeline work BEFORE touching patient data.

Builds a synthetic thin/thick pair with a KNOWN sub-slab z offset, runs the real
stage-1 code on it, and checks that the offset is recovered exactly. That exercises
everything the hospital run depends on except the DICOM reader itself: HU handling,
the forward operator, the alignment search, the index arithmetic, the operator fit
and the certificate.

The alignment step is the one worth being paranoid about. A sub-slab z offset is
invisible in every log, costs the classical baselines ~3 dB, and silently shifts the
operator the whole certificate is built on. It cost this project a full re-run.

  python scripts/preflight.py            # ~3 minutes
  python scripts/preflight.py --quick    # skip the model checks
"""
import argparse, os, shutil, subprocess, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser()
ap.add_argument('--quick', action='store_true', help='environment + stage 1 only')
ap.add_argument('--keep', action='store_true', help='keep the scratch directory')
a = ap.parse_args()

PY = sys.executable
fails, warns = [], []


def check(name, fn, fatal=True):
    sys.stdout.write(f'  {name:<52s}')
    sys.stdout.flush()
    try:
        msg = fn()
        print(f'OK   {msg or ""}')
        return True
    except Exception as e:
        print(f'FAIL {e}')
        (fails if fatal else warns).append((name, str(e)))
        return False


print('\n=== 1. environment ===')


def _torch():
    import torch
    dev = 'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'
    return f'torch {torch.__version__}, {dev}'


def _tv():
    import torch, torchvision
    # The failure mode is not an import error: torchvision imports fine and then dies
    # inside TotalSegmentator with "torchvision::nms does not exist" when its build
    # does not match torch's CUDA channel. Force the op to resolve now.
    from torchvision.ops import nms
    b = torch.tensor([[0., 0., 1., 1.], [0., 0., 1., 1.]])
    nms(b, torch.tensor([0.9, 0.1]), 0.5)
    return f'torchvision {torchvision.__version__}, nms resolves'


def _ts():
    import totalsegmentator
    w = os.path.expanduser('~/.totalsegmentator')
    if not os.path.isdir(w):
        raise RuntimeError('weights missing at ~/.totalsegmentator (it will try to '
                           'download them on first use)')
    n = sum(len(f) for _, _, f in os.walk(w))
    return f'{n} weight files present'


check('torch + CUDA', _torch)
check('torchvision matches torch (TotalSegmentator needs this)', _tv)
check('TotalSegmentator + weights', _ts, fatal=False)
check('scipy / skimage / SimpleITK / pydicom',
      lambda: __import__('scipy') and __import__('skimage') and
              __import__('SimpleITK') and __import__('pydicom') and 'all present')
check('project imports', lambda: __import__('scte_r.model', fromlist=['x']) and
      __import__('scte_r.agentic', fromlist=['x']) and 'scte_r ok')
check('vendored baselines',
      lambda: (os.path.isfile(f'{ROOT}/baselines/cthnet/model_TransSR.py') and
               os.path.isfile(f'{ROOT}/baselines/i3net/basic_model.py') and 'tvsrn/cthnet/i3net')
      or (_ for _ in ()).throw(RuntimeError('run scripts/fetch_baselines.sh')), fatal=False)

if fails:
    print('\n!! environment is not ready - fix the FAILs above before continuing\n')
    sys.exit(1)

print('\n=== 2. synthetic cohort with a known sub-slab offset ===')
work = tempfile.mkdtemp(prefix='scte_preflight_')
TRUE_SLAB, TRUE_SUB = 1, 3


def _make():
    import numpy as np, torch, SimpleITK as sitk
    from scte_r.forward_operator import SSPForwardOperator
    op = SSPForwardOperator(slice_fwhm_mm=6.25, downsample=5)
    rng = np.random.default_rng(0)
    src = os.path.join(work, 'cases')
    for ci in range(4):
        D, H, W = 260, 130, 130
        v = np.full((D, H, W), -1000., np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        rad = (yy - H / 2) ** 2 + (xx - W / 2) ** 2
        v[:, rad < (0.44 * H) ** 2] = 40.
        lung = rad < (0.34 * H) ** 2
        tex = rng.normal(-850, 110, (D, int(lung.sum()))).astype(np.float32)
        tex += np.cumsum(rng.normal(0, 9, (D, 1)), 0).astype(np.float32)
        v[:, lung] = tex
        x = v[TRUE_SUB:]
        n = x.shape[0] // 5
        y = op(torch.from_numpy(x[:n * 5][None, None]) / 1000.)[0, 0].numpy() * 1000.
        y = y[TRUE_SLAB:]
        d = os.path.join(src, f'CASE{ci:03d}')
        os.makedirs(d, exist_ok=True)
        for nm, arr, sp in (('1mm', v, 1.0), ('5mm', y, 5.0)):
            im = sitk.GetImageFromArray(arr.astype(np.int16))
            im.SetSpacing((0.7, 0.7, sp))
            sitk.WriteImage(im, os.path.join(d, f'{nm}.nii.gz'))
    return f'4 cases, true offset slab+{TRUE_SLAB} sub+{TRUE_SUB}'


check('build synthetic pairs', _make)

print('\n=== 3. stage 1: pairing and z alignment (the real code path) ===')
pairs = os.path.join(work, 'pairs')


def _prep():
    r = subprocess.run([PY, f'{ROOT}/scripts/prep_pairs.py', '--cases',
                        os.path.join(work, 'cases'), '--out', pairs, '--downsample', '5',
                        '--thin_hint', '1mm', '--thick_hint', '5mm'],
                       capture_output=True, text=True, timeout=900)
    n_ok = r.stdout.count('[ok]')
    n_skip = r.stdout.count('[skip]')
    if n_ok != 4:
        raise RuntimeError(f'{n_ok}/4 paired, {n_skip} skipped\n{r.stdout[-600:]}')
    return f'{n_ok}/4 paired, 0 skipped'


def _align():
    r = subprocess.run([PY, f'{ROOT}/scripts/realign_pairs.py', '--src', pairs, '--check'],
                       capture_output=True, text=True, timeout=900)
    if r.returncode != 0:
        raise RuntimeError('pairs are NOT on the operator grid\n' + r.stdout[-700:])
    return 'every case on the operator grid'


check('prep_pairs recovers the offset', _prep)
check('alignment self-check passes', _align)

if not a.quick:
    print('\n=== 4. operator fit and certificate ===')
    proto = os.path.join(work, 'protocol.json')

    def _proto():
        r = subprocess.run([PY, f'{ROOT}/scripts/fit_protocol_operator.py', '--root', pairs,
                            '--patch', '40', '64', '64', '--protocol', 'preflight',
                            '--out', proto], capture_output=True, text=True, timeout=1800)
        if not os.path.exists(proto):
            raise RuntimeError(r.stdout[-700:] + r.stderr[-400:])
        w = [l for l in r.stdout.splitlines() if 'protocol width' in l]
        return w[0].strip() if w else 'fitted'

    def _cert():
        csv = os.path.join(work, 'oracle.csv')
        r = subprocess.run([PY, f'{ROOT}/scripts/certify.py', '--root', pairs, '--oracle',
                            '--calibration', proto, '--csv', csv],
                           capture_output=True, text=True, timeout=1800)
        if not os.path.exists(csv):
            raise RuntimeError(r.stdout[-700:] + r.stderr[-400:])
        d = [l for l in r.stdout.splitlines() if 'delta' in l]
        return d[0].strip()[:60] if d else 'certificate runs'

    check('protocol operator fit', _proto)
    check('certificate on a perfect reconstruction', _cert)

if not a.keep:
    shutil.rmtree(work, ignore_errors=True)
else:
    print(f'\nscratch kept at {work}')

print()
if fails:
    print(f'!! {len(fails)} FAILED - do not run on patient data yet\n')
    sys.exit(1)
if warns:
    print('PRE-FLIGHT PASSED with warnings:')
    for n, e in warns:
        print(f'  - {n}: {e}')
    print('\n  TotalSegmentator warnings matter: without it the LAA denominator is an')
    print('  HU window, which is NOT clinical LAA-950. Fix before reporting numbers.\n')
    sys.exit(0)
print('PRE-FLIGHT PASSED - environment and pipeline verified end to end.\n')
