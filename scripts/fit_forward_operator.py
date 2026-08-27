"""Fit A_theta on real pairs and report how much of the thick series it explains."""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scte_r.datasets import PreparedPairDataset
from scte_r.forward_learned import LearnedOperator
from scte_r.forward_operator import SSPForwardOperator

ap = argparse.ArgumentParser()
ap.add_argument('--root', required=True); ap.add_argument('--val_root', default=None)
ap.add_argument('--out', required=True)
ap.add_argument('--downsample', type=int, default=5)
ap.add_argument('--patch', type=int, nargs=3, default=[60, 128, 128])
ap.add_argument('--epochs', type=int, default=400); ap.add_argument('--lr', type=float, default=3e-3)
ap.add_argument('--nominal', type=float, default=6.25, help='baseline width to beat')
ap.add_argument('--no_correction', action='store_true')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()

ds = PreparedPairDataset(a.root, patch=tuple(a.patch), downsample=a.downsample, random_crop=True)
va = PreparedPairDataset(a.val_root or a.root, patch=tuple(a.patch), downsample=a.downsample)
op = LearnedOperator(a.downsample, correction=not a.no_correction).to(a.device)
base = SSPForwardOperator(slice_fwhm_mm=a.nominal, downsample=a.downsample)
opt = torch.optim.Adam(op.parameters(), lr=a.lr)

def unexplained(o, dset, n=15):
    """||A(x) - y|| / ||y - mean(y)||, the same quantity rho reports."""
    tot = []
    with torch.no_grad():
        for i in range(min(n, len(dset))):
            y, x, _ = dset[i]; y = y[None].to(a.device); x = x[None].to(a.device)
            r = o(x) - y
            tot.append(float(r.norm() / (y - y.mean()).norm()))
    return float(np.mean(tot))

print('before: fixed Gaussian w=%.2f -> unexplained %.4f' % (a.nominal, unexplained(base, va)))
for ep in range(a.epochs):
    y, x, _ = ds[ep % len(ds)]
    y = y[None].to(a.device); x = x[None].to(a.device)
    loss = ((op(x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if (ep + 1) % 50 == 0:
        print('ep%4d  train mse %.6f | val unexplained %.4f | eff.FWHM %.2f mm'
              % (ep + 1, float(loss), unexplained(op, va), op.effective_fwhm()), flush=True)

u_before, u_after = unexplained(base, va), unexplained(op, va)
print('\nunexplained residual: %.4f -> %.4f  (%.0f%% of it removed)'
      % (u_before, u_after, 100 * (1 - u_after / u_before)))
print('fitted SSP basis (width mm : weight):')
for w, c in zip(op.widths, op.coeffs.detach().cpu().tolist()):
    print('   %5.1f : %.4f %s' % (w, c, '#' * int(60 * c)))
print('effective FWHM %.2f mm' % op.effective_fwhm())
torch.save({'state_dict': op.state_dict(), 'widths': op.widths,
            'downsample': a.downsample, 'correction': not a.no_correction,
            'unexplained_before': u_before, 'unexplained_after': u_after}, a.out)
print('wrote', a.out)
