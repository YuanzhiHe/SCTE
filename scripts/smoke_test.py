"""End-to-end smoke test: no data, no GPU required.
Builds SCTE-R, runs forward + loss + backward on a synthetic batch, checks
shapes and that the quantitative-fidelity losses are finite. Run:  python scripts/smoke_test.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch.utils.data import DataLoader
from scte_r.model import SCTE_R
from scte_r.losses import SCTERLoss
from scte_r.datasets import SyntheticPairedDataset
from scte_r import metrics


def main():
    torch.manual_seed(0)
    ds = SyntheticPairedDataset(n=4, shape=(20, 32, 32), downsample=5)
    dl = DataLoader(ds, batch_size=2)
    model = SCTE_R(downsample=5, base_ch=16, n_blocks=3)
    loss_fn = SCTERLoss(model.op)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    y, x, kid = next(iter(dl))
    assert y.shape[2] * 5 == x.shape[2], "z-downsample factor mismatch"
    out = model(y, kid)
    assert out["x_hat"].shape == x.shape, f"recon shape {out['x_hat'].shape} != {x.shape}"
    assert out["reliability"].shape[0] == x.shape[0]

    loss, logs = loss_fn(out, x, y)
    loss.backward()
    opt.step()

    assert torch.isfinite(loss), "loss not finite"
    n_grad = sum(p.grad is not None for p in model.parameters())
    laa_err = (metrics.laa950(out["x_hat"]) - metrics.laa950(x)).abs().item()

    print("[smoke] input thick :", tuple(y.shape))
    print("[smoke] recon 1mm   :", tuple(out["x_hat"].shape))
    print("[smoke] reliability :", out["reliability"].detach().tolist())
    print("[smoke] loss terms  :", {k: round(v, 3) for k, v in logs.items()})
    print("[smoke] params w/ grad:", n_grad)
    print("[smoke] |LAA err| (untrained): %.2f pp" % laa_err)
    print("[smoke] PASS")


if __name__ == "__main__":
    main()
