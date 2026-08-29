"""Export MoGe's DINOv2 encoder at a fixed shape and census its operators.

RFD 1167 rung 1 for this model, and rung 2 if the census is clean.

The cut is after normalization, not before. MoGe normalizes inside the module and
VoxHammer normalizes outside it, so exporting this one whole would double-apply
what `compile_hef.py` folds into the input layer.

Weights are not the measurement here. A rung-1 export is about the graph and its
operator set, both of which are fixed by the architecture.
"""
from __future__ import annotations

import argparse
import collections
import os
import sys
import tempfile

TOKEN_ROWS = 37
TOKEN_COLS = 37
PATCH = 14
REL_TOL = 1e-4

_HERE = os.path.dirname(os.path.abspath(__file__))
LINKED_GATE = os.path.join(_HERE, "..", "..", "hailo_device_ops.py")
SIBLING_GATE = os.path.join(_HERE, "..", "rf-detr-cpp", "scripts", "gate_onnx_device.py")


def default_gate():
    """The manifest link if `repo sync` wrote it, else the sibling checkout."""
    return LINKED_GATE if os.path.exists(LINKED_GATE) else SIBLING_GATE


def load_device_ops(path):
    import importlib.util

    if not os.path.exists(path):
        sys.exit("FAIL  no allowlist at %s; `repo sync` writes the link, or pass --gate" % path)
    spec = importlib.util.spec_from_file_location("rfdetr_gate", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return set(mod.DEVICE_OPS), dict(mod.KNOWN_BLOCKERS)


def self_test(gate):
    """A census that has never refused an operator has not shown it can refuse one."""
    device_ops, blockers = load_device_ops(gate)
    planted = "GridSample"
    if planted in device_ops:
        sys.exit("FAIL  %s is inside the allowlist; the control cannot plant it" % planted)
    counts = collections.Counter({"Conv": 2, planted: 1})
    outside = {op: c for op, c in counts.items() if op not in device_ops}
    if outside != {planted: 1}:
        sys.exit("FAIL  the census passed a graph carrying %s" % planted)
    print("self-test: a %s in the census is refused, and named -- %s"
          % (planted, blockers[planted]))
    return 0


def build(backbone, layers, dim_out, compat):
    import torch

    sys.path.insert(0, _HERE)
    from moge.model.modules.dinov2_encoder import DINOv2Encoder

    enc = DINOv2Encoder(backbone=backbone, intermediate_layers=layers, dim_out=dim_out)
    enc.onnx_compatible_mode = compat
    enc.eval()

    class _M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enc = enc

        def forward(self, normalized):
            f = self.enc.backbone.get_intermediate_layers(
                normalized, n=self.enc.intermediate_layers, return_class_token=True)
            return torch.stack([
                p(t.permute(0, 2, 1).unflatten(2, (TOKEN_ROWS, TOKEN_COLS)).contiguous())
                for p, (t, _) in zip(self.enc.output_projections, f)
            ], dim=1).sum(dim=1)

    return _M().eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(),
                                              "moge_encoder_device_half.onnx"))
    ap.add_argument("--gate", default=default_gate())
    ap.add_argument("--backbone", default="dinov2_vitb14")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim-out", type=int, default=256)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-compat", action="store_true",
                    help="leave onnx_compatible_mode off")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()

    if a.self_test:
        return self_test(a.gate)

    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch

    device_ops, blockers = load_device_ops(a.gate)
    net = build(a.backbone, a.layers, a.dim_out, not a.no_compat)

    h, w = TOKEN_ROWS * PATCH, TOKEN_COLS * PATCH
    x = torch.randn(1, 3, h, w)
    with torch.no_grad():
        want = net(x)

    torch.onnx.export(net, (x,), a.out, opset_version=a.opset,
                      input_names=["normalized"], output_names=["features"],
                      dynamo=False)

    sess = ort.InferenceSession(a.out, providers=["CPUExecutionProvider"])
    got = sess.run(None, {"normalized": x.numpy()})[0]
    ref = want.numpy()
    scale = float(np.abs(ref).max())
    rel = float(np.abs(got - ref).max()) / scale

    g = onnx.load(a.out).graph
    counts = collections.Counter(n.op_type for n in g.node)
    outside = {op: c for op, c in counts.items() if op not in device_ops}

    print("  %s at %dx%d, onnx_compatible_mode=%s: %d nodes, %d operators"
          % (a.backbone, h, w, not a.no_compat, len(g.node), len(counts)))
    print("  output %s, max|ref| %.3f, %.3e relative" % (tuple(ref.shape), scale, rel))
    for op, c in sorted(outside.items()):
        print("  outside DEVICE_OPS: %s x%d -- %s"
              % (op, c, blockers.get(op, "not measured; the DFC decides")))

    problems = []
    if rel > REL_TOL:
        problems.append("onnxruntime disagrees with torch by %.3e relative" % rel)
    if outside:
        problems.append("%d operator(s) outside the allowlist" % len(outside))
    if problems:
        print("\nFAIL (%d):" % len(problems))
        for p in problems:
            print("  %s" % p)
        return 1
    print("\nPASS: exports at a fixed shape, runs, and every operator is inside DEVICE_OPS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
