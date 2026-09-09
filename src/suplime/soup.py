# MIT License
#
# Copyright (c) 2026 Re:WayAI
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Checkpoint soup: uniform average of the k best validation checkpoints of one run.

    suplime-soup runs/suplime/checkpoints -k 5 -o runs/suplime/suplime_avg5.ckpt
    suplime-soup a.ckpt b.ckpt c.ckpt -o avg.ckpt

Checkpoints are the ``{epoch:02d}-{DER:.4f}.ckpt`` files written by ``suplime-train``; the
best ``k`` are the lowest DER. Float tensors are averaged, other buffers copied from the
first checkpoint, optimizer/scheduler/callback state dropped. The output is a plain
pyannote model checkpoint: ``Model.from_pretrained(out)`` loads it and
``tools/convert_checkpoints.py`` turns it into the Hub layout.
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List, Sequence

import torch

# `-vN` appears when Lightning refuses to overwrite an existing name — a preempted run that
# resumes mid-epoch and re-validates to the same DER writes 05-0.1234-v1.ckpt. Those are real
# checkpoints; ignoring them silently averaged fewer than -k asked for.
_CKPT_RE = re.compile(r"^(\d+)-(\d+\.\d+)(?:-v(\d+))?\.ckpt$")


def best_checkpoints(ckpt_dir: Path, k: int) -> List[Path]:
    """The ``k`` lowest-DER ``EE-D.DDDD.ckpt`` files in ``ckpt_dir``."""
    if k < 2:
        raise SystemExit(f"-k must be at least 2, got {k}")
    best = {}
    for p in Path(ckpt_dir).glob("*.ckpt"):
        m = _CKPT_RE.match(p.name)
        if not m:
            continue
        der, epoch, version = float(m.group(2)), int(m.group(1)), int(m.group(3) or 0)
        # one entry per (epoch, DER); the highest -vN is the last one written
        key = (epoch, m.group(2))
        if key not in best or version > best[key][0]:
            best[key] = (version, der, p)
    scored = [(der, p) for _, der, p in best.values()]
    if len(scored) < 2:
        raise SystemExit(f"need at least 2 scored checkpoints in {ckpt_dir}, found {len(scored)}")
    if len(scored) < k:
        print(f"warning: -k {k} but only {len(scored)} scored checkpoints in {ckpt_dir}; "
              f"averaging {len(scored)}", flush=True)
    return [p for _, p in sorted(scored)[:k]]


def average_checkpoints(paths: Sequence[Path], out: Path) -> dict:
    """Average ``paths`` into ``out``; returns the written checkpoint dict.

    Lightning checkpoints are pickles and are read with ``weights_only=False`` (they carry
    hyper-parameters and the architecture pointer, not just tensors), so loading one runs
    whatever it was built to run: only pass checkpoints you produced or otherwise trust.
    """
    paths = [Path(p) for p in paths]
    template = torch.load(paths[0], map_location="cpu", weights_only=False)
    sds = [template["state_dict"]] + [
        torch.load(p, map_location="cpu", weights_only=False)["state_dict"] for p in paths[1:]
    ]
    keys = list(sds[0])
    hp = template.get("hyper_parameters")
    for sd, p in zip(sds[1:], paths[1:]):
        if list(sd) != keys:
            missing = set(keys) ^ set(sd)
            raise SystemExit(f"state_dict keys of {p} differ from {paths[0]} "
                             f"({len(missing)} key(s) differ, e.g. {sorted(missing)[:3]})")
        # names matching is not enough: a shape mismatch with a singleton dimension would
        # broadcast into a silently wrong average that only fails at load time
        for k in keys:
            if sd[k].shape != sds[0][k].shape:
                raise SystemExit(f"{k} is {tuple(sd[k].shape)} in {p} but "
                                 f"{tuple(sds[0][k].shape)} in {paths[0]}; different runs?")
    # `template` carries checkpoint[0]'s hyper-parameters and architecture pointer into the
    # output, so souping across runs would stamp run A's metadata onto run B's weights
    for p in paths[1:]:
        other = torch.load(p, map_location="cpu", weights_only=False).get("hyper_parameters")
        if other != hp:
            raise SystemExit(f"hyper-parameters of {p} differ from {paths[0]}; "
                             f"these checkpoints are from different runs, refusing to average")
    avg = {}
    for k in keys:
        t0 = sds[0][k]
        if torch.is_floating_point(t0):
            avg[k] = (sum(sd[k].double() for sd in sds) / len(sds)).to(t0.dtype)
        else:
            avg[k] = t0.clone()
    template["state_dict"] = avg
    for key in ("optimizer_states", "lr_schedulers", "callbacks", "loops"):
        template.pop(key, None)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(template, out)
    return template


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="a checkpoints/ directory or explicit .ckpt files")
    ap.add_argument("-k", type=int, default=5, help="how many best checkpoints to average (directory input)")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if len(a.inputs) == 1 and Path(a.inputs[0]).is_dir():
        paths = best_checkpoints(Path(a.inputs[0]), a.k)
    else:
        paths = [Path(p) for p in a.inputs]
    for p in paths:
        print("  ", p, file=sys.stderr)
    average_checkpoints(paths, a.out)
    print(f"averaged {len(paths)} checkpoints -> {a.out}")


if __name__ == "__main__":
    main()
