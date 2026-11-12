#!/usr/bin/env python3
"""Fixed-seed consistency check (provided by the course staff, do not modify).

    python3 -I tools/check_consistency.py build/consistency tests/consistency_expected.json

For every (model, seed) in the expected file, runs `build/consistency MODEL SEED`
(5 steps of forward / backward / SGD from a fixed initialisation and fixed random
input) and compares its trace with the reference values (PyTorch float32 runs):
the loss of every step, the gradients after the first backward() and the
parameters after the last SGD step.

Every compared number gets a relative error |yours - ref| / scale, where scale
is |ref| for a loss and the root-mean-square of the reference tensor for a
gradient or parameter (so values near zero are not blown up). A run that crashes,
times out or prints the wrong shapes counts as an infinite error.

Results are the maximum and the median of these errors:

    完全正确  median <= 1e-6 and max <= 1e-5
    大体正确  max <= 1e-3 (correct, but precision is lost somewhere)
    有错误    max >  1e-3
    未完成    a run gave no usable output

On the server every hidden seed of a model is banded on its own; a model's credit is
100 x the mean weight of its seeds' bands (see README 评分).
Grading uses only the unpublished seeds on the server; the public seeds here
are for checking your own work and do not count.
"""
import json
import math
import os
import statistics
import subprocess
import sys

TIMEOUT = 5  # seconds per run, same as the grading server
EXACT_MEDIAN, EXACT_MAX, ROUGH_MAX = 1e-6, 1e-5, 1e-3
BANDS = ("完全正确", "大体正确", "有错误", "未完成")


def _number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _rel(g, r, scale):
    return abs(g - r) / scale if _number(g) else math.inf


def _tensor_errors(got, ref, what):
    errors = []
    for name, r in ref.items():
        g = (got or {}).get(name) if isinstance(got, dict) else None
        rms = r["norm"] / math.sqrt(max(1, r["n"])) or 1e-30
        labels = [f"{what} {name}（范数）"] + [f"{what} {name}（第 {k + 1} 个抽样元素）" for k in range(len(r["at"]))]
        if not (isinstance(g, dict) and g.get("n") == r["n"] and isinstance(g.get("at"), list)
                and len(g["at"]) == len(r["at"])):
            errors += [(label, math.inf) for label in labels]
            continue
        values = [_rel(g.get("norm"), r["norm"], max(abs(r["norm"]), 1e-30))]
        values += [_rel(x, y, rms) for x, y in zip(g["at"], r["at"])]
        errors += list(zip(labels, values))
    return errors


def labeled_errors(got, ref):
    """(what was compared, relative error) for every compared number of one run."""
    steps = len(ref["loss"])
    loss = got.get("loss") if isinstance(got, dict) and isinstance(got.get("loss"), list) else []
    loss = loss[:steps] + [None] * (steps - len(loss))
    out = [(f"第 {i + 1} 步的 loss", _rel(g, r, max(abs(r), 1e-30)))
           for i, (g, r) in enumerate(zip(loss, ref["loss"]))]
    out += _tensor_errors(got.get("grad") if isinstance(got, dict) else None, ref["grad"], "第 1 步后的梯度")
    out += _tensor_errors(got.get("param") if isinstance(got, dict) else None, ref["param"], f"第 {steps} 步后的参数")
    return out


def errors(got, ref):
    """Relative error of every compared number of one run (inf where unusable)."""
    return [e for _, e in labeled_errors(got, ref)]


def summarize(errs):
    """{"max", "median", "band"}; max/median are None when infinite (no usable output)."""
    if not errs:
        return {"max": None, "median": None, "band": BANDS[3]}
    worst, mid = max(errs), statistics.median(errs)
    if not math.isfinite(worst):
        band = BANDS[3]
    elif worst <= EXACT_MAX and mid <= EXACT_MEDIAN:
        band = BANDS[0]
    elif worst <= ROUGH_MAX:
        band = BANDS[1]
    else:
        band = BANDS[2]
    finite = lambda v: v if math.isfinite(v) else None
    return {"max": finite(worst), "median": finite(mid), "band": band}


def fmt(v):
    return "—" if v is None else ("0" if v == 0 else f"{v:.1e}")


def parse(stdout):
    lines = [l for l in stdout.splitlines() if l.startswith("{")]
    try:
        return json.loads(lines[-1]) if lines else None
    except ValueError:
        return None


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    exe, expected = sys.argv[1], json.load(open(sys.argv[2], encoding="utf-8"))
    if not os.access(exe, os.X_OK):
        sys.exit(f"找不到 {exe}，先运行 make")
    every, per_model = [], {}
    for ref in expected["runs"]:
        err = ""
        try:
            p = subprocess.run([exe, ref["model"], str(ref["seed"])], capture_output=True, text=True, timeout=TIMEOUT)
            got = parse(p.stdout) if p.returncode == 0 else None
            if p.returncode != 0:
                err = (p.stderr.strip().splitlines() or [f"退出码 {p.returncode}"])[-1]
        except subprocess.TimeoutExpired:
            got, err = None, f"超时（超过 {TIMEOUT} 秒）"
        labeled = labeled_errors(got, ref)
        errs = [e for _, e in labeled]
        every += errs
        per_model.setdefault(ref["model"], []).extend(errs)
        s = summarize(errs)
        print(f"{ref['model']:4s} seed {ref['seed']}: 最大误差 {fmt(s['max'])}  中位数 {fmt(s['median'])}  {s['band']}"
              + (f"  （{err}）" if err else ""))
        if got is not None and s["band"] != BANDS[0]:
            where, worst = max(labeled, key=lambda item: item[1])
            print(f"    误差最大的一项：{where}，相对误差 {fmt(worst if math.isfinite(worst) else None)}")
    for model, errs in per_model.items():
        s = summarize(errs)
        print(f"{model.upper():5} 最大误差 {fmt(s['max'])}  中位数 {fmt(s['median'])}  {s['band']}")
    s = summarize(every)
    print(f"总计（公开种子，不计分）：最大误差 {fmt(s['max'])}，中位数 {fmt(s['median'])}，{s['band']}")
    sys.exit(0 if s["band"] in BANDS[:2] else 1)


if __name__ == "__main__":
    main()
