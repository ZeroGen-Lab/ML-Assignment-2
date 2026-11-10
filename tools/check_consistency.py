#!/usr/bin/env python3
"""Fixed-seed consistency check (provided by the course staff, do not modify).

    python3 -I tools/check_consistency.py build/consistency tests/consistency_expected.json

For every (model, seed) in the expected file, runs `build/consistency MODEL SEED`
(5 steps of forward / backward / SGD from a fixed initialisation and fixed random
input) and compares its trace with the reference implementation's values:
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

The grading server runs the same comparison with more (unpublished) seeds.
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


def _tensor_errors(got, ref):
    errors = []
    for name, r in ref.items():
        g = (got or {}).get(name) if isinstance(got, dict) else None
        rms = r["norm"] / math.sqrt(max(1, r["n"])) or 1e-30
        count = 1 + len(r["at"])
        if not (isinstance(g, dict) and g.get("n") == r["n"] and isinstance(g.get("at"), list)
                and len(g["at"]) == len(r["at"])):
            errors += [math.inf] * count
            continue
        errors.append(_rel(g.get("norm"), r["norm"], max(abs(r["norm"]), 1e-30)))
        errors += [_rel(x, y, rms) for x, y in zip(g["at"], r["at"])]
    return errors


def errors(got, ref):
    """Relative error of every compared number of one run (inf where unusable)."""
    loss = got.get("loss") if isinstance(got, dict) and isinstance(got.get("loss"), list) else []
    loss = loss[:len(ref["loss"])] + [None] * (len(ref["loss"]) - len(loss))
    out = [_rel(g, r, max(abs(r), 1e-30)) for g, r in zip(loss, ref["loss"])]
    out += _tensor_errors(got.get("grad") if isinstance(got, dict) else None, ref["grad"])
    out += _tensor_errors(got.get("param") if isinstance(got, dict) else None, ref["param"])
    return out


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
        errs = errors(got, ref)
        every += errs
        per_model.setdefault(ref["model"], []).extend(errs)
        s = summarize(errs)
        print(f"{ref['model']:4s} seed {ref['seed']}: 最大误差 {fmt(s['max'])}  中位数 {fmt(s['median'])}  {s['band']}"
              + (f"  （{err}）" if err else ""))
    for model, errs in per_model.items():
        s = summarize(errs)
        print(f"{model.upper():5} 最大误差 {fmt(s['max'])}  中位数 {fmt(s['median'])}  {s['band']}")
    s = summarize(every)
    print(f"总计：最大误差 {fmt(s['max'])}，中位数 {fmt(s['median'])}，{s['band']}")
    sys.exit(0 if s["band"] in BANDS[:2] else 1)


if __name__ == "__main__":
    main()
