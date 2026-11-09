#!/usr/bin/env python3
"""Fixed-seed consistency check (provided by the course staff, do not modify).

    python3 -I tools/check_consistency.py build/consistency tests/consistency_expected.json

For every (model, seed) in the expected file, runs `build/consistency MODEL SEED`
and compares its trace with the reference implementation's values. Each run has
three parts, each worth the same:

    forward   loss of the first step
    backward  gradients of every parameter after the first backward()
    training  losses of the later steps and parameters after the last SGD step

Values match when |yours - ref| <= RTOL * |ref| + ATOL * scale, where scale is 1
for losses and the root-mean-square of the reference tensor otherwise.
The grading server runs the same comparison with more (unpublished) seeds.
"""
import json
import math
import os
import subprocess
import sys

RTOL, ATOL = 1e-3, 1e-3
TIMEOUT = 5  # seconds per run, same as the grading server
PARTS = ("forward", "backward", "training")


def close(a, b, scale=1.0):
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or not math.isfinite(a):
        return False
    return abs(a - b) <= RTOL * abs(b) + ATOL * scale


def tensors_match(got, ref):
    """got/ref: {name: {"n", "norm", "at"}} -> list of mismatching names."""
    bad = []
    for name, r in ref.items():
        g = (got or {}).get(name)
        rms = r["norm"] / math.sqrt(max(1, r["n"]))
        ok = (isinstance(g, dict) and g.get("n") == r["n"] and close(g.get("norm"), r["norm"], rms)
              and isinstance(g.get("at"), list) and len(g["at"]) == len(r["at"])
              and all(close(x, y, rms) for x, y in zip(g["at"], r["at"])))
        if not ok:
            bad.append(name)
    return bad


def compare(got, ref):
    """Returns {"forward": bool, "backward": bool, "training": bool, "detail": str}."""
    if not isinstance(got, dict):
        return {**{p: False for p in PARTS}, "detail": "没有输出"}
    loss = got.get("loss") if isinstance(got.get("loss"), list) else []
    loss += [None] * (len(ref["loss"]) - len(loss))
    fwd = close(loss[0], ref["loss"][0])
    bad_grad = tensors_match(got.get("grad"), ref["grad"])
    bad_loss = [i + 1 for i in range(1, len(ref["loss"])) if not close(loss[i], ref["loss"][i])]
    bad_param = tensors_match(got.get("param"), ref["param"])
    detail = []
    if not fwd:
        detail.append(f"第 1 步 loss {loss[0]} ≠ {ref['loss'][0]:.6g}")
    if bad_grad:
        detail.append("梯度不一致: " + ", ".join(bad_grad))
    if bad_loss:
        detail.append("loss 不一致的步: " + ", ".join(map(str, bad_loss)))
    if bad_param:
        detail.append("最终参数不一致: " + ", ".join(bad_param))
    return {"forward": fwd, "backward": not bad_grad, "training": not bad_loss and not bad_param,
            "detail": "; ".join(detail)}


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
    total = passed = 0
    for ref in expected["runs"]:
        try:
            p = subprocess.run([exe, ref["model"], str(ref["seed"])], capture_output=True, text=True, timeout=TIMEOUT)
            got = parse(p.stdout) if p.returncode == 0 else None
            err = p.stderr.strip().splitlines()[-1:] if p.returncode != 0 else []
        except subprocess.TimeoutExpired:
            got, err = None, [f"超时（超过 {TIMEOUT} 秒）"]
        res = compare(got, ref)
        ok = sum(res[k] for k in PARTS)
        total += len(PARTS)
        passed += ok
        marks = " ".join(f"{k}={'ok' if res[k] else 'FAIL'}" for k in PARTS)
        print(f"{ref['model']:4s} seed {ref['seed']}: {marks}")
        if res["detail"] or err:
            print("    " + "; ".join(err + ([res["detail"]] if res["detail"] else [])))
    print(f"{passed}/{total} 项一致")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
