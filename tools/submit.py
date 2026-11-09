#!/usr/bin/env python3
"""Submit assignment 2 for grading and check results.

Run from the assignment directory (the one with src/):

    python3 tools/submit.py submit --id 你的学号      # upload src/{autograd,ops,nn,optim}.cpp
    python3 tools/submit.py status --id 你的学号      # queue position and latest results
    python3 tools/submit.py status --id 你的学号 -n 5 # show the latest 5 submissions
    python3 tools/submit.py submit --id 你的学号 --nickname 昵称   # also set the leaderboard name

The first accepted submission registers the student id; the leaderboard shows only
the assigned number (S0001, ...) and the nickname, never the student id.

Only the four files above are uploaded; everything else is the teacher's copy on
the grading server. Leaderboard: https://coideas.me/ml-assignment-2/
"""
import argparse
import http.client
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://coideas.me/ml-assignment-2/"
FILES = ("autograd.cpp", "ops.cpp", "nn.cpp", "optim.cpp")
STATUS = {"queued": "排队中", "running": "评测中", "done": "已评测", "error": "系统故障（不计次数）"}
HTTP_TEXT = {413: "提交内容过大（每个文件最多 512 KB）", 429: "请求太频繁，请过一分钟再试"}
MAX_FILE_BYTES = 512 * 1024


def call(base, path, data=None):
    req = urllib.request.Request(urllib.parse.urljoin(base, path),
                                 data=None if data is None else json.dumps(data, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("message")
        except Exception:  # not the site's JSON: an error page of the proxy in front of it
            msg = HTTP_TEXT.get(e.code) or ("评测服务器暂时不可用，请稍后重试" if e.code >= 500 else None)
        sys.exit(f"失败：{msg or e}")
    except (urllib.error.URLError, OSError) as e:  # OSError: timeout or reset while reading the reply
        sys.exit(f"连不上评测服务器：{getattr(e, 'reason', e) or type(e).__name__}，请稍后重试")
    except (http.client.HTTPException, ValueError):  # reply cut off or not JSON
        sys.exit("评测服务器的回复不完整，请稍后重试")


def pct(v):
    return "—" if v is None else f"{v * 100:.2f}%"


def eta_text(minutes):
    return "马上开始" if not minutes else f"预计约 {minutes} 分钟后开始"


def show(item):
    print(f"\n#{item['id']}  {item['received_at']}  {STATUS.get(item['status'], item['status'])}"
          + ("  ★ 榜单采用" if item.get("best") else ""))
    if "queue_position" in item:
        print(f"  排队第 {item['queue_position']} 位，{eta_text(item['eta_minutes'])}")
    rep = item.get("report") or {}
    if rep.get("system_error"):
        print("  " + rep["system_error"])
    for line in rep.get("check_errors", []):
        print("  代码检查未通过：" + line)
    for target, err in (rep.get("compile_errors") or {}).items():
        print(f"  编译/链接失败 [{target}]:\n    " + err.strip().replace("\n", "\n    "))
    t = rep.get("tests")
    if t:
        print(f"  公开测试 {t['public_pass']}/{t['public_total']}，隐藏测试 {t['hidden_pass']}/{t['hidden_total']}"
              + "（" + "，".join(f"{g} {p}/{n}" for g, (p, n) in t["hidden_groups"].items()) + "）")
        if t["public_failed"]:
            print("  未通过的公开测试：" + "、".join(t["public_failed"]))
    for key, m in (rep.get("models") or {}).items():
        parts = "，".join(f"{p} {v}" for p, v in (m.get("parts") or {}).items())
        print(f"  {key.upper():5} 一致性 {pct(m.get('ratio'))}（{parts}），得分 {m.get('score')}")
        for line in m.get("public_failed") or []:
            print("    公开 " + line)
        if m.get("error"):
            print("    " + m["error"].strip().replace("\n", "\n    "))
    s = rep.get("score")
    if s:
        print(f"  得分：测试 {s['tests']} + 一致性 {s['consistency']} = 总分 {s['total']} / 100")


def main():
    ap = argparse.ArgumentParser(description="作业二提交与查询")
    ap.add_argument("command", choices=["submit", "status"])
    ap.add_argument("--id", required=True, help="学号")
    ap.add_argument("--src", default="src", help="源码目录（默认 src）")
    ap.add_argument("--nickname", help="排行榜上显示的昵称（最多 20 字，可随提交修改；默认“匿名同学”）")
    ap.add_argument("-n", type=int, default=1, help="status 显示最近几次提交（默认 1）")
    ap.add_argument("--url", default=URL, help=argparse.SUPPRESS)
    args = ap.parse_args()
    base = args.url if args.url.endswith("/") else args.url + "/"

    if args.command == "submit":
        src = Path(args.src)
        missing = [f for f in FILES if not (src / f).is_file()]
        if missing:
            sys.exit(f"找不到 {', '.join(str(src / f) for f in missing)}；请在作业目录下运行")
        files = {}
        for f in FILES:
            try:
                files[f] = (src / f).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                sys.exit(f"{src / f} 不是 UTF-8 编码，请用 UTF-8 重新保存")
        big = [f for f, text in files.items() if len(text.encode()) > MAX_FILE_BYTES]
        if big:
            sys.exit(f"文件过大（每个最多 512 KB）：{', '.join(big)}")
        data = {"student_id": args.id, "files": files}
        if args.nickname is not None:
            data["nickname"] = args.nickname
        r = call(base, "api/submit", data)
        print(f"已受理：提交编号 #{r['submission_id']}，时间 {r['received_at']}")
        hint = "；改昵称：提交时加 --nickname 昵称" if r.get("nickname_editable") else ""
        print(f"排行榜上显示为 {r['public_id']} {r['nickname']}（不显示学号{hint}）")
        if r["replaced"]:
            print("已替换排队中的上一份提交 " + "、".join(f"#{i}" for i in r["replaced"]))
        print(f"今天已用 {r['today_used']}/{r['daily_limit']} 次；排队第 {r['queue_position']} 位，"
              f"{eta_text(r['eta_minutes'])}，评测通常只要几秒。")
        print(f"查询结果：python3 tools/submit.py status --id {args.id}")
        return

    r = call(base, "api/status?" + urllib.parse.urlencode({"student_id": args.id}))
    print(f"{r['public_id']} {r['nickname']}  今天已用 {r['today_used']}/{r['daily_limit']} 次；"
          f"队列：评测中 {r['queue']['running']}，排队 {r['queue']['queued']}")
    if not r["items"]:
        print("还没有提交记录。")
    for item in r["items"][:max(1, args.n)]:
        show(item)


if __name__ == "__main__":
    main()
