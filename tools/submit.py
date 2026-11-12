#!/usr/bin/env python3
"""Submit assignment 2 for grading and check results.

Run from the assignment directory (the one with src/):

    python3 tools/submit.py submit --id 你的学号      # upload src/{autograd,ops,nn,optim}.cpp
    python3 tools/submit.py status --id 你的学号      # queue position and results
    python3 tools/submit.py submit --id 你的学号 --nickname 昵称   # also set the leaderboard name

Each student has 3 submissions for the whole assignment. Before uploading, submit
builds and runs the public consistency check here (like `make check`), shows the
result and asks for confirmation. Not counted: code that fails the source check
or does not compile/link on the server, a resubmission of identical code (e.g.
after a lost reply), a queued submission replaced by a newer one, system errors,
and anything refused before it was accepted.

The first accepted submission registers the student id; the leaderboard shows only
the assigned number (S0001, ...) and the nickname, never the student id.

Only the four files above are uploaded; everything else is the teacher's copy on
the grading server. Leaderboard: https://coideas.me/ml-assignment-2/
"""
import argparse
import http.client
import json
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

URL = "https://coideas.me/ml-assignment-2/"
FILES = ("autograd.cpp", "ops.cpp", "nn.cpp", "optim.cpp")
STATUS = {"queued": "排队中", "running": "评测中", "done": "已评测", "error": "系统故障（不计次数）",
          "superseded": "已被新提交替换（不计次数）"}
HTTP_TEXT = {413: "提交内容过大（每个文件最多 512 KB）", 429: "请求太频繁，请过一分钟再试"}
MAX_FILE_BYTES = 512 * 1024


class NotFound(Exception):
    pass


def call(base, path, data=None, not_found_ok=False):
    req = urllib.request.Request(urllib.parse.urljoin(base, path),
                                 data=None if data is None else json.dumps(data, ensure_ascii=False).encode(),
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    sent = " 如果刚才的上传其实已被受理，用 status 可以看到；重新提交同样的代码不会重复计次。" if data else ""
    try:
        with urllib.request.urlopen(req, timeout=60) as res:
            return json.load(res)
    except urllib.error.HTTPError as e:
        try:
            msg = json.load(e).get("message")
        except Exception:  # not the site's JSON: an error page of the proxy in front of it
            msg = HTTP_TEXT.get(e.code) or ("评测服务器暂时不可用，请稍后重试" if e.code >= 500 else None)
        if e.code == 404 and not_found_ok:
            raise NotFound(msg)
        sys.exit(f"失败：{msg or e}" + (sent if e.code >= 500 else ""))
    except (urllib.error.URLError, OSError) as e:  # OSError: timeout or reset while reading the reply
        sys.exit(f"连不上评测服务器：{getattr(e, 'reason', e) or type(e).__name__}，请稍后重试。" + sent)
    except (http.client.HTTPException, ValueError):  # reply cut off or not JSON
        sys.exit("评测服务器的回复不完整，请稍后重试。" + sent)


def err(v):
    return "—" if v is None else ("0" if v == 0 else f"{v:.1e}")


def eta_text(minutes):
    return "马上开始" if not minutes else f"预计约 {minutes} 分钟后开始"


def show(item):
    status = STATUS.get(item["status"], item["status"])
    if item["status"] == "done" and not item.get("counted"):
        status = "未通过代码检查或编译（不计次数）"
    print(f"\n#{item['id']}  {item['received_at']}  {status}" + ("  ★ 榜单采用" if item.get("best") else ""))
    if "queue_position" in item:
        print(f"  排队第 {item['queue_position']} 位，{eta_text(item['eta_minutes'])}")
    rep = item.get("report") or {}
    if rep.get("system_error"):
        print("  " + rep["system_error"])
    for line in rep.get("check_errors", []):
        print("  代码检查未通过：" + line)
    for target, text in (rep.get("compile_errors") or {}).items():
        print(f"  编译/链接失败 [{target}]:\n    " + text.strip().replace("\n", "\n    "))
    e, c = rep.get("errors"), rep.get("credit") or {}
    if c:
        print(f"  credit（隐藏种子，计分）：{c.get('total')} / 300")
    if e:
        print(f"  三个模型合计：最大误差 {err(e['max'])}，误差中位数 {err(e['median'])}")
    for key, m in (rep.get("models") or {}).items():
        print(f"  {key.upper():5}（隐藏种子）最大误差 {err(m.get('max'))}  中位数 {err(m.get('median'))}  {m.get('band')}"
              + (f"  credit {c[key]}" if key in c else ""))
        for run in m.get("public") or []:
            print(f"    公开 seed {run['seed']}（不计分）: 最大误差 {err(run['max'])}  中位数 {err(run['median'])}"
                  + (f"  （{run['error']}）" if run.get("error") else ""))
        if m.get("error"):
            print("    隐藏种子运行出错：" + m["error"].strip().replace("\n", "\n    "))
    t = rep.get("tests")
    if t:
        print(f"  （仅供参考，不计分）公开测试 {t['public_pass']}/{t['public_total']}，"
              f"隐藏测试 {t['hidden_pass']}/{t['hidden_total']}"
              + "（" + "，".join(f"{g} {p}/{n}" for g, (p, n) in t["hidden_groups"].items()) + "）")
        if t["public_failed"]:
            print("  未通过的公开测试：" + "、".join(t["public_failed"]))


def local_check():
    """Build and run the public consistency check here; returns False if it cannot build."""
    print("本机检查：编译并运行公开种子的一致性检查（和 make check 相同）……")
    try:
        # -B: always rebuild; make 3.81 (macOS) keeps a build made in the same second as an edit.
        p = subprocess.run(["make", "-s", "-B", "build/consistency"], capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as e:
        print(f"无法运行 make（{type(e).__name__}）。可以加 --skip-local-check 跳过本机检查。")
        return False
    if p.returncode != 0:
        print("本机编译失败：\n    " + (p.stderr or p.stdout).strip()[-1500:].replace("\n", "\n    "))
        return False
    p = subprocess.run([sys.executable, "-I", "tools/check_consistency.py", "build/consistency",
                        "tests/consistency_expected.json"], capture_output=True, text=True)
    lines = p.stdout.strip().splitlines()
    print("    " + "\n    ".join(lines[-4:]))
    if p.returncode != 0:
        print("注意：本机公开种子上有模型的结果为“有错误”或“未完成”（见上方各模型结果）。"
              "三个模型分别计 credit，提交将使用一次机会，建议修改后再提交。")
    return True


def confirm(prompt, expected):
    try:
        return input(prompt).strip() == expected
    except EOFError:
        return False


def main():
    ap = argparse.ArgumentParser(description="作业二提交与查询")
    ap.add_argument("command", choices=["submit", "status"])
    ap.add_argument("--id", required=True, help="学号")
    ap.add_argument("--src", default="src", help="源码目录（默认 src）")
    ap.add_argument("--nickname", help="排行榜上显示的昵称（最多 20 字，可随提交修改；默认“匿名同学”）")
    ap.add_argument("-n", type=int, default=3, help="status 显示最近几次提交（默认 3）")
    ap.add_argument("--yes", action="store_true", help="不再询问确认（学号和次数仍会显示）")
    ap.add_argument("--skip-local-check", action="store_true", help="上传前不在本机编译和运行一致性检查")
    ap.add_argument("--url", default=URL, help=argparse.SUPPRESS)
    args = ap.parse_args()
    base = args.url if args.url.endswith("/") else args.url + "/"
    args.id = args.id.strip()

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
        if not args.skip_local_check and not local_check():
            sys.exit("未上传，没有用掉提交次数。")

        try:
            st = call(base, "api/status?" + urllib.parse.urlencode({"student_id": args.id}), not_found_ok=True)
        except NotFound as e:
            if "还没有提交过" not in str(e):
                sys.exit(f"失败：{e}")
            st = None
        if st is None:
            print(f"\n学号 {args.id} 还没有登记，这次提交会用它登记，成绩记在这个学号下，之后不能自己修改。")
            if not args.yes and not confirm("请再输入一遍学号确认：", args.id):
                sys.exit("学号不一致，未上传。")
            used, limit = 0, None
        else:
            used, limit = st["used"], st["limit"]
            print(f"\n学号 {args.id}（榜单显示为 {st['public_id']} {st['nickname']}）")
            best = next((i for i in st["items"] if i.get("best")), None)
            if best:
                print(f"目前榜单采用 #{best['id']}：credit {'—' if best.get('credit') is None else format(best['credit'], 'g')}，最大误差 {err(best['max_err'])}，"
                      f"误差中位数 {err(best['median_err'])}（榜单取排名最好的一次，再交不会变差）")
            if used >= limit:
                sys.exit(f"{limit} 次提交机会已经用完。")
        which = f"第 {used + 1} 次" + (f"（共 {limit} 次）" if limit else "（共 3 次）")
        print(f"这次提交将用掉{which}提交机会。代码检查或编译失败、与之前完全相同的代码不计次数。")
        if not args.yes and not confirm("确认提交请输入 yes：", "yes"):
            sys.exit("已取消，没有上传。")

        data = {"student_id": args.id, "files": files}
        if args.nickname is not None:
            data["nickname"] = args.nickname
        r = call(base, "api/submit", data)
        if r.get("duplicate_of"):
            print(r["message"])
            print(f"查询结果：python3 tools/submit.py status --id {args.id}")
            return
        print(f"已受理：提交编号 #{r['submission_id']}，时间 {r['received_at']}")
        hint = "；改昵称：提交时加 --nickname 昵称" if r.get("nickname_editable") else ""
        print(f"排行榜上显示为 {r['public_id']} {r['nickname']}（不显示学号{hint}）")
        if r["replaced"]:
            print("已替换排队中的上一份提交 " + "、".join(f"#{i}" for i in r["replaced"]) + "（被替换的不计次数）")
        print(f"已用 {r['used']}/{r['limit']} 次（如果服务器上代码检查或编译失败，这次会退回）；"
              f"排队第 {r['queue_position']} 位，{eta_text(r['eta_minutes'])}，评测通常不到一分钟。")
        print(f"查询结果：python3 tools/submit.py status --id {args.id}")
        return

    r = call(base, "api/status?" + urllib.parse.urlencode({"student_id": args.id}))
    print(f"{r['public_id']} {r['nickname']}  已用 {r['used']}/{r['limit']} 次提交机会；"
          f"队列：评测中 {r['queue']['running']}，排队 {r['queue']['queued']}")
    if not r["items"]:
        print("还没有提交记录。")
    for item in r["items"][:max(1, args.n)]:
        show(item)


if __name__ == "__main__":
    main()
