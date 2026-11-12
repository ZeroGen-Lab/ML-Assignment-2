#!/usr/bin/env python3
"""Build and self-check assignment 2 without make (Windows, macOS and Linux).

    python tools/build.py              # build everything into build/      (= make)
    python tools/build.py test         # gradient checks and unit tests    (= make test)
    python tools/build.py test ops     # one group or one test             (= ./build/gradcheck ops)
    python tools/build.py check        # fixed-seed consistency check      (= make check)
    python tools/build.py consistency  # build one program only
    python tools/build.py clean        # delete build/

The compiler is $CXX if set, else the first of g++, clang++, c++ found on PATH;
$CXXFLAGS replaces the default flags. Only sources that changed (or whose headers
changed) are recompiled; --force rebuilds everything. On Windows -march=native is
left out (MinGW GCC can misalign AVX stack spills and crash) and programs are
linked with -static, so they do not depend on whichever MinGW DLLs are on PATH.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

WINDOWS = os.name == "nt"
EXE = ".exe" if WINDOWS else ""
ROOT = Path(__file__).resolve().parent.parent
BUILD = Path("build")
INC = ["-Iinclude", "-Imodels"]
LIB = ["src/tensor.cpp", "src/autograd.cpp", "src/ops.cpp", "src/nn.cpp", "src/optim.cpp"]
PROGRAMS = {"gradcheck": "tests/gradcheck.cpp", "consistency": "tests/consistency.cpp"}
QUIET = False
FORCE = False


def say(text):
    if not QUIET:
        print(text, flush=True)


def find_compiler():
    if os.environ.get("CXX"):
        return os.environ["CXX"]
    for name in ("g++", "clang++", "c++"):
        if shutil.which(name):
            return name
    hint = "Windows 请按 README“使用 Windows”安装 g++，并重新打开终端。" if WINDOWS else "请安装 g++ 或 clang++。"
    sys.exit("没有找到 C++ 编译器（g++ / clang++）。" + hint)


def accepts(cxx, flag):
    """True if the compiler builds an empty program with this flag."""
    try:
        p = subprocess.run([cxx, flag, "-x", "c++", "-c", "-", "-o", os.devnull], input="int main() {}\n",
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return False
    return p.returncode == 0


def default_flags(cxx):
    if os.environ.get("CXXFLAGS"):
        return os.environ["CXXFLAGS"].split()
    flags = ["-O3", "-std=c++17", "-Wall"]
    if not WINDOWS and accepts(cxx, "-march=native"):
        flags.insert(1, "-march=native")
    return flags


def headers():
    return sorted(str(p) for p in Path("include/tinydl").glob("*.hpp")) + ["models/nets.hpp"]


def stale(target, deps):
    if FORCE or not target.exists():
        return True
    built = target.stat().st_mtime
    return any(Path(d).stat().st_mtime >= built for d in deps)


def run_compiler(cmd, what):
    say(what)
    try:
        p = subprocess.run(cmd)
    except OSError as e:
        sys.exit(f"无法运行编译器 {cmd[0]}：{e}")
    if p.returncode != 0:
        sys.exit(f"编译失败：{what}")


def obj_for(src):
    return BUILD / "obj" / (Path(src).stem + ".o")


def build(names, cxx, flags):
    (BUILD / "obj").mkdir(parents=True, exist_ok=True)
    hdr = headers()
    for src in LIB:
        obj = obj_for(src)
        if stale(obj, [src] + hdr):
            run_compiler([cxx, *flags, *INC, "-c", src, "-o", str(obj)], f"编译 {src}")
    link = ["-static"] if WINDOWS else []
    for name in names:
        main = PROGRAMS[name]
        objs = [str(obj_for(s)) for s in LIB]
        exe = BUILD / (name + EXE)
        if stale(exe, [main] + objs + hdr):
            run_compiler([cxx, *flags, *INC, main, *objs, *link, "-o", str(exe)], f"链接 {exe}")


def run(cmd):
    try:
        return subprocess.run(cmd).returncode
    except OSError as e:
        print(f"无法运行 {cmd[0]}：{e}")
        return 1


def main():
    global QUIET, FORCE
    args = sys.argv[1:]
    QUIET, FORCE = "--quiet" in args, "--force" in args
    args = [a for a in args if a not in ("--quiet", "--force")]
    if args and args[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    os.chdir(ROOT)
    command, rest = (args[0], args[1:]) if args else ("all", [])

    if command == "clean":
        shutil.rmtree(BUILD, ignore_errors=True)
        return 0
    if command not in ("all", "test", "check") and command not in PROGRAMS:
        sys.exit(f"未知命令 {command}；可用：all、test、check、clean，或程序名 {'、'.join(PROGRAMS)}")
    cxx = find_compiler()
    flags = default_flags(cxx)
    if command == "all":
        build(list(PROGRAMS), cxx, flags)
    elif command == "test":
        build(["gradcheck"], cxx, flags)
        return run([str(BUILD / ("gradcheck" + EXE)), *rest])
    elif command == "check":
        build(["consistency"], cxx, flags)
        return run([sys.executable, "-I", "tools/check_consistency.py", str(BUILD / ("consistency" + EXE)),
                    "tests/consistency_expected.json"])
    else:
        build([command], cxx, flags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
