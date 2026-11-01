r"""Puts the demo on a public link: the app with the demo racks, the demo feed and a Cloudflare quick tunnel.

Standard library only: python tools/deploy.py   (Windows PowerShell: py tools\deploy.py)
Asks once before installing what is missing of uv, Python 3.12, the app packages and cloudflared (into
code/app/.deploy and code/app/.venv; deleting both folders removes everything). Seeds a week of history if the
history file does not exist yet, starts the app on 127.0.0.1 and the demo feed, opens a quick tunnel, opens its
link in a new browser window and prints it as a QR code; the link changes at every start. Ctrl+C stops
everything; a git pull while it runs reloads the app and keeps the link. Logs go to code/app/.deploy.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import re
import signal
import shutil
import socket
import subprocess
import sys
import tarfile
import time
import urllib.request
import webbrowser
import zipfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
DEPLOY = APP / ".deploy"
BIN = DEPLOY / "bin"
EXE = ".exe" if os.name == "nt" else ""
ARCH = {"amd64": "x86_64", "x64": "x86_64", "arm64": "aarch64"}
UV_URL = "https://github.com/astral-sh/uv/releases/download/0.13.6/"
CF_URL = "https://github.com/cloudflare/cloudflared/releases/download/2026.10.7/"
# (system, machine): (release file, its SHA256, download size in MB)
UV_FILES = {
    ("Linux", "x86_64"): ("uv-x86_64-unknown-linux-gnu.tar.gz", "f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645", 17),
    ("Linux", "aarch64"): ("uv-aarch64-unknown-linux-gnu.tar.gz", "229e194255c7958d86d56e8790d969a6f74736592133f0dfd0076f754c642c1d", 16),
    ("Windows", "x86_64"): ("uv-x86_64-pc-windows-msvc.zip", "a2a8820d5eea32f08ef467ef3b7f66883a9b076d5737a2fc2fbb5f458d1bb50f", 15),
    ("Darwin", "x86_64"): ("uv-x86_64-apple-darwin.tar.gz", "63ba17fdda55ad1a230d4b37fc6017ffa10c23688c83818bd3bfdf914c652a02", 17),
    ("Darwin", "aarch64"): ("uv-aarch64-apple-darwin.tar.gz", "60fbd5c8a1ed681845eea28d9d622fcde03c51e91a43465b64a1e17db1ed3472", 14),
}
# The macOS digests cover the .tgz; the release notes list the checksum of the program inside it instead.
CF_FILES = {
    ("Linux", "x86_64"): ("cloudflared-linux-amd64", "1f99fb518b83dec2ba8d870a7cc991d82a77e6196295a3d2754ba17ab9f91872", 40),
    ("Linux", "aarch64"): ("cloudflared-linux-arm64", "3df43aedc9ff291d1735b5cb576cb97e0b157488796db2e32b28aecabb445775", 38),
    ("Windows", "x86_64"): ("cloudflared-windows-amd64.exe", "2493dbc168923bb3e59e5e7716f71deced92992eea9acd6e9d74f84ac7751343", 55),
    ("Darwin", "x86_64"): ("cloudflared-darwin-amd64.tgz", "02f3edfd87684dd5b705607688e4eb4b323f6e777b81e7b3a85af556aa356848", 22),
    ("Darwin", "aarch64"): ("cloudflared-darwin-arm64.tgz", "3586ccf397e98e5f437e60e7c307ca6ac463e110580503af8bb8e6f9ebf831bd", 20),
}
HOW_TO = ("To set it up by hand: install uv (https://docs.astral.sh/uv/) and cloudflared "
          "(https://github.com/cloudflare/cloudflared/releases), run `uv sync` in code/app, then run this again.")


def say(text: str) -> None:
    print(text, flush=True)


def is_yes(answer: str) -> bool:
    return answer.strip().lower() in ("", "y", "yes")  # Enter alone means yes


def pick(files: dict, tool: str, system: str | None = None, machine: str | None = None) -> tuple:
    machine = (machine or platform.machine()).lower()
    key = (system or platform.system(), ARCH.get(machine, machine))
    if key not in files:
        sys.exit(f"No {tool} download for {key[0]} {key[1]}: install {tool} yourself and run this again.")
    return files[key]


def checksum_ok(data: bytes, digest: str) -> bool:
    return hashlib.sha256(data).hexdigest() == digest


def unpack(name: str, data: bytes, program: str) -> bytes:
    """The program from a release file: the file itself, or the member with that name in a .zip or .tar.gz."""
    if name.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            for member in archive.namelist():
                if member.rsplit("/", 1)[-1] == program:
                    return archive.read(member)
    elif name.endswith((".tar.gz", ".tgz")):
        with tarfile.open(fileobj=io.BytesIO(data)) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.rsplit("/", 1)[-1] == program:
                    return archive.extractfile(member).read()
    else:
        return data
    sys.exit(f"{name} does not contain {program}")


def fetch(base: str, file: tuple, program: str) -> None:
    name, digest, size = file
    say(f"Downloading {name} ({size} MB) ...")
    try:
        with urllib.request.urlopen(base + name, timeout=60) as response:
            data = response.read()
    except OSError as exc:
        sys.exit(f"Download of {name} failed: {exc}")
    if not checksum_ok(data, digest):
        sys.exit(f"{name} does not match its published checksum, so it was not installed.")
    BIN.mkdir(parents=True, exist_ok=True)
    part = BIN / (program + ".part")
    part.write_bytes(unpack(name, data, program))
    part.chmod(0o755)
    os.replace(part, BIN / program)  # an interrupted download never leaves half a program behind


def tool(name: str, base: str, files: dict, what: str, missing: list) -> str:
    """The program on PATH or in .deploy/bin; if it is in neither, its download joins `missing`."""
    found, local = shutil.which(name), BIN / (name + EXE)
    if not found and not local.exists():
        file = pick(files, name)
        missing.append((f"{what} ({file[2]} MB download)", BIN, lambda: fetch(base, file, local.name)))
    return found or str(local)


def ok(command: list) -> bool:
    return subprocess.run(command, cwd=APP, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def run(command: list) -> None:
    if subprocess.run(command, cwd=APP).returncode != 0:
        sys.exit(f"Failed: {' '.join(command)}")


def prepare() -> tuple:
    """uv and cloudflared, after installing whatever is missing; asks one question for all of it."""
    missing = []  # (what, where, how)
    uv = tool("uv", UV_URL, UV_FILES, "uv 0.13.6, the Python project tool", missing)
    own_uv = Path(uv).parent == BIN
    if own_uv:  # its cache and Pythons stay in .deploy too, so one folder removes everything
        os.environ.update(UV_CACHE_DIR=str(DEPLOY / "uv-cache"), UV_PYTHON_INSTALL_DIR=str(DEPLOY / "python"))
    python_ok = Path(uv).exists() and ok([uv, "python", "find"])  # the project's requires-python
    if not python_ok:
        missing.append(("Python 3.12 via uv, unless one is found (about 30 MB download)", DEPLOY / "python" if own_uv else "uv's Python folder",
                        lambda: ok([uv, "python", "find"]) or run([uv, "python", "install", "3.12", "--no-bin", "--no-registry"])))
    if not (python_ok and (APP / ".venv").exists() and ok([uv, "sync", "--check", "--frozen"])):
        missing.append(("app packages (about 200 MB on disk)", APP / ".venv", lambda: run([uv, "sync", "--frozen"])))
    cloudflared = tool("cloudflared", CF_URL, CF_FILES, "cloudflared 2026.10.7, makes the public link", missing)
    if missing:
        say("Missing for the public demo:")
        for what, where, _ in missing:
            say(f"  - {what} into {where}")
        if not sys.stdin.isatty():
            sys.exit("No terminal to answer in, so nothing was installed. " + HOW_TO)
        try:
            answer = input("Install now? [Y/n] ")
        except EOFError:
            answer = "n"
        if not is_yes(answer):
            sys.exit("Nothing was installed. " + HOW_TO)
        for _, _, how in missing:
            how()
    return uv, cloudflared


def tunnel_link(log: str) -> str | None:
    """The quick tunnel's address, from the lines after cloudflared's "Your quick Tunnel has been created!"."""
    found = re.search(r"https://[-a-z0-9]+\.trycloudflare\.com", log.partition("Your quick Tunnel has been created")[2])
    return found.group(0) if found else None


def answers(url: str, seconds: float, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and (proc is None or proc.poll() is None):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                if response.status == 200:
                    return True
        except OSError:  # not up yet: refused, no DNS entry yet, or an error page from the tunnel
            pass
        time.sleep(1)
    return False


def start(command: list, log: Path, env: dict | None = None) -> subprocess.Popen:
    """Shares this console, so Ctrl+C reaches it too; its output goes to a log so the QR code stays on screen."""
    with open(log, "a") as out:
        return subprocess.Popen(command, cwd=APP, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)


def fail(message: str, log: Path) -> None:
    say(f"{message}. Last lines of {log}:\n" + "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    raise SystemExit(1)


def open_window(link: str) -> bool:
    """A new browser window, so the demo does not end up between other tabs."""
    try:
        if os.name == "nt" or "microsoft" in platform.release().lower():  # Windows, or WSL using the Windows browsers
            for exe in ("chrome.exe", "msedge.exe"):
                key = rf"\SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{exe}"
                # start shows an error dialog for an unknown program, so ask the registry first
                if any(ok(["reg.exe", "query", root + key]) for root in ("HKCU", "HKLM")):
                    return ok(["cmd.exe", "/c", "start", "", exe, "--new-window", link])
        elif sys.platform == "darwin":
            return ok(["open", "-na", "Google Chrome", "--args", "--new-window", link]) or ok(["open", link])
    except OSError:
        pass
    if os.name == "nt" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):  # never a text browser
        return webbrowser.open_new(link)
    return False


def print_qr(link: str) -> None:
    """Explicit black on white, so phones read it on dark and light terminal themes alike; runs in the app's .venv."""
    import segno

    if os.name == "nt":  # older Windows consoles print escape codes as text until this mode is on
        import ctypes
        kernel32, mode = ctypes.windll.kernel32, ctypes.c_uint32()
        handle = kernel32.GetStdHandle(-11)
        kernel32.GetConsoleMode(handle, ctypes.byref(mode)) and kernel32.SetConsoleMode(handle, mode.value | 4)
    rows = [list(row) for row in segno.make(link).matrix_iter(border=4)]
    rows.append([0] * len(rows[0]))  # pads an odd row count with quiet zone
    colour = {0: "231", 1: "16"}  # white and black from the fixed part of the 256-colour table
    for top, bottom in zip(rows[0::2], rows[1::2]):
        # equal halves as a plain space, so no terminal "adjusts" a glyph drawn in its own background colour
        cells = (f"\033[48;5;{colour[b]}m " if a == b else f"\033[38;5;{colour[a]};48;5;{colour[b]}m\u2580" for a, b in zip(top, bottom))
        print("".join(cells) + "\033[0m")


def watch(procs: dict, feed: list, logs: dict) -> None:
    told = set()
    while True:
        time.sleep(5)  # also the pause before the feed restarts (it can die while the app reloads)
        if procs["feed"].poll() is not None and procs["app"].poll() is None:
            if "feed" not in told:
                say(f"The demo feed stopped; it restarts by itself whenever that happens (see {logs['feed']}).")
            told.add("feed")
            procs["feed"] = start(feed, logs["feed"])
        for name, what in (("app", "The app stopped"), ("tunnel", "The tunnel stopped, so the link no longer works")):
            if procs[name].poll() is not None and name not in told:
                told.add(name)  # never a new tunnel in a loop: Cloudflare limits how many quick tunnels one can open
                say(f"{what} (see {logs[name]}). Press Ctrl+C and run this again.")


def stop(procs: dict, interrupted: bool) -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the clean-up short
    say("Stopping ...")
    deadline = time.monotonic() + (10 if interrupted else 0)  # Ctrl+C has reached them already, give them time
    for proc in procs.values():
        try:
            proc.wait(max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.terminate()  # uv passes this on to the program it runs; a kill would leave that running
            try:
                proc.wait(5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    say("Everything stopped (app, demo feed, tunnel).")


def in_use(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) == 0


def free_port(first: int) -> int:
    """`first`, or the next port above it that nothing listens on (the vision service may have taken 8002)."""
    port = next(p for p in range(first, first + 100) if not in_use(p))
    if port != first:
        say(f"Port {first} is in use, so the app takes {port}.")
    return port


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, help="local port of the app (default: 8002, or the next free one above it)")
    parser.add_argument("--racks", default="racks.demo.yaml", help="racks file, relative to code/app")
    parser.add_argument("--live", action="store_true", help="the real camera feeds its rack (CAMERA_PULL=1); the feed only feeds the demo racks")
    parser.add_argument("--qr", metavar="LINK", help=argparse.SUPPRESS)  # used by this script inside the app's .venv
    args = parser.parse_args()
    if args.qr:
        print_qr(args.qr)
        return
    racks = APP / args.racks
    if not racks.exists():
        sys.exit(f"{racks} not found")
    if args.port is None:
        args.port = free_port(8002)
    elif in_use(args.port):
        sys.exit(f"Port {args.port} is already in use: stop that program, or pick another port with --port.")
    local = f"http://127.0.0.1:{args.port}"
    os.environ.pop("VIRTUAL_ENV", None)  # another activated venv only makes uv warn; it uses code/app/.venv anyway
    uv, cloudflared = prepare()
    uv_run = [uv, "run", "--no-sync"]  # the checked environment as it is: no installs behind the question's back
    DEPLOY.mkdir(exist_ok=True)
    logs = {name: DEPLOY / f"{name}.log" for name in ("app", "feed", "tunnel")}
    for log in logs.values():
        log.write_text("")
    history = os.environ.get("HISTORY_FILE", "data/history.json")  # empty: the app keeps it in memory only
    if history and not (APP / history).exists():
        say("No history yet: writing seven days of demo history ...")
        run(uv_run + ["python", "tools/demo_data.py", "--seed", "--racks", str(racks), "--history", history, "--app", local])
    env = dict(os.environ, BIKESLOT_RACKS=str(racks), CAMERA_PULL="1" if args.live else "0")
    env.setdefault("STALE_AFTER_S", "30")
    # one argument with "=": on Windows a lone *.yaml is expanded into the file names in code/app
    app = uv_run + ["uvicorn", "bikeslot.main:app", "--host", "127.0.0.1", "--port", str(args.port), "--reload", "--reload-include=*.yaml"]
    feed = uv_run + ["python", "tools/demo_data.py", "--app", local] + ([] if args.live else ["--feed-real"])
    procs, interrupted = {}, False
    try:
        say(f"Starting the app on {local} ...")
        procs["app"] = start(app, logs["app"], env)
        if not answers(f"{local}/api/health", 60, procs["app"]):
            fail("The app did not start", logs["app"])
        procs["feed"] = start(feed, logs["feed"])
        say("Opening the public link (Cloudflare quick tunnel) ...")
        procs["tunnel"] = start([cloudflared, "tunnel", "--no-autoupdate", "--url", local], logs["tunnel"])
        deadline, link = time.monotonic() + 60, None
        while not link and time.monotonic() < deadline and procs["tunnel"].poll() is None:
            time.sleep(0.5)
            text = logs["tunnel"].read_text(encoding="utf-8", errors="replace")
            link = tunnel_link(text) if "Registered tunnel connection" in text else None
        if not link:
            fail("The tunnel did not start", logs["tunnel"])
        if not answers(f"{link}/api/health", 90):
            say("The link does not answer from this computer yet; it may need another minute.")
        say(f"\nPublic link (a new one at every start):\n\n    {link}\n")
        if not open_window(link):
            say("Open the link above in a browser.")
        subprocess.run(uv_run + ["python", str(Path(__file__).resolve()), "--qr", link], cwd=APP)
        say(f"\nPress Ctrl+C to stop. Logs: {DEPLOY}")
        watch(procs, feed, logs)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop(procs, interrupted)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("Stopped.")
