#!/usr/bin/env python3
"""One command for a Blackfly + tactile capture with the D455 recording alongside.

    python capture_with_d455.py --output C:\\tactile_data\\run01 --near -- --port COM3 --cameras 4 --no-open-cell-interpolation

Everything after `--` goes to capture_3blackfly_sensor_force.py unchanged
(except --output, which is shared). The D455 needs nothing but its USB cable:
it free-runs at 30 fps and is matched to the Blackfly frames by timestamp.

Order, which is what makes the matching complete:
  1. record_d455.py starts and is given a few seconds to stream;
  2. the capture runs as usual -- press Ctrl+C ONCE to stop it, as always;
  3. only then is the D455 stopped (through a stop file, so the Ctrl+C meant
     for the capture never reaches it early);
  4. align_d455.py attaches the D455 frames to the new session
     (<session>/d455/), unless --no-align.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def pump(stream, prefix):
    for line in iter(stream.readline, ""):
        print(f"{prefix}{line}", end="", flush=True)


def main() -> int:
    if "--" in sys.argv:
        split = sys.argv.index("--")
        own, capture_args = sys.argv[1:split], sys.argv[split + 1:]
    else:
        own, capture_args = sys.argv[1:], []
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--near", action="store_true",
                        help="record the D455 in its close-range mode (record_d455.py --near)")
    parser.add_argument("--d455-args", default="",
                        help='more record_d455.py arguments; values starting with "--" '
                             'need the = form: --d455-args="--preset high_accuracy"')
    parser.add_argument("--warmup", type=float, default=3.0,
                        help="seconds the D455 streams before the capture starts")
    parser.add_argument("--no-align", action="store_true")
    args = parser.parse_args(own)
    if any(a == "--output" for a in capture_args):
        parser.error("give --output before `--`; it is shared by both recordings")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stop_file = output / ".d455_stop"
    stop_file.unlink(missing_ok=True)
    before = {p.name for p in output.glob("capture_*")}

    python = sys.executable
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    d455 = subprocess.Popen(
        [python, "-u", str(HERE / "record_d455.py"), "--output", str(output),
         "--stop-file", str(stop_file), *(["--near"] if args.near else []),
         *args.d455_args.split()],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=flags,
        start_new_session=(os.name != "nt"))
    threading.Thread(target=pump, args=(d455.stdout, "[d455] "), daemon=True).start()
    time.sleep(args.warmup)
    if d455.poll() is not None:
        print("D455 recording did not start; see the [d455] lines above.")
        return 1

    # Ctrl+C belongs to the capture; this process just waits for it to finish.
    signal.signal(signal.SIGINT, lambda *_: None)
    capture = subprocess.Popen(
        [python, str(HERE / "capture_3blackfly_sensor_force.py"), "--output", str(output),
         *capture_args])
    code = capture.wait()

    stop_file.touch()
    try:
        d455.wait(timeout=20)
    except subprocess.TimeoutExpired:
        print("D455 recording did not stop in 20 s; terminating it.")
        d455.terminate()
    stop_file.unlink(missing_ok=True)

    new = sorted(p for p in output.glob("capture_*") if p.name not in before and p.is_dir())
    if not new:
        print("no new capture session found; nothing to align.")
        return code or 1
    session = new[-1]
    print(f"\ncapture session: {session}  (capture exit code {code})")
    if args.no_align:
        print(f"next: python align_d455.py \"{session}\"")
        return code
    return subprocess.call([python, str(HERE / "align_d455.py"), str(session)]) or code


if __name__ == "__main__":
    raise SystemExit(main())
