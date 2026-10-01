#!/usr/bin/env python3
"""External RSS watchdog for NanoJev model services.

A service over its RSS ceiling is restarted via `launchctl kickstart -k`;
KeepAlive relaunches it with a clean heap/compressor footprint. Down ports
are ignored — KeepAlive owns those restarts.

Runs under launchd (ai.nanojev.memwatch) every StartInterval seconds.
Loopback inspection only; no network calls.
"""
import os
import subprocess
import sys
import time

UID = os.getuid()

# (launchd label, port, rss ceiling in GB)
TARGETS = [
    ("ai.nanojev.kev", 8092, 40),
    ("ai.nanojev.winnow", 8091, 24),
    ("ai.nanojev.valen-lora", 8094, 32),  # net on top of in-proc watchdog
    ("ai.nanojev.service", 8876, 8),
]


def pid_on_port(port):
    try:
        out = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
            capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def rss_gb(pid):
    try:
        out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)],
                             capture_output=True, text=True,
                             timeout=10).stdout.strip()
        return int(out) / 1048576 if out else None
    except Exception:
        return None


def main():
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    for label, port, limit in TARGETS:
        pid = pid_on_port(port)
        if pid is None:
            continue
        rss = rss_gb(pid)
        if rss is None or rss <= limit:
            continue
        print(f"{ts} {label} pid={pid} rss={rss:.1f}GB > {limit}GB "
              f"— kickstart", flush=True)
        subprocess.run(["launchctl", "kickstart", "-k",
                        f"gui/{UID}/{label}"],
                       capture_output=True, timeout=15)


if __name__ == "__main__":
    sys.exit(main())
