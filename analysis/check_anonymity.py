#!/usr/bin/env python3
"""Check the distributable package for credential and local identity markers."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKERS = {
    'local_machine_path': re.compile(r'/(?:home|Users|mnt|tmp)/[^\s"\']+'),
    'email_address': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'),
    'api_secret': re.compile(r'\b(?:sk-[A-Za-z0-9_-]{15,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16})\b'),
    'private_key': re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
}


def main() -> None:
    problems = []
    scanned = 0
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.resolve() == Path(__file__).resolve():
            continue
        if path.name == '.env' or path.suffix in {'.pem', '.key'}:
            problems.append((path.relative_to(ROOT), 'credential_file'))
        text = path.read_text(encoding='utf-8', errors='replace')
        scanned += 1
        for name, pattern in MARKERS.items():
            if pattern.search(text):
                problems.append((path.relative_to(ROOT), name))
    if problems:
        for path, reason in problems:
            print(f'FAIL {path}: {reason}')
        raise SystemExit(1)
    print(f'PASS: scanned {scanned} files; no local path, email, or credential markers')


if __name__ == '__main__':
    main()
