#!/usr/bin/env python3
"""Generate API keys for taco-backend.

Usage:
    python generate_keys.py           # Generate 5 keys, write to .api_keys
    python generate_keys.py 3         # Generate 3 keys
    python generate_keys.py --append  # Append to existing file
"""
import secrets
import sys
from pathlib import Path

KEYS_FILE = Path(__file__).parent / ".api_keys"


def main():
    count = 5
    append = False
    for arg in sys.argv[1:]:
        if arg == "--append":
            append = True
        elif arg.isdigit():
            count = int(arg)

    keys = [secrets.token_urlsafe(32) for _ in range(count)]

    mode = "a" if append else "w"
    with open(KEYS_FILE, mode) as f:
        if not append:
            f.write("# taco-backend API keys (one per line)\n")
            f.write("# Regenerate with: python generate_keys.py\n")
        for key in keys:
            f.write(f"{key}\n")

    print(f"{'Appended' if append else 'Wrote'} {count} keys to {KEYS_FILE}")
    print()
    for i, key in enumerate(keys, 1):
        print(f"  Key {i}: {key}")
    print()
    print("Frontend: set Authorization header to 'Bearer <key>'")


if __name__ == "__main__":
    main()
