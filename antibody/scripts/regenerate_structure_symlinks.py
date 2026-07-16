#!/usr/bin/env python3
"""Retarget Absolut structure symlinks to a new structures directory."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


DEFAULT_OLD_PREFIX = "/mnt/data0/shared/AntBO/Absolut/src/structures"
DEFAULT_NEW_PREFIX = "/mnt/data0/ys/Absolut/src/structures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate symlinks in LINK_DIR so links that currently point under "
            "OLD_PREFIX instead point under NEW_PREFIX with the same filename/path suffix. "
            "Runs as a dry run unless --apply is passed."
        )
    )
    parser.add_argument(
        "link_dir",
        type=Path,
        help="Directory containing the symlinks to retarget.",
    )
    parser.add_argument(
        "--old-prefix",
        default=DEFAULT_OLD_PREFIX,
        help=f"Existing symlink target prefix to replace. Default: {DEFAULT_OLD_PREFIX}",
    )
    parser.add_argument(
        "--new-prefix",
        default=DEFAULT_NEW_PREFIX,
        help=f"New symlink target prefix. Default: {DEFAULT_NEW_PREFIX}",
    )
    parser.add_argument(
        "--pattern",
        default="*",
        help="Only consider symlink names matching this glob. Default: *",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan LINK_DIR recursively instead of only its direct children.",
    )
    parser.add_argument(
        "--from-link-name",
        action="store_true",
        help=(
            "For symlinks whose target does not start with OLD_PREFIX, rebuild the "
            "target as NEW_PREFIX/<link filename> instead of skipping them."
        ),
    )
    parser.add_argument(
        "--check-targets",
        action="store_true",
        help="Warn when the new target path does not currently exist.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually replace symlinks. Without this flag, only print planned changes.",
    )
    return parser.parse_args()


def iter_symlinks(root: Path, pattern: str, recursive: bool):
    if recursive:
        for current_root, dirnames, filenames in os.walk(root, followlinks=False):
            names = sorted(dirnames + filenames)
            for name in names:
                path = Path(current_root) / name
                if path.match(pattern) and path.is_symlink():
                    yield path
    else:
        for path in sorted(root.iterdir()):
            if path.match(pattern) and path.is_symlink():
                yield path


def planned_target(
    link: Path,
    current_target: str,
    old_prefix: str,
    new_prefix: str,
    from_link_name: bool,
) -> str | None:
    old_prefix = old_prefix.rstrip("/")
    new_prefix = new_prefix.rstrip("/")

    if current_target == old_prefix:
        return new_prefix

    old_prefix_with_sep = old_prefix + "/"
    if current_target.startswith(old_prefix_with_sep):
        suffix = current_target[len(old_prefix_with_sep) :]
        return f"{new_prefix}/{suffix}"

    if from_link_name:
        return f"{new_prefix}/{link.name}"

    return None


def replace_symlink(link: Path, target: str) -> None:
    tmp_link = link.with_name(f".{link.name}.relink-{os.getpid()}")
    try:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()
        os.symlink(target, tmp_link)
        os.replace(tmp_link, link)
    finally:
        if tmp_link.exists() or tmp_link.is_symlink():
            tmp_link.unlink()


def main() -> int:
    args = parse_args()
    link_dir = args.link_dir.expanduser()

    if not link_dir.is_dir():
        print(f"error: LINK_DIR is not a directory: {link_dir}", file=os.sys.stderr)
        return 2

    changed = 0
    skipped = 0
    missing = 0

    action = "RELINK" if args.apply else "DRY-RUN"
    for link in iter_symlinks(link_dir, args.pattern, args.recursive):
        current_target = os.readlink(link)
        new_target = planned_target(
            link=link,
            current_target=current_target,
            old_prefix=args.old_prefix,
            new_prefix=args.new_prefix,
            from_link_name=args.from_link_name,
        )

        if new_target is None:
            skipped += 1
            print(f"SKIP    {link} -> {current_target}")
            continue

        if args.check_targets and not Path(new_target).exists():
            missing += 1
            print(f"MISSING target does not exist: {new_target}")

        changed += 1
        print(f"{action:<7} {link} -> {new_target}")

        if args.apply:
            replace_symlink(link, new_target)

    print()
    print(f"summary: changed={changed} skipped={skipped} missing_targets={missing}")
    if not args.apply:
        print("dry run only; rerun with --apply to rewrite the symlinks")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
