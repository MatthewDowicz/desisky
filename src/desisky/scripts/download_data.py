# SPDX-FileCopyrightText: 2025-present MatthewDowicz <mjdowicz@gmail.com>
#
# SPDX-License-Identifier: MIT

from __future__ import annotations
import argparse
from pathlib import Path

from desisky.data.skyspec import SkySpecVAC
from desisky.data._core import default_root


def main() -> None:
    """
    CLI tool for managing DESI Sky data.

    Provides commands to show the default data directory and download datasets.
    """
    p = argparse.ArgumentParser(
        prog="desisky-data", description="DESI Sky data utilities"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # Show default data dir
    p_dir = sub.add_parser("dir", help="Print the default data directory")
    p_dir.add_argument(
        "--root", type=Path, default=None, help="Override root (optional)"
    )

    # Fetch VAC file
    p_fetch = sub.add_parser("fetch", help="Download the SkySpec VAC FITS file")
    p_fetch.add_argument(
        "--version", default="v1.0", help="Dataset version (default: v1.0)"
    )
    p_fetch.add_argument(
        "--root", type=Path, default=None, help="Destination root directory"
    )
    p_fetch.add_argument(
        "--no-verify", action="store_true", help="Skip checksum verification"
    )

    args = p.parse_args()

    if args.cmd == "dir":
        print(args.root if args.root is not None else default_root())
        return

    if args.cmd == "fetch":
        ds = SkySpecVAC(
            root=args.root,
            version=args.version,
            download=True,
            verify=not args.no_verify,
        )
        print(ds.filepath())


if __name__ == "__main__":
    main()
