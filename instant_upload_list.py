# -*- coding: utf-8 -*-
"""
Batch import PikPak:// links using PikPakApi.instant_upload.
"""
import argparse
import asyncio
import os
from typing import List

from pikpakapi import PikPakApi


def load_links(path: str) -> List[str]:
    links: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("PikPak://"):
                links.append(line)
    return links


async def main():
    parser = argparse.ArgumentParser(description="Batch import PikPak:// links")
    parser.add_argument("--input", default="output/FC2_links.txt", help="Links file path")
    parser.add_argument("--parent-id", default=None, help="Target folder id (optional)")
    parser.add_argument("--username", default=None, help="PikPak username (or env PIKPAK_USERNAME)")
    parser.add_argument("--password", default=None, help="PikPak password (or env PIKPAK_PASSWORD)")
    args = parser.parse_args()

    username = args.username or os.getenv("PIKPAK_USERNAME")
    password = args.password or os.getenv("PIKPAK_PASSWORD")
    if not username or not password:
        raise SystemExit("Missing credentials: use --username/--password or set env PIKPAK_USERNAME/PIKPAK_PASSWORD")

    links = load_links(args.input)
    if not links:
        raise SystemExit("No PikPak:// links found in input file")

    api = PikPakApi(username=username, password=password)
    await api.login()

    success = 0
    failed = 0
    for link in links:
        try:
            await api.instant_upload(link, parent_id=args.parent_id)
            success += 1
        except Exception as e:
            failed += 1
            print(f"Failed: {link} -> {e}")

    await api.httpx_client.aclose()
    print(f"Done. success={success} failed={failed} total={len(links)}")


if __name__ == "__main__":
    asyncio.run(main())
