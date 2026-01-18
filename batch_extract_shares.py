# -*- coding: utf-8 -*-
"""
PikPak 分享链接批量提取工具（无需登录）
通过 captcha_token 访问公开分享链接，递归提取文件信息。
"""
import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

import httpx

from pikpakapi.utils import (
    build_custom_user_agent,
    device_id_generator,
    get_timestamp,
)


PIKPAK_API_HOST = "api-drive.mypikpak.com"
PIKPAK_USER_HOST = "user.mypikpak.com"

# Web (drive.mypikpak.com) captcha config extracted from JS bundle.
WEB_DEFAULTS = {
    "client_id": "YUMx5nI8ZU8Ap8pm",
    "client_version": "undefined",
    "package_name": "drive.mypikpak.com",
    "timestamp": "1768554341901",
    "salts": [
        "fyZ4+p77W1U4zcWBUwefAIFhFxvADWtT1wzolCxhg9q7etmGUjXr",
        "uSUX02HYJ1IkyLdhINEFcCf7l2",
        "iWt97bqD/qvjIaPXB2Ja5rsBWtQtBZZmaHH2rMR41",
        "3binT1s/5a1pu3fGsN",
        "8YCCU+AIr7pg+yd7CkQEY16lDMwi8Rh4WNp5",
        "DYS3StqnAEKdGddRP8CJrxUSFh",
        "crquW+4",
        "ryKqvW9B9hly+JAymXCIfag5Z",
        "Hr08T/NDTX1oSJfHk90c",
        "i",
    ],
}


def load_web_captcha_config(js_dir: Path) -> Dict[str, Any]:
    js_file = js_dir / "B-CBPbzY.js"
    if not js_file.exists():
        return WEB_DEFAULTS.copy()

    text = js_file.read_text(encoding="utf-8", errors="ignore")
    match = re.search(
        r'W=\\{clientId:"([^"]+)",clientVersion:"([^"]*)",packageName:"([^"]+)",timestamp:"([^"]+)",algorithms:\\[(.*?)\\]\\}',
        text,
        re.S,
    )
    if not match:
        return WEB_DEFAULTS.copy()

    salts = re.findall(r'salt:"([^"]+)"', match.group(5))
    return {
        "client_id": match.group(1),
        "client_version": match.group(2),
        "package_name": match.group(3),
        "timestamp": match.group(4),
        "salts": salts or WEB_DEFAULTS["salts"],
    }


def captcha_sign_web(
    client_id: str,
    client_version: str,
    package_name: str,
    device_id: str,
    timestamp: str,
    salts: List[str],
) -> str:
    base = f"{client_id}{client_version}{package_name}{device_id}{timestamp}"
    sign = base
    for salt in salts:
        sign = hashlib.md5((sign + salt).encode()).hexdigest()
    return f"1.{sign}"


@dataclass
class ShareItem:
    title: str
    link: str
    password: Optional[str] = None


class AnonymousPikPakClient:
    def __init__(self, device_id: Optional[str] = None, timeout: float = 12.0, js_dir: Path = Path("js")):
        self.device_id = device_id or device_id_generator()
        self.http = httpx.AsyncClient(timeout=timeout)
        self.web_cfg = load_web_captcha_config(js_dir)
        self.rate_limiter: Optional["AsyncRateLimiter"] = None

    async def close(self):
        await self.http.aclose()

    def _headers(self, captcha_token: Optional[str] = None) -> Dict[str, str]:
        headers = {
            "User-Agent": (
                build_custom_user_agent(self.device_id, "")
                if captcha_token
                else "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36"
            ),
            "Content-Type": "application/json; charset=utf-8",
            "X-Device-Id": self.device_id,
            "X-Client-Id": self.web_cfg["client_id"],
        }
        if captcha_token:
            headers["X-Captcha-Token"] = captcha_token
        return headers

    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        data = response.json()
        if not data:
            return {}
        if "error" in data:
            raise RuntimeError(data.get("error_description", "Unknown Error"))
        return data

    async def captcha_init(self, action: str) -> str:
        url = f"https://{PIKPAK_USER_HOST}/v1/shield/captcha/init"
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        ts = f"{get_timestamp()}"
        sign = captcha_sign_web(
            client_id=self.web_cfg["client_id"],
            client_version=self.web_cfg["client_version"],
            package_name=self.web_cfg["package_name"],
            device_id=self.device_id,
            timestamp=ts,
            salts=self.web_cfg["salts"],
        )
        meta = {
            "captcha_sign": sign,
            "client_version": self.web_cfg["client_version"],
            "package_name": self.web_cfg["package_name"],
            "user_id": "",
            "timestamp": ts,
        }
        payload = {
            "client_id": self.web_cfg["client_id"],
            "action": action,
            "device_id": self.device_id,
            "meta": meta,
        }
        resp = await self.http.post(url, json=payload, headers=self._headers())
        data = self._handle_response(resp)
        token = data.get("captcha_token", "")
        if not token:
            raise RuntimeError("captcha_token get failed")
        return token

    async def _get_with_captcha(self, url: str, params: Dict[str, Any]) -> Dict[str, Any]:
        if self.rate_limiter:
            await self.rate_limiter.acquire()
        captcha_token = await self.captcha_init(action=f"GET:{url}")
        resp = await self.http.get(url, params=params, headers=self._headers(captcha_token))
        return self._handle_response(resp)

    async def get_share_info(
        self, share_id: str, parent_id: Optional[str] = None, pass_code: Optional[str] = None
    ) -> Dict[str, Any]:
        url = f"https://{PIKPAK_API_HOST}/drive/v1/share"
        params: Dict[str, Any] = {
            "limit": "100",
            "thumbnail_size": "SIZE_LARGE",
            "order": "3",
            "share_id": share_id,
        }
        if parent_id:
            params["parent_id"] = parent_id
        if pass_code:
            params["pass_code"] = pass_code
        return await self._get_with_captcha(url, params)

    async def list_share_detail(
        self,
        share_id: str,
        pass_code_token: str,
        parent_id: Optional[str] = None,
        limit: int = 100,
        page_token: Optional[str] = None,
    ) -> Dict[str, Any]:
        url = f"https://{PIKPAK_API_HOST}/drive/v1/share/detail"
        params: Dict[str, Any] = {
            "limit": str(limit),
            "thumbnail_size": "SIZE_LARGE",
            "order": "6",
            "folders_first": "true",
            "share_id": share_id,
            "pass_code_token": pass_code_token,
        }
        if parent_id:
            params["parent_id"] = parent_id
        if page_token:
            params["page_token"] = page_token
        return await self._get_with_captcha(url, params)


class ShareLinkExtractor:
    def __init__(self, request_delay: float = 0.6, client: Optional[AnonymousPikPakClient] = None):
        self.client = client or AnonymousPikPakClient()
        self.request_delay = request_delay

    async def close(self):
        await self.client.close()

    @staticmethod
    def parse_share_link(share_link: str) -> Tuple[str, List[str]]:
        match = re.search(r"/s/([^/?#]+)(?:/([^?#]+))?", share_link)
        if not match:
            raise ValueError("无效的分享链接格式")
        share_id = match.group(1)
        path = match.group(2) or ""
        path_ids = [p for p in path.split("/") if p]
        return share_id, path_ids

    async def resolve_parent_id(
        self, share_id: str, pass_code_token: str, path_ids: List[str]
    ) -> Optional[str]:
        parent_id = None
        for target_id in path_ids:
            found = await self._find_child_by_id(share_id, pass_code_token, parent_id, target_id)
            if not found:
                raise RuntimeError(f"路径中的文件夹 ID 未找到: {target_id}")
            parent_id = target_id
        return parent_id

    async def _find_child_by_id(
        self, share_id: str, pass_code_token: str, parent_id: Optional[str], target_id: str
    ) -> bool:
        page_token = None
        while True:
            result = await self.client.list_share_detail(
                share_id, pass_code_token, parent_id=parent_id, page_token=page_token
            )
            for item in result.get("files", []):
                if item.get("id") == target_id:
                    return True
            page_token = result.get("next_page_token", "")
            if not page_token:
                return False
            await asyncio.sleep(self.request_delay)

    async def walk_files(
        self,
        share_id: str,
        pass_code_token: str,
        parent_id: Optional[str],
        current_path: str,
        seen_folders: Set[str],
    ) -> List[Dict[str, Any]]:
        all_files: List[Dict[str, Any]] = []
        page_token = None
        while True:
            result = await self.client.list_share_detail(
                share_id, pass_code_token, parent_id=parent_id, page_token=page_token
            )
            for item in result.get("files", []):
                name = item.get("name", "")
                kind = item.get("kind", "")
                item_path = f"{current_path}{name}".strip("/")
                if "folder" in kind:
                    folder_id = item.get("id", "")
                    if folder_id and folder_id not in seen_folders:
                        seen_folders.add(folder_id)
                        await asyncio.sleep(self.request_delay)
                        all_files.extend(
                            await self.walk_files(
                                share_id,
                                pass_code_token,
                                parent_id=folder_id,
                                current_path=f"{item_path}/",
                                seen_folders=seen_folders,
                            )
                        )
                else:
                    item["_path"] = item_path
                    all_files.append(item)
            page_token = result.get("next_page_token", "")
            if not page_token:
                break
            await asyncio.sleep(self.request_delay)
        return all_files

    async def walk_tree(
        self,
        share_id: str,
        pass_code_token: str,
        parent_id: Optional[str],
        seen_folders: Set[str],
    ) -> Dict[str, Any]:
        tree: Dict[str, Any] = {}
        page_token = None
        while True:
            result = await self.client.list_share_detail(
                share_id, pass_code_token, parent_id=parent_id, page_token=page_token
            )
            for item in result.get("files", []):
                name = item.get("name", "")
                kind = item.get("kind", "")
                if "folder" in kind:
                    folder_id = item.get("id", "")
                    if folder_id and folder_id not in seen_folders:
                        seen_folders.add(folder_id)
                        await asyncio.sleep(self.request_delay)
                        tree[name] = await self.walk_tree(
                            share_id,
                            pass_code_token,
                            parent_id=folder_id,
                            seen_folders=seen_folders,
                        )
                else:
                    tree[name] = {
                        "size": int(item.get("size", 0)),
                        "hash": item.get("hash") or item.get("md5_checksum") or "",
                        "kind": kind,
                    }
            page_token = result.get("next_page_token", "")
            if not page_token:
                break
            await asyncio.sleep(self.request_delay)
        return tree

    async def walk_tree_concurrent(
        self,
        share_id: str,
        pass_code_token: str,
        parent_id: Optional[str],
        max_workers: int = 3,
        progress_path: Optional[Path] = None,
        progress_interval: float = 5.0,
        resume_path: Optional[Path] = None,
        progress_tree_json_path: Optional[Path] = None,
        progress_root_name: str = "ROOT",
        on_dir_complete: Optional[Callable[[str, List[Tuple[str, int, str]]], None]] = None,
        on_dir_error: Optional[Callable[[str, Exception], None]] = None,
        progress_summary_path: Optional[Path] = None,
        print_progress: bool = False,
        max_retries_per_dir: int = 3,
        retry_backoff: float = 2.0,
        progress_extra: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        queue: asyncio.Queue = asyncio.Queue()
        tree: Dict[str, Any] = {}
        seen: Set[str] = set()
        done: Set[str] = set()
        failed: Set[str] = set()
        queued: Set[str] = set()
        retries: Dict[str, int] = {}
        last_progress_write = 0.0
        progress_lock = asyncio.Lock()
        counters = {
            "total_files": 0,
            "total_size": 0,
            "working_dirs": 0,
        }
        start_time = time.monotonic()
        counter_lock = asyncio.Lock()
        root_task_id = parent_id or "__ROOT__"

        # Map folder id to (node, display name)
        node_by_id: Dict[str, Dict[str, Any]] = {}

        if progress_tree_json_path and progress_tree_json_path.exists():
            loaded = load_progress_tree_json(progress_tree_json_path)
            if isinstance(loaded, dict) and loaded:
                tree = loaded
                rebuild_node_index(tree, node_by_id)
                root_task_id = tree.get("__id__", root_task_id)
                tree.setdefault("__id__", root_task_id)
                tree.setdefault("__path__", "")

        if not node_by_id:
            node_by_id[root_task_id] = tree
            if isinstance(tree, dict):
                tree["__id__"] = root_task_id
                tree["__path__"] = ""

        if resume_path and resume_path.exists():
            data = load_progress(resume_path)
            done = set(data.get("done_dirs", []))
            failed = set(data.get("failed_dirs", []))
            queued = set(data.get("queued_dirs", []))
            stats = data.get("stats", {})
            counters["total_files"] = int(stats.get("total_files", 0))
            counters["total_size"] = int(stats.get("total_size", 0))
            missing_ids = [qid for qid in queued if qid not in node_by_id]
            if missing_ids:
                await queue.put(root_task_id)
                queued.add(root_task_id)
            else:
                for qid in list(queued):
                    await queue.put(qid)
            if not queued and root_task_id not in done:
                await queue.put(root_task_id)
                queued.add(root_task_id)
        else:
            await queue.put(root_task_id)
            queued.add(root_task_id)

        if parent_id:
            seen.add(parent_id)

        async def maybe_write_progress(force: bool = False):
            nonlocal last_progress_write
            now = time.monotonic()
            if not force and (now - last_progress_write) < progress_interval:
                return
            async with progress_lock:
                now = time.monotonic()
                if not force and (now - last_progress_write) < progress_interval:
                    return
                completed_dirs = count_completed_dirs(tree)
                scanned_dirs = count_scanned_dirs(tree)
                if progress_path:
                    write_progress_tree(progress_path, progress_root_name, tree)
                if progress_tree_json_path:
                    write_progress_tree_json(progress_tree_json_path, tree)
                if progress_summary_path:
                    write_progress_report(
                        progress_summary_path,
                        {
                            "scanned_dirs": scanned_dirs,
                            "completed_dirs": completed_dirs,
                            "failed_dirs": len(failed),
                            "queued_dirs": len(queued),
                            "working_dirs": counters["working_dirs"],
                            "total_files": counters["total_files"],
                            "total_size": counters["total_size"],
                            "elapsed_seconds": (now - start_time),
                        },
                    )
                if print_progress:
                    size_tb = counters["total_size"] / 1024 / 1024 / 1024 / 1024
                    if size_tb >= 1:
                        size_display = f"{size_tb:.2f}TB"
                    else:
                        size_gb = counters["total_size"] / 1024 / 1024 / 1024
                        size_display = f"{size_gb:.2f}GB"
                    elapsed = now - start_time
                    hours = int(elapsed // 3600)
                    minutes = int((elapsed % 3600) // 60)
                    seconds = int(elapsed % 60)
                    print(
                        f"[progress] scanned={scanned_dirs} completed={completed_dirs} failed={len(failed)} "
                        f"queued={len(queued)} working={counters['working_dirs']} "
                        f"files={counters['total_files']} size={size_display} "
                        f"elapsed={hours:02d}:{minutes:02d}:{seconds:02d}"
                    )
                last_progress_write = now
                if resume_path:
                    stats_payload = {
                        "total_files": counters["total_files"],
                        "total_size": counters["total_size"],
                        "elapsed_seconds": (now - start_time),
                    }
                    extra_payload = progress_extra() if progress_extra else {}
                    save_progress(resume_path, done, queued, failed, stats_payload, extra_payload)

        async def progress_poller():
            # Periodic progress output even if no page completes yet.
            while True:
                await asyncio.sleep(progress_interval)
                await maybe_write_progress()

        async def worker():
            while True:
                folder_id = await queue.get()
                try:
                    if folder_id in queued:
                        queued.discard(folder_id)
                    node = node_by_id.get(folder_id, tree)
                    if isinstance(node, dict):
                        node["__status__"] = "WORKING"
                        node["__page__"] = 0
                    async with counter_lock:
                        counters["working_dirs"] += 1
                    page_token = None
                    page_count = 0
                    dir_files: List[Tuple[str, int, str]] = []
                    dir_total_files = 0
                    dir_total_size = 0
                    try:
                        while True:
                            page_count += 1
                            if isinstance(node, dict):
                                node["__page__"] = page_count
                            api_parent_id = None if folder_id == root_task_id else folder_id
                            result = await self.client.list_share_detail(
                                share_id,
                                pass_code_token,
                                parent_id=api_parent_id,
                                page_token=page_token,
                            )
                            for item in result.get("files", []):
                                name = item.get("name", "")
                                kind = item.get("kind", "")
                                if "folder" in kind:
                                    child_id = item.get("id", "")
                                    if child_id and child_id not in seen:
                                        seen.add(child_id)
                                        child_node: Dict[str, Any] = {}
                                        child_node["__id__"] = child_id
                                        child_node["__status__"] = "PENDING"
                                        parent_path = node.get("__path__", "") if isinstance(node, dict) else ""
                                        child_node["__path__"] = f"{parent_path}{name}/"
                                        node[name] = child_node
                                        node_by_id[child_id] = child_node
                                        await queue.put(child_id)
                                        queued.add(child_id)
                                else:
                                    parent_path = node.get("__path__", "") if isinstance(node, dict) else ""
                                    node[name] = {
                                        "size": int(item.get("size", 0)),
                                        "hash": item.get("hash") or item.get("md5_checksum") or "",
                                        "kind": kind,
                                    }
                                    if on_dir_complete:
                                        file_path = f"{parent_path}{name}"
                                        dir_files.append(
                                            (
                                                file_path,
                                                int(item.get("size", 0)),
                                                item.get("hash") or item.get("md5_checksum") or "",
                                            )
                                        )
                                    dir_total_files += 1
                                    dir_total_size += int(item.get("size", 0))
                            page_token = result.get("next_page_token", "")
                            if not page_token:
                                break
                            await asyncio.sleep(self.request_delay)
                            await maybe_write_progress()
                        if isinstance(node, dict):
                            node["__status__"] = "SCANNED"
                    except Exception as e:
                        if isinstance(node, dict):
                            node["__status__"] = "FAILED"
                        err = e
                        raise
                except Exception as e:
                    count = retries.get(folder_id, 0) + 1
                    retries[folder_id] = count
                    if count <= max_retries_per_dir:
                        if isinstance(node, dict):
                            node["__status__"] = "RETRY"
                        await asyncio.sleep(retry_backoff * (2 ** (count - 1)))
                        await queue.put(folder_id)
                        queued.add(folder_id)
                    else:
                        failed.add(folder_id)
                        if isinstance(node, dict):
                            node["__status__"] = "FAILED"
                        if on_dir_error and folder_id:
                            dir_path = node.get("__path__", "") if isinstance(node, dict) else ""
                            on_dir_error(dir_path, e)
                else:
                    async with counter_lock:
                        counters["total_files"] += dir_total_files
                        counters["total_size"] += dir_total_size
                    if on_dir_complete and folder_id:
                        dir_path = node.get("__path__", "") if isinstance(node, dict) else ""
                        maybe_coro = on_dir_complete(dir_path, dir_files)
                        if asyncio.iscoroutine(maybe_coro):
                            await maybe_coro
                    done.add(folder_id)
                finally:
                    async with counter_lock:
                        counters["working_dirs"] = max(0, counters["working_dirs"] - 1)
                    queue.task_done()
                    await maybe_write_progress()

        await maybe_write_progress(force=True)
        poller = asyncio.create_task(progress_poller())
        workers = [asyncio.create_task(worker()) for _ in range(max_workers)]
        await queue.join()
        for w in workers:
            w.cancel()
        poller.cancel()
        await maybe_write_progress(force=True)
        return tree

    async def extract_share_link(self, share_link: str, pass_code: Optional[str] = None) -> Dict[str, Any]:
        share_id, path_ids = self.parse_share_link(share_link)

        share_info = await self.client.get_share_info(share_id, parent_id=None, pass_code=pass_code)
        pass_code_token = share_info.get("pass_code_token", "")
        if not pass_code_token:
            raise RuntimeError("未能获取 pass_code_token")

        parent_id = await self.resolve_parent_id(share_id, pass_code_token, path_ids)
        files = await self.walk_files(share_id, pass_code_token, parent_id, "", set())

        return {
            "status": "success",
            "share_id": share_id,
            "parent_id": parent_id,
            "files": files,
            "file_count": len(files),
            "timestamp": datetime.now().isoformat(),
        }

    async def extract_share_tree(self, share_link: str, pass_code: Optional[str] = None) -> Dict[str, Any]:
        share_id, path_ids = self.parse_share_link(share_link)

        share_info = await self.client.get_share_info(share_id, parent_id=None, pass_code=pass_code)
        pass_code_token = share_info.get("pass_code_token", "")
        if not pass_code_token:
            raise RuntimeError("未能获取 pass_code_token")

        parent_id = await self.resolve_parent_id(share_id, pass_code_token, path_ids)
        tree = await self.walk_tree(share_id, pass_code_token, parent_id, set())

        return {
            "status": "success",
            "share_id": share_id,
            "parent_id": parent_id,
            "tree": tree,
            "timestamp": datetime.now().isoformat(),
        }

    async def extract_share_tree_concurrent(
        self,
        share_link: str,
        pass_code: Optional[str] = None,
        max_workers: int = 3,
        progress_path: Optional[Path] = None,
        progress_interval: float = 5.0,
        resume_path: Optional[Path] = None,
        progress_tree_json_path: Optional[Path] = None,
        progress_root_name: str = "ROOT",
        on_dir_complete: Optional[Callable[[str, List[Tuple[str, int, str]]], None]] = None,
        progress_summary_path: Optional[Path] = None,
        print_progress: bool = False,
        on_dir_error: Optional[Callable[[str, Exception], None]] = None,
        max_retries_per_dir: int = 3,
        retry_backoff: float = 2.0,
        progress_extra: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        share_id, path_ids = self.parse_share_link(share_link)

        share_info = await self.client.get_share_info(share_id, parent_id=None, pass_code=pass_code)
        pass_code_token = share_info.get("pass_code_token", "")
        if not pass_code_token:
            raise RuntimeError("未能获取 pass_code_token")

        parent_id = await self.resolve_parent_id(share_id, pass_code_token, path_ids)
        tree = await self.walk_tree_concurrent(
            share_id,
            pass_code_token,
            parent_id,
            max_workers=max_workers,
            progress_path=progress_path,
            progress_interval=progress_interval,
            resume_path=resume_path,
            progress_tree_json_path=progress_tree_json_path,
            progress_root_name=progress_root_name,
            on_dir_complete=on_dir_complete,
            progress_summary_path=progress_summary_path,
            print_progress=print_progress,
            on_dir_error=on_dir_error,
            max_retries_per_dir=max_retries_per_dir,
            retry_backoff=retry_backoff,
            progress_extra=progress_extra,
        )

        return {
            "status": "success",
            "share_id": share_id,
            "parent_id": parent_id,
            "tree": tree,
            "timestamp": datetime.now().isoformat(),
        }


class BatchExtractor:
    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.extractor = ShareLinkExtractor()

    async def close(self):
        await self.extractor.close()

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, "_")
        return filename[:200]

    def save_result(self, title: str, share_link: str, result: Dict[str, Any]):
        output_file = self.output_dir / f"{self._sanitize_filename(title)}.txt"
        total_size = sum(int(f.get("size", 0)) for f in result["files"])

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write(f"# 分享链接: {share_link}\n")
            f.write(f"# 文件数量: {result['file_count']}\n")
            f.write(f"# 提取时间: {result['timestamp']}\n\n")

            for file in result["files"]:
                path = file.get("_path", file.get("name", ""))
                size = int(file.get("size", 0))
                file_hash = file.get("hash") or file.get("md5_checksum") or ""
                size_mb = size / 1024 / 1024
                size_gb = size / 1024 / 1024 / 1024

                if size_gb >= 1:
                    f.write(f"# {path} ({size_gb:.2f} GB)\n")
                else:
                    f.write(f"# {path} ({size_mb:.2f} MB)\n")

                f.write(f"PikPak://{path}|{size}|{file_hash}\n\n")

            total_gb = total_size / 1024 / 1024 / 1024
            f.write(f"# 总大小: {total_gb:.2f} GB\n")

    def save_tree_result(self, title: str, share_link: str, result: Dict[str, Any]):
        output_file = self.output_dir / f"{self._sanitize_filename(title)}_tree.txt"

        def fmt_size(size: int) -> str:
            gb = size / 1024 / 1024 / 1024
            if gb >= 1:
                return f"{gb:.2f} GB"
            mb = size / 1024 / 1024
            if mb >= 1:
                return f"{mb:.2f} MB"
            kb = size / 1024
            return f"{kb:.2f} KB"

        def write_tree(node: Dict[str, Any], prefix: str, f):
            # Preserve API order by iterating insertion order (no sorting).
            items = [item for item in node.items() if not item[0].startswith("__")]
            for i, (name, value) in enumerate(items):
                is_last = i == len(items) - 1
                branch = "└── " if is_last else "├── "
                next_prefix = prefix + ("    " if is_last else "│   ")
                if isinstance(value, dict) and "size" not in value:
                    f.write(f"{prefix}{branch}{name}/\n")
                    write_tree(value, next_prefix, f)
                else:
                    size = value.get("size", 0) if isinstance(value, dict) else 0
                    f.write(f"{prefix}{branch}{name} ({fmt_size(int(size))})\n")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write(f"# 分享链接: {share_link}\n")
            f.write(f"# 提取时间: {result['timestamp']}\n\n")
            f.write(f"{title}/\n")
            write_tree(result["tree"], "", f)

    def save_links_result(self, title: str, share_link: str, result: Dict[str, Any]):
        output_file = self.output_dir / f"{self._sanitize_filename(title)}_links.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write(f"# 分享链接: {share_link}\n")
            f.write(f"# 文件数量: {result['file_count']}\n")
            f.write(f"# 提取时间: {result['timestamp']}\n\n")
            for file in result["files"]:
                path = file.get("_path", file.get("name", ""))
                size = int(file.get("size", 0))
                file_hash = file.get("hash") or file.get("md5_checksum") or ""
                f.write(f"PikPak://{path}|{size}|{file_hash}\n")

    def save_stats_result(self, title: str, share_link: str, stats: Dict[str, Any]):
        output_file = self.output_dir / f"{self._sanitize_filename(title)}_stats.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write(f"# 分享链接: {share_link}\n")
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")

    def save_links_from_tree(self, title: str, share_link: str, tree: Dict[str, Any]):
        output_file = self.output_dir / f"{self._sanitize_filename(title)}_links.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(f"# {title}\n")
            f.write(f"# 分享链接: {share_link}\n")
            f.write(f"# 提取时间: {datetime.now().isoformat()}\n\n")
            for path, size, file_hash in iter_links_from_tree(tree):
                f.write(f"PikPak://{path}|{size}|{file_hash}\n")

    async def run(self, items: List[ShareItem]):
        for item in items:
            print(f"\n处理: {item.title}")
            print(f"链接: {item.link}")
            try:
                result = await self.extractor.extract_share_link(item.link, item.password)
                self.save_result(item.title, item.link, result)
                print(f"  ✓ 文件数量: {result['file_count']}")
            except Exception as e:
                print(f"  ❌ 提取失败: {e}")


class AsyncRateLimiter:
    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_time = 0.0

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_time)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_time = time.monotonic()


def iter_links_from_tree(tree: Dict[str, Any], prefix: str = ""):
    for name, value in tree.items():
        if name.startswith("__"):
            continue
        if isinstance(value, dict) and "size" not in value:
            next_prefix = f"{prefix}{name}/"
            yield from iter_links_from_tree(value, next_prefix)
        else:
            size = int(value.get("size", 0)) if isinstance(value, dict) else 0
            file_hash = value.get("hash", "") if isinstance(value, dict) else ""
            yield f"{prefix}{name}", size, file_hash


def compute_stats_from_tree(tree: Dict[str, Any]) -> Dict[str, int]:
    files = 0
    dirs = 0
    total_size = 0
    for name, value in tree.items():
        if name.startswith("__"):
            continue
        if isinstance(value, dict) and "size" not in value:
            dirs += 1
            sub = compute_stats_from_tree(value)
            files += sub["files"]
            dirs += sub["dirs"]
            total_size += sub["total_size"]
        else:
            files += 1
            total_size += int(value.get("size", 0)) if isinstance(value, dict) else 0
    return {"files": files, "dirs": dirs, "total_size": total_size}


def write_progress_report(path: Path, stats: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        for k, v in stats.items():
            f.write(f"{k}: {v}\n")


def all_subdirs_done(node: Dict[str, Any]) -> bool:
    for name, value in node.items():
        if name.startswith("__"):
            continue
        if isinstance(value, dict) and "size" not in value:
            status = value.get("__status__")
            if status == "WORKING":
                return False
            if status not in ("DONE", "SCANNED") or not all_subdirs_done(value):
                return False
    return True


def count_scanned_dirs(node: Dict[str, Any]) -> int:
    count = 0
    for name, value in node.items():
        if name.startswith("__"):
            continue
        if isinstance(value, dict) and "size" not in value:
            status = value.get("__status__")
            if status in ("SCANNED", "DONE"):
                count += 1
            count += count_scanned_dirs(value)
    return count


def count_completed_dirs(node: Dict[str, Any]) -> int:
    count = 0
    for name, value in node.items():
        if name.startswith("__"):
            continue
        if isinstance(value, dict) and "size" not in value:
            status = value.get("__status__")
            if status in ("SCANNED", "DONE") and all_subdirs_done(value):
                count += 1
            count += count_completed_dirs(value)
    return count


def write_progress_tree(path: Path, root_name: str, tree: Dict[str, Any]):
    def fmt_size(size: int) -> str:
        gb = size / 1024 / 1024 / 1024
        if gb >= 1:
            return f"{gb:.2f} GB"
        mb = size / 1024 / 1024
        if mb >= 1:
            return f"{mb:.2f} MB"
        kb = size / 1024
        return f"{kb:.2f} KB"

    def fmt_status(node: Dict[str, Any]) -> str:
        status = node.get("__status__")
        if not status:
            return ""
        if status == "WORKING":
            page = node.get("__page__")
            return f" [{status} p{page}]"
        if status in ("SCANNED", "DONE") and all_subdirs_done(node):
            return " [DONE]"
        if status == "SCANNED":
            return " [SCANNED]"
        return f" [{status}]"

    def write_node(node: Dict[str, Any], prefix: str, f):
        items = [item for item in node.items() if not item[0].startswith("__")]
        for i, (name, value) in enumerate(items):
            is_last = i == len(items) - 1
            branch = "└── " if is_last else "├── "
            next_prefix = prefix + ("    " if is_last else "│   ")
            if isinstance(value, dict) and "size" not in value:
                f.write(f"{prefix}{branch}{name}/{fmt_status(value)}\n")
                write_node(value, next_prefix, f)
            else:
                size = value.get("size", 0) if isinstance(value, dict) else 0
                f.write(f"{prefix}{branch}{name} ({fmt_size(int(size))})\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{root_name}/\n")
        write_node(tree, "", f)


def write_progress_tree_json(path: Path, tree: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree, f, ensure_ascii=False, indent=2)


def load_progress_tree_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return None
        return data
    except Exception:
        return None


def rebuild_node_index(node: Dict[str, Any], node_by_id: Dict[str, Dict[str, Any]]):
    node_id = node.get("__id__")
    if node_id:
        node_by_id[node_id] = node
    for name, value in node.items():
        if name.startswith("__"):
            continue
        if isinstance(value, dict) and "size" not in value:
            rebuild_node_index(value, node_by_id)


def load_progress(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"done_dirs": [], "queued_dirs": [], "failed_dirs": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(
    path: Path,
    done_dirs: Set[str],
    queued_dirs: Set[str],
    failed_dirs: Set[str],
    stats: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    data = {
        "done_dirs": list(done_dirs),
        "queued_dirs": list(queued_dirs),
        "failed_dirs": list(failed_dirs),
    }
    if stats:
        data["stats"] = stats
    if extra:
        data.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_share_links(path: str) -> List[ShareItem]:
    items: List[ShareItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|||")
            if len(parts) >= 2:
                title = parts[0].strip()
                link = parts[1].strip()
                password = parts[2].strip() if len(parts) >= 3 and parts[2].strip() else None
                items.append(ShareItem(title=title, link=link, password=password))
    return items


async def main():
    input_file = "share_links.txt"
    if not Path(input_file).exists():
        print(f"未找到 {input_file}")
        return

    items = load_share_links(input_file)
    if not items:
        print("❌ 未找到有效的分享链接")
        return

    batch = BatchExtractor(output_dir="output")
    try:
        await batch.run(items)
    finally:
        await batch.close()


if __name__ == "__main__":
    asyncio.run(main())
