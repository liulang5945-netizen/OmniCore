"""
模型下载管理器
支持从 HuggingFace 下载 GGUF 量化模型，带进度条、断点续传和镜像加速
v2.1 - 增强版：镜像自动切换、指数退避重试、网络诊断、SSL 可选
"""
import os
import sys
import ssl
import socket
import logging
import hashlib
import json
import time
import tempfile
import shutil
from typing import Optional, Callable, List, Tuple
from dataclasses import dataclass, field
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError

logger = logging.getLogger("ModelDownloader")


# ======================== 自定义异常 ========================

class RateLimitError(RuntimeError):
    """HTTP 429 请求限流异常，携带更长的建议等待时间"""
    def __init__(self, message: str = ""):
        super().__init__(message)
        self.is_rate_limit = True


# ======================== 配置 ========================

# 镜像源列表（按优先级排列：先镜像，后官方）
MIRROR_URLS = [
    ("hf-mirror.com", "https://hf-mirror.com"),
    ("huggingface.co", "https://huggingface.co"),
]

HF_OFFICIAL_URL = "https://huggingface.co"

# 分块下载大小 4MB（平衡：响应速度 vs 网络吞吐量）
CHUNK_SIZE = 4 * 1024 * 1024

# 默认模型保存目录
DEFAULT_MODEL_DIR = os.path.join(os.path.expanduser("~"), "OmniCore", "models")

# 重试配置
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2  # 指数退避基数（秒）: 2^1=2s, 2^2=4s, 2^3=8s

# HTTP 429 限流重试配置（更长的退避时间）
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_BASE = 10  # 429 限流时：30s, 60s, 120s, 240s, 480s
REQUEST_INTERVAL = 0  # 请求间隔（0=不延迟）（秒），避免短时间内多发请求

# 网络超时配置
CONNECT_TIMEOUT = 120  # 连接超时（秒）
READ_TIMEOUT = 30      # 读取超时（秒）注意：不使用全局 socket.setdefaulttimeout

# ======================== 工具函数 ========================

def _check_url_reachable(url: str, timeout: int = 10) -> Tuple[bool, str]:
    """检查 URL 是否可达，返回 (是否可达, 错误消息)"""
    try:
        ctx = ssl.create_default_context()
        req = Request(url, headers={"User-Agent": "OmniCore/2.0"})
        # 测试小范围读取（只读取 1KB 的头响应）
        with urlopen(req, timeout=timeout, context=ctx) as resp:
            resp.read(1024)
        return True, ""
    except Exception as e:
        return False, str(e)


def diagnose_network() -> dict:
    """网络诊断：检测镜像和官方源的连通性"""
    results = {
        "mirrors": [],
        "recommendation": "",
        "overall_status": "unknown",
    }
    
    test_urls = [
        ("hf-mirror.com", "https://hf-mirror.com"),
        ("huggingface.co", "https://huggingface.co"),
    ]
    
    reachable_count = 0
    for name, url in test_urls:
        ok, err = _check_url_reachable(url, timeout=8)
        status = "reachable" if ok else f"unreachable: {err[:80]}"
        results["mirrors"].append({"name": name, "url": url, "status": status})
        if ok:
            reachable_count += 1
    
    if reachable_count == 2:
        results["overall_status"] = "all_ok"
        results["recommendation"] = "所有源均可达，优先使用国内镜像加速"
    elif reachable_count == 1:
        results["overall_status"] = "partial"
        if results["mirrors"][0]["status"] == "reachable":
            results["recommendation"] = "国内镜像可用，使用镜像下载"
        else:
            results["recommendation"] = "国内镜像不可用，将使用官方源下载"
    else:
        results["overall_status"] = "all_down"
        results["recommendation"] = "所有源均不可达，请检查网络连接或代理设置"
    
    return results


@dataclass
class DownloadProgress:
    """下载进度"""
    filename: str = ""
    total_bytes: int = 0
    downloaded_bytes: int = 0
    speed_mbps: float = 0.0
    eta_seconds: float = 0.0
    status: str = "idle"  # idle | downloading | verifying | completed | error
    error_message: str = ""
    current_mirror: str = ""  # 当前使用的镜像源
    retry_count: int = 0      # 当前重试次数
    
    @property
    def percent(self) -> float:
        if self.total_bytes > 0:
            return min(100.0, self.downloaded_bytes / self.total_bytes * 100)
        return 0.0
    
    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)
    
    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / (1024 * 1024)


class ModelDownloader:
    """模型下载器 v2.1 - 增强版"""
    
    def __init__(self, save_dir: str = "", mirror: bool = True, verify_ssl: bool = True):
        """
        Args:
            save_dir: 模型保存目录
            mirror: 是否优先使用国内镜像
            verify_ssl: 是否验证 SSL 证书（在某些网络环境下需要关闭）
        """
        self.save_dir = save_dir or DEFAULT_MODEL_DIR
        self.prefer_mirror = mirror  # 是否优先使用镜像
        self.verify_ssl = verify_ssl
        self.progress = DownloadProgress()
        self._cancel_flag = False
        self._pause_flag = False
        self._current_source_index = 0  # 当前使用的镜像源索引
        os.makedirs(self.save_dir, exist_ok=True)
    
    def cancel(self):
        """取消下载"""
        self._cancel_flag = True
        self._pause_flag = False
        self.progress.status = "idle"
    
    def pause(self):
        """暂停下载（保留已下载部分，可恢复）"""
        self._pause_flag = True
        self.progress.status = "paused"
    
    def resume(self):
        """恢复下载"""
        self._pause_flag = False
        self.progress.status = "downloading"
    
    def _get_mirror_urls(self) -> List[Tuple[str, str]]:
        """获取按优先级排序的镜像源列表"""
        if self.prefer_mirror:
            return MIRROR_URLS
        else:
            # 优先使用官方源
            return [("huggingface.co", HF_OFFICIAL_URL),
                    ("hf-mirror.com", "https://hf-mirror.com")]
    
    def _build_url(self, repo_id: str, filename: str, base_url: str) -> str:
        """构建下载 URL"""
        return f"{base_url}/{repo_id}/resolve/main/{filename}"
    
    def _get_file_path(self, model_name: str, filename: str) -> str:
        """获取模型文件保存路径"""
        safe_name = model_name.replace("/", "_").replace("\\", "_")
        model_dir = os.path.join(self.save_dir, safe_name)
        os.makedirs(model_dir, exist_ok=True)
        return os.path.join(model_dir, filename)
    
    def _create_ssl_context(self) -> Optional[ssl.SSLContext]:
        """创建 SSL 上下文"""
        if not self.verify_ssl:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            return ctx
        return None  # 使用默认验证
    
    def _verify_sha256(self, file_path: str, expected_hash: str = "") -> bool:
        """验证文件 SHA256"""
        if not expected_hash:
            return True
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        actual = sha.hexdigest()
        if actual != expected_hash:
            logger.error(f"SHA256 不匹配: 期望 {expected_hash[:16]}..., 实际 {actual[:16]}...")
            return False
        return True
    
    def _parse_content_length(self, resp_headers: dict, downloaded: int) -> int:
        """
        正确解析文件总大小
        优先使用 Content-Range（断点续传响应），其次 Content-Length
        """
        content_range = resp_headers.get("Content-Range", "")
        if "/" in content_range and downloaded > 0:
            # HTTP 206 响应: "bytes 123456-987654/987655"
            try:
                total = int(content_range.split("/")[-1])
                return total
            except (ValueError, IndexError):
                pass
        
        # 使用 Content-Length
        content_length = resp_headers.get("Content-Length", "")
        if content_length:
            try:
                return int(content_length)
            except ValueError:
                pass
        
        return 0
    
    def download_file(
        self,
        repo_id: str,
        filename: str,
        model_name: str = "",
        sha256: str = "",
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> str:
        """
        下载单个模型文件（增强版：镜像切换 + 429 限流专用重试）
        
        Args:
            repo_id: HuggingFace 仓库 ID，如 "unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF"
            filename: 文件名，如 "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf"
            model_name: 模型显示名称
            sha256: 可选 SHA256 校验值
            progress_callback: 进度回调函数
        
        Returns:
            下载完成的文件路径
        
        Raises:
            RuntimeError: 所有镜像源均下载失败
        """
        self.progress = DownloadProgress(filename=filename)
        self.progress.status = "downloading"
        self._cancel_flag = False
        
        file_path = self._get_file_path(model_name or repo_id, filename)
        
        # 检查是否已存在完整文件
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            if file_size > 0:
                logger.info(f"模型文件已存在: {file_path}")
                self.progress.total_bytes = file_size
                self.progress.downloaded_bytes = file_size
                self.progress.status = "completed"
                if progress_callback:
                    progress_callback(self.progress)
                return file_path
        
        # 获取所有镜像源
        mirror_urls = self._get_mirror_urls()
        last_error = None
        
        # 遍历镜像源（每个源最多重试 MAX_RETRIES 次普通重试 + RATE_LIMIT_RETRIES 次限流重试）
        for mirror_idx, (mirror_name, mirror_url) in enumerate(mirror_urls):
            self.progress.current_mirror = mirror_name
            logger.info(f"尝试镜像源 [{mirror_idx + 1}/{len(mirror_urls)}]: {mirror_name}")
            
            # 全局 429 限流重试计数器（跨普通重试和镜像源切换）
            rate_limit_attempt = 0
            
            for attempt in range(MAX_RETRIES):
                if self._cancel_flag:
                    logger.info("下载已取消")
                    self.progress.status = "idle"
                    return ""
                
                self.progress.retry_count = attempt
                
                try:
                    if attempt > 0:
                        wait_time = RETRY_BACKOFF_BASE ** attempt
                        logger.info(f"第 {attempt + 1} 次重试，等待 {wait_time}s...")
                        # 更新进度为等待状态
                        self.progress.error_message = f"等待 {wait_time}s 后重试..."
                        if progress_callback:
                            progress_callback(self.progress)
                        time.sleep(wait_time)
                    
                    # 请求间隔（避免短时间内多发请求触发限流）
                    time.sleep(REQUEST_INTERVAL)
                    
                    url = self._build_url(repo_id, filename, mirror_url)
                    logger.info(f"开始下载: {url} (尝试 {attempt + 1}/{MAX_RETRIES})")
                    
                    file_path = self._do_download(
                        url=url,
                        file_path=file_path,
                        progress_callback=progress_callback,
                    )
                    
                    # 检查取消（_do_download 返回空且 self.progress.status == "idle"）
                    if self._cancel_flag or self.progress.status == "idle":
                        self.progress.status = "cancelled"
                        self.progress.error_message = "下载已取消"
                        if progress_callback:
                            progress_callback(self.progress)
                        logger.info("GGUF 下载已取消")
                        return ""
                    
                    if file_path and self.progress.status == "completed":
                        # 下载成功
                        logger.info(f"下载完成: {mirror_name} -> {file_path}")
                        break
                    
                except RateLimitError as e:
                    if self._cancel_flag:
                        break
                    # HTTP 429 限流特殊处理：使用更长的退避时间
                    rate_limit_attempt += 1
                    last_error = str(e)
                    logger.warning(f"触发限流 [{mirror_name}][第 {rate_limit_attempt} 次限流]")
                    
                    if rate_limit_attempt <= RATE_LIMIT_RETRIES:
                        wait_time = RATE_LIMIT_BACKOFF_BASE * (2 ** (rate_limit_attempt - 1))
                        msg = (
                            f"⏳ 请求过于频繁 (HTTP 429)，请等待 {wait_time}s 后自动重试 "
                            f"(第 {rate_limit_attempt}/{RATE_LIMIT_RETRIES} 次限流等待)\n"
                            f"💡 建议：减少同时下载的模型数量，或稍后再试"
                        )
                        self.progress.error_message = msg
                        self.progress.retry_count = attempt
                        if progress_callback:
                            progress_callback(self.progress)
                        logger.info(f"429 限流，等待 {wait_time}s 后重试...")
                        time.sleep(wait_time)
                        continue  # 在当前镜像源继续重试
                    else:
                        logger.warning(f"限流重试已达上限 ({RATE_LIMIT_RETRIES})，切换到下一个镜像源")
                        break  # 尝试下一个镜像源
                    
                except Exception as e:
                    if self._cancel_flag:
                        break
                    last_error = str(e)
                    logger.warning(f"下载尝试失败 [{mirror_name}][尝试 {attempt + 1}]: {e}")
                    self.progress.error_message = f"[{mirror_name}] {e}"
                    
                    if attempt == MAX_RETRIES - 1:
                        logger.warning(f"镜像源 {mirror_name} 所有重试已耗尽")
                        break  # 尝试下一个镜像源
                    continue
            
            if self.progress.status == "completed":
                break
            
            # 镜像源切换前等待，避免立即触发下一个源的限流
            if mirror_idx < len(mirror_urls) - 1:
                logger.info("切换镜像源前等待 2s...")
                time.sleep(2)
        
        # 再次确认取消状态
        if self._cancel_flag:
            self.progress.status = "cancelled"
            self.progress.error_message = "下载已取消"
            if progress_callback:
                progress_callback(self.progress)
            return ""

        if self.progress.status != "completed":
            self.progress.error_message = (
                f"所有下载源均失败，最后错误: {last_error or '未知错误'}\n"
                f"请检查网络连接，或尝试使用代理 / VPN\n"
                f"如果提示 HTTP 429，请等待几分钟后重试"
            )
            raise RuntimeError(self.progress.error_message)
        
        # SHA256 校验
        if sha256:
            self.progress.status = "verifying"
            if progress_callback:
                progress_callback(self.progress)
            if not self._verify_sha256(file_path, sha256):
                self.progress.status = "error"
                self.progress.error_message = "SHA256 校验失败"
                os.remove(file_path)
                raise RuntimeError(f"SHA256 校验失败: {filename}")
        
        return file_path
    
    
    def download_hf_model(
        self,
        repo_id: str,
        model_name: str = "",
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> str:
        """
        下载 HuggingFace 完整模型（逐文件下载，带实时进度、速度、ETA）

        改进：预取仓库文件大小 -> 直接 URL 下载（urllib 逐块）-> 实时回调每块的进度/速度/ETA。

        Args:
            repo_id: HuggingFace 仓库 ID
            model_name: 模型显示名称
            progress_callback: 进度回调 (实时报告 total_mb, downloaded_mb, speed_mbps, eta_seconds)

        Returns:
            下载完成的目录路径，失败返回空字符串
        """
        save_dir = os.path.join(self.save_dir, repo_id.replace("/", "_"))
        os.makedirs(save_dir, exist_ok=True)

        try:
            from huggingface_hub import list_repo_files
            from huggingface_hub.hf_api import HfApi

            mirror_base = "https://hf-mirror.com" if self.prefer_mirror else "https://huggingface.co"

            # ── 第一步：获取文件列表 ──
            logger.info(f"获取 HF 模型文件列表: {repo_id}")
            try:
                all_files = list(list_repo_files(repo_id))
            except Exception:
                all_files = [f.rfilename for f in HfApi().list_repo_files(repo_id)]

            if not all_files:
                raise RuntimeError(f"仓库 {repo_id} 没有文件")

            # 过滤无关文件
            download_files = [f for f in all_files
                            if not f.startswith(".")
                            and "flax_model" not in f
                            and "tf_model" not in f
                            and "onnx" not in f.lower()]

            # ── 第二步：预取所有文件大小（构建总进度基线）──
            file_size_map: dict = {}  # filename -> bytes
            logger.info(f"正在获取 {len(download_files)} 个文件的大小信息...")
            try:
                paths_info = HfApi().get_paths_info(
                    repo_id=repo_id,
                    paths=download_files,
                )
                for pi in paths_info:
                    size = getattr(pi, 'size', 0) or getattr(pi, 'lfs', {}).get('size', 0) or 0
                    file_size_map[pi.path] = size
            except Exception:
                logger.warning("get_paths_info 失败，将使用估算值（进度百分比可能不完全准确）")

            total_bytes_preset = sum(file_size_map.values()) if file_size_map else 0
            logger.info(
                f"需下载 {len(download_files)} 个文件"
                + (f"，总计 {total_bytes_preset / (1024**3):.1f} GB" if total_bytes_preset > 0 else "（大小未知）")
            )

            # ── 第三步：初始化进度 ──
            self.progress = DownloadProgress(filename=f"{repo_id} (HF)")
            self.progress.status = "downloading"
            self.progress.total_bytes = total_bytes_preset
            self.progress.downloaded_bytes = 0
            if progress_callback:
                progress_callback(self.progress)

            session_start = time.time()
            total_downloaded = 0
            file_count = 0

            for fname in download_files:
                if self._cancel_flag:
                    self.progress.status = "idle"
                    return ""

                file_count += 1
                fpath = os.path.join(save_dir, fname)

                # 跳过已存在的完整文件（校验大小，防止残留半截文件导致加载崩溃）
                known_size = file_size_map.get(fname, 0)
                if os.path.exists(fpath):
                    fsize = os.path.getsize(fpath)
                    if fsize > 0 and (known_size == 0 or fsize >= known_size):
                        # 文件完整（大小匹配或未知大小时非空），跳过
                        total_downloaded += fsize
                        self.progress.downloaded_bytes = total_downloaded
                        if self.progress.total_bytes == 0:
                            self.progress.total_bytes = int(fsize * len(download_files) / max(file_count, 1))
                        logger.info(f"  [{file_count}/{len(download_files)}] {fname} 已存在 ({fsize/(1024**2):.0f}MB)，跳过")
                        continue
                    elif known_size > 0 and fsize < known_size:
                        # 文件不完整（半截），删除后重新下载
                        logger.warning(f"  [{file_count}/{len(download_files)}] {fname} 不完整 ({fsize/(1024**2):.0f}MB/{known_size/(1024**2):.0f}MB)，重新下载")
                        try:
                            os.remove(fpath)
                        except Exception:
                            pass

                os.makedirs(os.path.dirname(fpath), exist_ok=True)

                # 从预取中获取此文件的大小
                known_size = file_size_map.get(fname, 0)

                # ── 第四步：逐文件下载（直接 URL，获得逐块实时进度）──
                file_ok = False
                last_file_error = ""
                mirrors = [mirror_base] if mirror_base else ["https://hf-mirror.com", "https://huggingface.co"]

                for mirror_url in mirrors:
                    if file_ok:
                        break
                    url = f"{mirror_url}/{repo_id}/resolve/main/{fname}"

                    for attempt in range(MAX_RETRIES):
                        if self._cancel_flag:
                            self.progress.status = "idle"
                            return ""

                        try:
                            # 保留部分文件以支持断点续传。_do_download 发送 Range 头从断点继续。
                            # 仅在网络错误/连接重置时保留；如果文件存在且不完整，让 HTTP 206 接管。
                            # 注意：仅在 HTTP 错误（如404/403）时才清理文件，网络错误保留。
                            size_str = f" ({known_size / (1024**2):.0f}MB)" if known_size > 0 else ""
                            logger.info(
                                f"  [{file_count}/{len(download_files)}] {fname}{size_str}"
                                f" (mirror={mirror_url.rsplit('/', 1)[-1]}, attempt={attempt + 1})"
                            )

                            self.progress.error_message = f"[{file_count}/{len(download_files)}] {fname[:80]}"

                            # 单文件进度对象（隔离，不污染 self.progress 全局状态）
                            per_file_prog = DownloadProgress(filename=fname)
                            per_file_prog.status = "downloading"
                            downloaded_before = [0]

                            def per_file_callback(prog):
                                """将单文件逐块进度映射到全局进度"""
                                added = prog.downloaded_bytes - downloaded_before[0]
                                nonlocal total_downloaded
                                total_downloaded += added
                                downloaded_before[0] = prog.downloaded_bytes
                                self.progress.downloaded_bytes = total_downloaded
                                # 修正总大小
                                if self.progress.total_bytes == 0 and known_size > 0 and file_count > 0:
                                    remaining_files = len(download_files) - file_count + 1
                                    self.progress.total_bytes = int(
                                        total_downloaded + known_size * remaining_files
                                    )
                                elif self.progress.total_bytes == 0 and prog.total_bytes > 0:
                                    self.progress.total_bytes = int(prog.total_bytes * len(download_files))
                                # 全局速度 + ETA
                                elapsed = time.time() - session_start
                                if elapsed > 0 and total_downloaded > 0:
                                    self.progress.speed_mbps = (total_downloaded / elapsed) / (1024 * 1024)
                                    remaining = max(0, self.progress.total_bytes - total_downloaded)
                                    self.progress.eta_seconds = (
                                        remaining / (total_downloaded / elapsed) if total_downloaded > 0 else 0
                                    )
                                if progress_callback:
                                    progress_callback(self.progress)

                            result_path = self._do_download(
                                url=url,
                                file_path=fpath,
                                progress_callback=per_file_callback,
                                progress=per_file_prog,
                            )

                            # 检查是否被取消
                            if self._cancel_flag or self.progress.status == "idle":
                                self.progress.status = "cancelled"
                                self.progress.error_message = "下载已取消"
                                if progress_callback:
                                    progress_callback(self.progress)
                                logger.info("HF 模型下载已取消")
                                return ""

                            if result_path and os.path.exists(fpath) and os.path.getsize(fpath) > 0:
                                fsize = os.path.getsize(fpath)
                                # 确保总下载量正确
                                if fsize > downloaded_before[0]:
                                    total_downloaded += fsize - downloaded_before[0]
                                self.progress.downloaded_bytes = total_downloaded
                                file_ok = True
                                break

                        except RateLimitError:
                            last_file_error = "HTTP 429 限流"
                            if self._cancel_flag:
                                break
                            if attempt < MAX_RETRIES - 1:
                                wait = RATE_LIMIT_BACKOFF_BASE * (2 ** attempt)
                                logger.warning(f"    限流，等待 {wait}s")
                                time.sleep(wait)
                            else:
                                break
                        except Exception as e:
                            last_file_error = str(e)
                            if self._cancel_flag:
                                break
                            # 网络错误保留部分文件，支持断点续传
                            partial_exists = os.path.exists(fpath)
                            partial_size = os.path.getsize(fpath) if partial_exists else 0
                            if partial_size > 0:
                                logger.info(f"    网络错误但已保留 {partial_size/(1024**2):.1f}MB，下次将断点续传")
                            wait = RETRY_BACKOFF_BASE ** attempt
                            logger.warning(f"    尝试 {attempt + 1}/{MAX_RETRIES} 失败: {e}")
                            if attempt < MAX_RETRIES - 1:
                                time.sleep(wait)

                # 再次检查取消状态（可能在重试等待中被取消）
                if self._cancel_flag:
                    self.progress.status = "cancelled"
                    self.progress.error_message = "下载已取消"
                    if progress_callback:
                        progress_callback(self.progress)
                    logger.info("HF 模型下载已取消")
                    return ""

                if not file_ok:
                    raise RuntimeError(
                        f"文件下载失败: {fname} (已尝试所有镜像源和重试。最后错误: {last_file_error})\n"
                        f"请检查网络连接或稍后重试"
                    )

                # 每个文件完成后更新一次全局标题
                elapsed = time.time() - session_start
                if elapsed > 0 and total_downloaded > 0:
                    self.progress.speed_mbps = (total_downloaded / elapsed) / (1024 * 1024)
                    remaining = max(0, self.progress.total_bytes - total_downloaded)
                    self.progress.eta_seconds = (
                        remaining / (total_downloaded / elapsed) if total_downloaded > 0 else 0
                    )

                self.progress.status = "downloading"
                self.progress.error_message = f"[{file_count}/{len(download_files)}] {fname[:80]}"
                if progress_callback:
                    progress_callback(self.progress)

            # ── 第五步：验证与结算 ──
            if not os.path.exists(os.path.join(save_dir, "config.json")):
                raise RuntimeError(f"下载完成但缺少 config.json: {save_dir}")

            final_total = sum(
                os.path.getsize(os.path.join(dp, fn))
                for dp, _, fns in os.walk(save_dir) for fn in fns
            )
            self.progress.total_bytes = final_total
            self.progress.downloaded_bytes = final_total
            self.progress.status = "completed"
            self.progress.speed_mbps = 0
            self.progress.eta_seconds = 0
            self.progress.error_message = ""
            if progress_callback:
                progress_callback(self.progress)

            logger.info(f"HF 模型下载完成: {save_dir} ({final_total / (1024**3):.1f} GB)")
            return save_dir

        except ImportError:
            raise RuntimeError("需要安装 huggingface_hub: pip install huggingface_hub") from None
        except Exception as e:
            logger.error(f"HF 模型下载失败: {e}")
            if progress_callback:
                self.progress.status = "error"
                self.progress.error_message = str(e)[:200]
                progress_callback(self.progress)
            return ""

    def _do_download(
        self,
        url: str,
        file_path: str,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        progress: Optional[DownloadProgress] = None,
    ) -> str:
        """执行实际的下载操作（使用 urllib，Python 内置，无需额外依赖）
        
        Args:
            progress: 可选的外部进度对象。不为 None 时，进度写入该对象而非 self.progress。
                      当 download_hf_model 调用本方法时传入独立对象，避免覆盖全局进度。
        
        注意：不使用全局 socket.setdefaulttimeout，改为通过 resp 对象的底层 socket 设置超时，
        避免影响其他线程的网络请求。
        """
        ssl_ctx = self._create_ssl_context()
        prog = progress if progress is not None else self.progress
        
        downloaded = 0
        headers = {"User-Agent": "OmniCore/2.0"}
        
        # 断点续传：检查已有文件大小
        if os.path.exists(file_path):
            downloaded = os.path.getsize(file_path)
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                logger.info(f"断点续传: 从 {downloaded} 字节继续")
        
        try:
            req = Request(url, headers=headers)
            
            with urlopen(req, timeout=CONNECT_TIMEOUT, context=ssl_ctx) as resp:
                status_code = resp.getcode()
                
                # 特殊处理 HTTP 429 限流
                if status_code == 429:
                    retry_after = resp.headers.get("Retry-After", "")
                    msg = f"请求过于频繁 (HTTP 429)，来自 {url}"
                    if retry_after:
                        msg += f"，建议等待 {retry_after}s"
                    logger.warning(msg)
                    prog.status = "error"
                    prog.error_message = msg
                    raise RateLimitError(msg)
                
                if status_code not in (200, 206):
                    raise RuntimeError(f"下载失败: HTTP {status_code} 来自 {url}")
                
                # 获取总大小
                resp_headers = dict(resp.headers)
                total = self._parse_content_length(resp_headers, downloaded)
                
                if downloaded == 0:
                    prog.total_bytes = total
                elif total > 0:
                    prog.total_bytes = total
                
                # 确定写入模式
                mode = "ab" if downloaded > 0 else "wb"
                start_time = time.time()
                last_update = 0.0  # 强制首次回调立即触发
                
                # 立即发出首次回调（前端可立即获取 total_mb）
                if progress_callback and total > 0:
                    prog.total_bytes = total
                    prog.speed_mbps = 0.0
                    prog.eta_seconds = 0.0
                    progress_callback(prog)
                
                # 安全地设置底层 socket 的读取超时（不影响全局）
                try:
                    sock = resp.fp.raw._sock if hasattr(resp.fp, 'raw') else resp.fp
                    if hasattr(sock, 'settimeout'):
                        sock.settimeout(READ_TIMEOUT)
                except Exception:
                    pass  # 如果无法设置，使用默认行为
                
                with open(file_path, mode) as f:
                    while True:
                        if self._cancel_flag:
                            logger.info("下载已取消")
                            prog.status = "idle"
                            self.progress.status = "idle"  # 同步主进度，让调用方检测
                            return ""
                        
                        # 暂停：自旋等待恢复或取消
                        while self._pause_flag and not self._cancel_flag:
                            time.sleep(0.2)
                        if self._cancel_flag:
                            logger.info("下载已取消（暂停中）")
                            prog.status = "idle"
                            self.progress.status = "idle"
                            return ""
                        
                        chunk = resp.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        prog.downloaded_bytes = downloaded
                        
                        # 每 0.2 秒更新一次进度（小块后更频繁，保证速度/ETA 实时）
                        now = time.time()
                        if now - last_update >= 0.2:
                            elapsed = now - start_time
                            if elapsed > 0:
                                prog.speed_mbps = (downloaded / elapsed) / (1024 * 1024)
                                remaining = max(0, prog.total_bytes - downloaded)
                                prog.eta_seconds = (
                                    remaining / (downloaded / elapsed) if downloaded > 0 else 0
                                )
                            last_update = now
                            if progress_callback:
                                progress_callback(prog)
        
        except RateLimitError:
            raise  # 向上传递，由 download_file 的 429 专用重试逻辑处理
        except HTTPError as e:
            prog.status = "error"
            msg = f"HTTP 错误 {e.code}: {e.reason} (URL: {url})"
            prog.error_message = msg
            if e.code == 429:
                raise RateLimitError(msg)
            raise RuntimeError(msg)
        except URLError as e:
            prog.status = "error"
            prog.error_message = str(e)
            raise RuntimeError(f"网络错误: {e}")
        except ssl.SSLError as e:
            prog.status = "error"
            prog.error_message = f"SSL 错误: {e}"
            raise RuntimeError(f"SSL 证书验证失败: {e}（可尝试设置 verify_ssl=False）")
        except socket.timeout:
            prog.status = "error"
            prog.error_message = "连接超时"
            raise RuntimeError(f"连接超时 ({CONNECT_TIMEOUT}s)")
        except OSError as e:
            prog.status = "error"
            prog.error_message = str(e)
            raise RuntimeError(f"文件操作错误: {e}")
        
        prog.status = "completed"
        prog.downloaded_bytes = os.path.getsize(file_path)
        prog.total_bytes = prog.downloaded_bytes
        if progress_callback:
            progress_callback(prog)
        
        return file_path


def download_model(
    model_name: str,
    quant: str = "Q4_K_M",
    save_dir: str = "",
    progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    verify_ssl: bool = True,
) -> str:
    """
    便捷函数：下载指定模型
    
    Args:
        model_name: 模型名称，如 "DeepSeek-R1-Distill-Qwen-7B"
        quant: 量化级别，如 "Q4_K_M"
        save_dir: 保存目录
        progress_callback: 进度回调
        verify_ssl: 是否验证 SSL 证书
    
    Returns:
        下载完成的 .gguf 文件路径
    """
    from model.model_registry import get_model_download_info
    
    info = get_model_download_info(model_name, quant)
    if not info:
        raise ValueError(f"未找到模型: {model_name} (量化: {quant})")
    
    downloader = ModelDownloader(save_dir=save_dir, verify_ssl=verify_ssl)
    file_path = downloader.download_file(
        repo_id=info["repo"],
        filename=info["filename"],
        model_name=model_name,
        progress_callback=progress_callback,
    )
    
    # 保存下载记录
    _save_download_record(model_name, quant, file_path, info)
    
    return file_path


def _save_download_record(model_name: str, quant: str, file_path: str, info: dict):
    """保存模型下载记录"""
    import json
    record_dir = os.path.join(os.path.dirname(file_path), ".omnicore")
    os.makedirs(record_dir, exist_ok=True)
    record = {
        "model_name": model_name,
        "quant": quant,
        "file_path": file_path,
        "repo": info.get("repo", ""),
        "filename": info.get("filename", ""),
        "parameters_b": info.get("parameters_b", 0),
        "family": info.get("family", ""),
        "description": info.get("description", ""),
        "downloaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    record_path = os.path.join(record_dir, f"{model_name}_{quant}.json")
    with open(record_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)


def list_downloaded_models(save_dir: str = "") -> list:
    """列出已下载的模型"""
    save_dir = save_dir or DEFAULT_MODEL_DIR
    models = []
    if not os.path.exists(save_dir):
        return models
    
    for root, dirs, files in os.walk(save_dir):
        if ".omnicore" in dirs:
            record_dir = os.path.join(root, ".omnicore")
            for f in os.listdir(record_dir):
                if f.endswith(".json"):
                    try:
                        with open(os.path.join(record_dir, f), "r", encoding="utf-8") as fp:
                            record = json.load(fp)
                            models.append(record)
                    except Exception:
                        pass
    
    return sorted(models, key=lambda m: m.get("model_name", ""))