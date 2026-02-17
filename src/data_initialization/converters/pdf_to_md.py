import sys
from pathlib import Path

# 确保项目根目录在 Python 路径中
_project_root = Path(__file__).resolve().parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import asyncio
import logging
import os
import subprocess
import time
import zipfile
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from urllib.parse import quote

import aiohttp
import oss2
import requests

from src.config.settings import (
    OSS_ACCESS_KEY_ID,
    OSS_ACCESS_KEY_SECRET,
    OSS_ENDPOINT,
    MINERU_API_KEY,
    MINERU_API_URL,
    OSS_BUCKET,
    PROJECT_ROOT,
    PDF_TO_MD_MAX_CONCURRENT_TASKS,
    PDF_TO_MD_TASK_INTERVAL,
    PDF_TO_MD_TASK_TIMEOUT,
)

# ---------------------- 类型与日志配置 ----------------------

PathLike = Union[str, os.PathLike]

logger = logging.getLogger(__name__)


def _win_long_path(path: Path) -> str:
    """Windows 下返回带长路径前缀的路径，避免超过 260 字符报错；非 Windows 返回 str(path)。"""
    resolved = path.resolve()
    if os.name != "nt":
        return str(resolved)
    s = str(resolved)
    if s.startswith("\\\\?\\"):
        return s
    if s.startswith("\\\\"):
        return "\\\\?\\UNC\\" + s[2:]
    return "\\\\?\\" + s


# ---------------------- 自定义异常 ----------------------


class MineruPdfConverterError(Exception):
    """Mineru PDF 转换过程中的统一异常基类。"""

    pass


class MineruApiError(MineruPdfConverterError):
    """MinerU API 调用相关异常。"""

    pass


class MineruCliError(MineruPdfConverterError):
    """本地 MinerU CLI 调用相关异常。"""

    pass


class DataInitializationError(Exception):
    """数据初始化过程中的统一异常基类。"""

    pass


class MineruConversionError(DataInitializationError):
    """MinerU PDF转换相关异常。"""

    pass


# ---------------------- 返回结果数据结构 ----------------------


@dataclass
class ConvertResult:
    """单个 PDF -> MD 转换结果（MinerU 内部使用）。"""

    md_content: str
    md_path: Path
    zip_path: Path


@dataclass
class PdfToMdResult:
    """
    PDF -> MD 转换结果（数据初始化专用，包含 layout.json 路径）。

    Attributes:
        md_content: Markdown文本内容
        md_path: Markdown文件路径
        zip_path: MinerU生成的zip文件路径
        layout_json_path: layout.json文件路径（如果存在）
        work_dir: MinerU工作目录路径
    """

    md_content: str
    md_path: Path
    zip_path: Path
    layout_json_path: Optional[Path]
    work_dir: Path


# ---------------------- MinerU PDF 转换器（原 preprocessing 逻辑） ----------------------


class MineruPdfConverter:
    """
    MinerU PDF -> Markdown 转换工具类。

    功能：上传 PDF 到 OSS、调用 MinerU API、下载 zip、解压得到 full.md，
    支持单文件/批量、同步/异步。供 PdfToMdConverter 复用。
    """

    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str,
        bucket_name: str,
        mineru_api_url: str,
        mineru_api_key: str,
        session: Optional[requests.Session] = None,
        max_concurrent_tasks: Optional[int] = None,
    ) -> None:
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.endpoint = endpoint
        self.bucket_name = bucket_name

        auth = oss2.Auth(self.access_key_id, self.access_key_secret)
        self.bucket = oss2.Bucket(auth, self.endpoint, self.bucket_name)

        self.mineru_api_url = mineru_api_url.rstrip("/")
        self.mineru_api_key = mineru_api_key

        self.session = session or requests.Session()
        self.session.trust_env = False

        self.max_concurrent_tasks = (
            max_concurrent_tasks
            if max_concurrent_tasks is not None
            else (PDF_TO_MD_MAX_CONCURRENT_TASKS or 5)
        )
        self.semaphore = asyncio.Semaphore(self.max_concurrent_tasks)

    def _upload_pdf_and_get_url_sync(self, local_path: PathLike) -> str:
        """同步上传本地 PDF 到 OSS，返回公网可访问的 URL。"""
        p = Path(local_path).resolve()
        if not p.is_file():
            raise FileNotFoundError(f"PDF not found: {p}")

        object_key = f"mineru_pdfs/{p.name}"
        logger.info("开始上传 PDF 到 OSS：%s -> %s", p, object_key)

        try:
            self.bucket.put_object_from_file(object_key, str(p))
        except Exception as e:
            logger.exception("上传到 OSS 失败：%s", p)
            raise MineruPdfConverterError(f"上传 PDF 到 OSS 失败: {p}") from e

        endpoint_host = self.endpoint.split("://", 1)[1]
        encoded_key = quote(object_key)
        url = f"https://{self.bucket_name}.{endpoint_host}/{encoded_key}"
        logger.debug("OSS 文件 URL: %s", url)
        return url

    async def _async_upload_pdf_and_get_url(self, local_path: PathLike) -> str:
        """异步上传本地 PDF 到 OSS，返回公网可访问的 URL。"""
        return await asyncio.to_thread(self._upload_pdf_and_get_url_sync, local_path)

    def _extract_full_md_from_zip_sync(
        self,
        zip_path: PathLike,
        extract_dir: PathLike,
    ) -> str:
        """同步解压 zip，找到 full.md，返回其内容。Windows 下使用长路径前缀避免超过 260 字符。"""
        zip_path = Path(zip_path)
        extract_dir = Path(extract_dir)
        extract_dir_io = _win_long_path(extract_dir)
        os.makedirs(extract_dir_io, exist_ok=True)

        logger.info("解压 zip 文件：%s -> %s", zip_path, extract_dir)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(extract_dir_io)
        except Exception as e:
            logger.exception("解压 zip 文件失败：%s", zip_path)
            raise MineruPdfConverterError(f"解压 zip 文件失败: {zip_path}") from e

        full_md_path: Optional[Path] = None
        for root, _, files in os.walk(extract_dir_io):
            if "full.md" in files:
                full_md_path = Path(root) / "full.md"
                break

        if full_md_path is None or not full_md_path.is_file():
            logger.error(
                "full.md 未在解压目录中找到，zip=%s, extract_dir=%s",
                zip_path,
                extract_dir,
            )
            raise FileNotFoundError("full.md not found in extracted zip")

        try:
            content = full_md_path.read_text(encoding="utf-8")
        except Exception as e:
            logger.exception("读取 full.md 失败：%s", full_md_path)
            raise MineruPdfConverterError(f"读取 full.md 失败: {full_md_path}") from e

        logger.debug("full.md 读取成功，长度=%d", len(content))
        return content

    async def _async_extract_full_md_from_zip(
        self,
        zip_path: PathLike,
        extract_dir: PathLike,
    ) -> str:
        """异步解压 zip，找到 full.md，返回其内容。"""
        return await asyncio.to_thread(
            self._extract_full_md_from_zip_sync, zip_path, extract_dir
        )

    @staticmethod
    def _find_pdf_paths_in_dir(pdf_dir: PathLike) -> List[Path]:
        """返回 pdf_dir 中所有 PDF 文件的绝对路径列表。"""
        pdf_dir_path = Path(pdf_dir)
        if not pdf_dir_path.exists():
            raise FileNotFoundError(f"目录不存在: {pdf_dir_path}")

        pdf_paths: List[Path] = [
            p.resolve()
            for p in pdf_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        ]

        if not pdf_paths:
            raise FileNotFoundError(f"目录中没有找到 PDF 文件: {pdf_dir_path}")

        return pdf_paths

    def pdf_to_md_local_mineru(
        self,
        input_pdf: PathLike,
        output_dir: PathLike,
        use_gpu: bool = False,
        device: str = "cuda",
    ) -> None:
        """调用本地 MinerU CLI 将单个 PDF 转成 Markdown。"""
        input_pdf_path = Path(input_pdf).resolve()
        if not input_pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {input_pdf_path}")

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)

        command = ["mineru", "-p", str(input_pdf_path), "-o", str(output_dir_path)]
        if use_gpu:
            command.extend(["--device", device])

        logger.info("执行本地 MinerU CLI 命令：%s", " ".join(command))

        try:
            subprocess.run(
                command,
                check=True,
                text=True,
                capture_output=True,
            )
            logger.info(
                "本地 MinerU CLI 转换成功：%s -> %s", input_pdf_path, output_dir_path
            )
        except FileNotFoundError as e:
            logger.exception("未找到 mineru 可执行文件")
            raise MineruCliError("未找到 mineru，可执行文件不在 PATH 中") from e
        except subprocess.CalledProcessError as e:
            logger.error(
                "本地 MinerU CLI 转换失败：%s, returncode=%s, stderr=%s",
                input_pdf_path,
                e.returncode,
                e.stderr,
            )
            raise MineruCliError(
                f"本地 MinerU CLI 转换失败: {input_pdf_path}, returncode={e.returncode}"
            ) from e

    def batch_pdf_to_md_local_mineru(
        self,
        pdf_dir: PathLike,
        md_dir: PathLike,
        use_gpu: bool = False,
        device: str = "cuda",
    ) -> None:
        """批量将 pdf_dir 下的所有 PDF 通过本地 MinerU CLI 转成 MD。"""
        pdf_dir_path = Path(pdf_dir)
        md_dir_path = Path(md_dir)
        md_dir_path.mkdir(parents=True, exist_ok=True)

        pdf_paths = self._find_pdf_paths_in_dir(pdf_dir_path)
        total = len(pdf_paths)
        logger.info("开始批量转换（本地 CLI 模式），共找到 %d 个 PDF 文件", total)

        success_count = 0
        fail_count = 0

        for i, pdf_path in enumerate(pdf_paths, start=1):
            logger.info("[%d/%d] 处理：%s", i, total, pdf_path)
            try:
                self.pdf_to_md_local_mineru(
                    input_pdf=pdf_path,
                    output_dir=md_dir_path,
                    use_gpu=use_gpu,
                    device=device,
                )
                success_count += 1
            except Exception as e:
                fail_count += 1
                logger.error("处理失败：%s, 错误信息：%r", pdf_path, e)

        logger.info(
            "本地 CLI 批量转换完成：总数=%d, 成功=%d, 失败=%d",
            total,
            success_count,
            fail_count,
        )

    def pdf_to_md_mineru_api(
        self,
        pdf_path: PathLike,
        zip_output_dir: PathLike,
        md_output_dir: PathLike,
        work_dir: PathLike = "./mineru_work",
        task_timeout: Optional[int] = None,
        task_interval: Optional[int] = None,
    ) -> ConvertResult:
        """同步调用 MinerU API 将单个 PDF 转换为 Markdown。"""
        return asyncio.run(
            self.async_pdf_to_md_mineru_api(
                pdf_path=pdf_path,
                zip_output_dir=zip_output_dir,
                md_output_dir=md_output_dir,
                work_dir=work_dir,
                task_timeout=task_timeout or (PDF_TO_MD_TASK_TIMEOUT or 600),
                task_interval=task_interval or (PDF_TO_MD_TASK_INTERVAL or 3),
            )
        )

    def batch_pdf_to_md_mineru_api(
        self,
        pdf_dir: PathLike,
        zip_output_dir: Optional[PathLike] = None,
        md_output_dir: Optional[PathLike] = None,
        work_dir: PathLike = "./mineru_work",
    ) -> None:
        """同步批量将 pdf_dir 下的所有 PDF 通过 MinerU API 转成 MD。"""
        asyncio.run(
            self.async_batch_pdf_to_md_mineru_api(
                pdf_dir=pdf_dir,
                zip_output_dir=zip_output_dir,
                md_output_dir=md_output_dir,
                work_dir=work_dir,
            )
        )

    async def _async_create_mineru_task(
        self, file_url: str, session: aiohttp.ClientSession
    ) -> str:
        """异步创建 MinerU 任务，返回 task_id。"""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mineru_api_key}",
        }
        data = {"url": file_url, "model_version": "vlm"}

        logger.info("创建 MinerU 任务，文件 URL: %s", file_url)
        try:
            async with session.post(
                self.mineru_api_url,
                headers=headers,
                json=data,
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                response.raise_for_status()
                j = await response.json()
        except Exception as e:
            logger.exception("调用 MinerU 创建任务接口失败")
            raise MineruApiError("调用 MinerU 创建任务接口失败") from e

        if j.get("code") != 0:
            logger.error("MinerU 创建任务失败，响应: %s", j)
            raise MineruApiError(f"create task failed: {j}")

        data_obj = j.get("data") or {}
        task_id = data_obj.get("task_id")
        if not task_id:
            logger.error("MinerU 创建任务响应中缺少 task_id，响应: %s", j)
            raise MineruApiError(f"create task failed: no task_id in response: {j}")

        logger.info("MinerU 任务创建成功，task_id=%s", task_id)
        return task_id

    async def _async_wait_mineru_done_and_get_zip_url(
        self,
        task_id: str,
        session: aiohttp.ClientSession,
        timeout: int = PDF_TO_MD_TASK_TIMEOUT or 600,
        interval: int = PDF_TO_MD_TASK_INTERVAL or 3,
    ) -> str:
        """异步轮询 MinerU 任务状态，直到 state == 'done'，返回 full_zip_url。"""
        url = f"{self.mineru_api_url}/{task_id}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.mineru_api_key}",
        }

        logger.info("开始轮询 MinerU 任务状态，task_id=%s", task_id)
        start = time.time()

        while True:
            try:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    response.raise_for_status()
                    j = await response.json()
            except Exception as e:
                logger.exception("查询 MinerU 任务状态失败，task_id=%s", task_id)
                raise MineruApiError(f"query task failed for task_id={task_id}") from e

            if j.get("code") != 0:
                logger.error("MinerU 任务状态查询失败，task_id=%s, 响应=%s", task_id, j)
                raise MineruPdfConverterError(f"query task failed for task_id={task_id}: {j}")

            data = j.get("data") or {}
            state = data.get("state")
            err_msg = data.get("err_msg", "")

            logger.debug("MinerU 任务状态，task_id=%s, state=%s", task_id, state)

            if state == "done":
                full_zip_url = data.get("full_zip_url")
                if not full_zip_url:
                    logger.error(
                        "state=done 但 data 中无 full_zip_url，task_id=%s, data=%s",
                        task_id,
                        data,
                    )
                    raise MineruApiError(
                        f"state=done but no full_zip_url in data for task_id={task_id}: {data}"
                    )
                logger.info(
                    "MinerU 任务完成，task_id=%s, zip_url=%s", task_id, full_zip_url
                )
                return full_zip_url

            if state in ("failed", "error"):
                logger.error(
                    "MinerU 任务失败，task_id=%s, state=%s, err_msg=%s",
                    task_id,
                    state,
                    err_msg,
                )
                raise MineruApiError(
                    f"task failed: task_id={task_id}, state={state}, err_msg={err_msg}"
                )

            if time.time() - start > timeout:
                logger.error(
                    "等待 MinerU 任务超时，task_id=%s, last_state=%s", task_id, state
                )
                raise MineruPdfConverterError(
                    f"wait task timeout for task_id={task_id}, last state={state}"
                )

            await asyncio.sleep(interval)

    async def _async_download_file(
        self, url: str, save_path: PathLike, session: aiohttp.ClientSession
    ) -> None:
        """异步下载远程文件到本地。"""
        save_path = Path(save_path)
        await asyncio.to_thread(save_path.parent.mkdir, parents=True, exist_ok=True)

        logger.info("开始下载文件：%s -> %s", url, save_path)
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                r.raise_for_status()
                chunks = []
                async for chunk in r.content.iter_chunked(8192):
                    if chunk:
                        chunks.append(chunk)
                await asyncio.to_thread(
                    lambda: save_path.write_bytes(b"".join(chunks))
                )
        except Exception as e:
            logger.exception("下载文件失败：%s", url)
            raise MineruPdfConverterError(f"下载文件失败: {url}") from e

    async def async_pdf_to_md_mineru_api(
        self,
        pdf_path: PathLike,
        zip_output_dir: PathLike,
        md_output_dir: PathLike,
        work_dir: PathLike = "./mineru_work",
        task_timeout: int = PDF_TO_MD_TASK_TIMEOUT or 600,
        task_interval: int = PDF_TO_MD_TASK_INTERVAL or 3,
    ) -> ConvertResult:
        """异步调用 MinerU API 将单个 PDF 转换为 Markdown 文件。"""
        pdf_path_obj = Path(pdf_path).resolve()
        if not await asyncio.to_thread(pdf_path_obj.is_file):
            raise FileNotFoundError(f"PDF not found: {pdf_path_obj}")

        logger.info("开始转换 %s", pdf_path_obj.name)

        base_name = pdf_path_obj.stem

        zip_output_dir = Path(zip_output_dir)
        md_output_dir = Path(md_output_dir)
        work_dir = Path(work_dir)

        await asyncio.to_thread(zip_output_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(md_output_dir.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(work_dir.mkdir, parents=True, exist_ok=True)

        md_path = md_output_dir / f"{base_name}.md"
        zip_path = zip_output_dir / f"{base_name}.zip"

        md_exists = await asyncio.to_thread(md_path.exists)
        md_is_file = await asyncio.to_thread(md_path.is_file) if md_exists else False

        if md_exists and md_is_file:
            logger.info(
                "目标文件已存在，跳过转换：%s -> %s",
                pdf_path_obj.name,
                md_path,
            )
            try:
                md_content = await asyncio.to_thread(
                    md_path.read_text, encoding="utf-8"
                )
                return ConvertResult(
                    md_content=md_content,
                    md_path=md_path,
                    zip_path=zip_path,
                )
            except Exception as e:
                logger.warning(
                    "读取已存在的目标文件失败，将重新转换：%s, 错误: %s",
                    md_path,
                    str(e),
                )

        async with aiohttp.ClientSession() as session:
            file_url = await self._async_upload_pdf_and_get_url(pdf_path_obj)
            task_id = await self._async_create_mineru_task(file_url, session)
            full_zip_url = await self._async_wait_mineru_done_and_get_zip_url(
                task_id=task_id,
                session=session,
                timeout=task_timeout,
                interval=task_interval,
            )

            zip_exists = await asyncio.to_thread(zip_path.exists)
            if not zip_exists:
                await self._async_download_file(full_zip_url, zip_path, session)
            else:
                logger.info("ZIP 文件已存在，跳过下载：%s", zip_path)

        extract_dir = work_dir / base_name
        md_content = await self._async_extract_full_md_from_zip(zip_path, extract_dir)

        md_path = md_output_dir / f"{base_name}.md"
        try:
            await asyncio.to_thread(
                md_path.write_text, md_content, encoding="utf-8"
            )
        except Exception as e:
            logger.exception("写入 markdown 文件失败：%s", md_path)
            raise MineruPdfConverterError(f"写入 markdown 文件失败: {md_path}") from e

        logger.info(
            "%s 转换完成：md_len=%d, md=%s, zip=%s",
            pdf_path_obj.name,
            len(md_content),
            md_path,
            zip_path,
        )

        return ConvertResult(
            md_content=md_content,
            md_path=md_path,
            zip_path=zip_path,
        )

    async def async_batch_pdf_to_md_mineru_api(
        self,
        pdf_dir: PathLike,
        zip_output_dir: Optional[PathLike] = None,
        md_output_dir: Optional[PathLike] = None,
        work_dir: PathLike = "./mineru_work",
        task_timeout: int = PDF_TO_MD_TASK_TIMEOUT or 600,
        task_interval: int = PDF_TO_MD_TASK_INTERVAL or 3,
    ) -> None:
        """异步批量将 pdf_dir 下的所有 PDF 通过 MinerU API 转成 MD。"""
        pdf_dir_path = Path(pdf_dir)

        if zip_output_dir is None:
            zip_output_dir = (
                PROJECT_ROOT / "files" / "file_store" / "zip_store" / "minerU_zip"
            )
        if md_output_dir is None:
            md_output_dir = PROJECT_ROOT / "files" / "file_store" / "md_store" / "raw_md"

        zip_output_dir_path = Path(zip_output_dir)
        md_output_dir_path = Path(md_output_dir)
        work_dir_path = Path(work_dir)

        await asyncio.to_thread(zip_output_dir_path.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(md_output_dir_path.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(work_dir_path.mkdir, parents=True, exist_ok=True)

        pdf_paths = self._find_pdf_paths_in_dir(pdf_dir_path)
        total = len(pdf_paths)
        file_names = ", ".join([p.name for p in pdf_paths])
        logger.info("开始批量转换 %s，共找到 %d 个文件", file_names, total)

        async def process_single_pdf(pdf_path: Path) -> Tuple[bool, bool]:
            async with self.semaphore:
                try:
                    base_name = pdf_path.stem
                    md_path = md_output_dir_path / f"{base_name}.md"
                    md_exists = await asyncio.to_thread(md_path.exists)
                    md_is_file = await asyncio.to_thread(md_path.is_file) if md_exists else False

                    if md_exists and md_is_file:
                        logger.info("跳过（文件已存在）：%s", pdf_path.name)
                        return (True, True)

                    await self.async_pdf_to_md_mineru_api(
                        pdf_path=pdf_path,
                        zip_output_dir=zip_output_dir_path,
                        md_output_dir=md_output_dir_path,
                        work_dir=work_dir_path,
                        task_timeout=task_timeout,
                        task_interval=task_interval,
                    )
                    return (True, False)
                except Exception as e:
                    logger.error("处理失败：%s, 错误信息：%r", pdf_path.name, e)
                    return (False, False)

        tasks = [process_single_pdf(pdf_path) for pdf_path in pdf_paths]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        skip_count = 0
        fail_count = 0

        for r in results:
            if isinstance(r, Exception):
                fail_count += 1
            elif isinstance(r, tuple):
                if r[0]:
                    success_count += 1
                    if r[1]:
                        skip_count += 1
                else:
                    fail_count += 1

        logger.info(
            "批量转换完成：总数=%d, 成功=%d, 跳过=%d, 失败=%d",
            total,
            success_count,
            skip_count,
            fail_count,
        )


# ---------------------- 数据初始化专用：带 layout.json 的 PDF -> MD 转换器 ----------------------


class PdfToMdConverter:
    """
    PDF -> Markdown 转换工具类（数据初始化专用）。

    功能：将 PDF 转为 MD，并在 MinerU 工作目录中查找 layout.json，
    返回 MD 路径与 layout.json 路径供后续处理使用。内部复用 MineruPdfConverter。
    """

    def __init__(
        self,
        access_key_id: str,
        access_key_secret: str,
        endpoint: str,
        bucket_name: str,
        mineru_api_url: str,
        mineru_api_key: str,
        max_concurrent_tasks: Optional[int] = None,
    ) -> None:
        self.mineru_converter = MineruPdfConverter(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            endpoint=endpoint,
            bucket_name=bucket_name,
            mineru_api_url=mineru_api_url,
            mineru_api_key=mineru_api_key,
            max_concurrent_tasks=max_concurrent_tasks,
        )

    def _find_layout_json_in_work_dir(
        self, work_dir: Path, pdf_name: Optional[str] = None
    ) -> Optional[Path]:
        """在 MinerU 工作目录中查找 layout.json 文件。"""
        if pdf_name:
            pdf_dir = work_dir / pdf_name
            layout_path = pdf_dir / "layout.json"
            if layout_path.exists() and layout_path.is_file():
                logger.debug("在PDF目录找到layout.json: %s", layout_path)
                return layout_path

        layout_path = work_dir / "layout.json"
        if layout_path.exists() and layout_path.is_file():
            logger.debug("在工作目录根目录找到layout.json: %s", layout_path)
            return layout_path

        for root, dirs, files in os.walk(work_dir):
            if "layout.json" in files:
                found_path = Path(root) / "layout.json"
                logger.debug("递归查找到layout.json: %s", found_path)
                return found_path

        logger.warning("未在工作目录中找到layout.json: %s", work_dir)
        return None

    async def async_convert_pdf_to_md_with_layout(
        self,
        pdf_path: PathLike,
        md_output_dir: PathLike,
        work_dir: PathLike,
        zip_output_dir: Optional[PathLike] = None,
        task_timeout: Optional[int] = None,
        task_interval: Optional[int] = None,
    ) -> PdfToMdResult:
        """异步转换 PDF 为 MD，并返回包含 layout.json 路径的结果。"""
        pdf_path_obj = Path(pdf_path).resolve()
        md_output_dir_path = Path(md_output_dir)
        work_dir_path = Path(work_dir)
        zip_output_dir_path = Path(zip_output_dir) if zip_output_dir else None

        await asyncio.to_thread(md_output_dir_path.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(work_dir_path.mkdir, parents=True, exist_ok=True)
        if zip_output_dir_path:
            await asyncio.to_thread(
                zip_output_dir_path.mkdir, parents=True, exist_ok=True
            )

        logger.info("开始转换PDF（数据初始化模式）: %s", pdf_path_obj.name)

        try:
            convert_result = await self.mineru_converter.async_pdf_to_md_mineru_api(
                pdf_path=pdf_path_obj,
                zip_output_dir=zip_output_dir_path
                or md_output_dir_path.parent / "zip_store",
                md_output_dir=md_output_dir_path,
                work_dir=work_dir_path,
                task_timeout=task_timeout or (PDF_TO_MD_TASK_TIMEOUT or 600),
                task_interval=task_interval or (PDF_TO_MD_TASK_INTERVAL or 3),
            )
        except Exception as e:
            logger.exception("MinerU转换失败: %s", pdf_path_obj)
            raise MineruConversionError(f"MinerU转换失败: {pdf_path_obj}") from e

        pdf_name = pdf_path_obj.stem
        layout_json_path = await asyncio.to_thread(
            self._find_layout_json_in_work_dir, work_dir_path, pdf_name
        )

        if layout_json_path is None:
            logger.warning(
                "转换完成但未找到layout.json，PDF: %s, work_dir: %s",
                pdf_path_obj.name,
                work_dir_path,
            )

        logger.info(
            "PDF转换完成: %s, md=%s, layout_json=%s",
            pdf_path_obj.name,
            convert_result.md_path,
            layout_json_path,
        )

        return PdfToMdResult(
            md_content=convert_result.md_content,
            md_path=convert_result.md_path,
            zip_path=convert_result.zip_path,
            layout_json_path=layout_json_path,
            work_dir=work_dir_path,
        )

    async def async_batch_convert_pdfs_with_layout(
        self,
        pdf_dir: PathLike,
        md_output_dir: Optional[PathLike] = None,
        work_dir: Optional[PathLike] = None,
        zip_output_dir: Optional[PathLike] = None,
        task_timeout: Optional[int] = None,
        task_interval: Optional[int] = None,
    ) -> List[PdfToMdResult]:
        """异步批量转换 PDF 为 MD，并返回包含 layout.json 路径的结果列表。"""
        pdf_dir_path = Path(pdf_dir)

        if md_output_dir is None:
            md_output_dir = (
                PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_md"
            )
        if work_dir is None:
            work_dir = (
                PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_work"
            )
        if zip_output_dir is None:
            zip_output_dir = (
                PROJECT_ROOT / "files" / "file_store" / "zip_store" / "minerU_zip"
            )

        md_output_dir_path = Path(md_output_dir)
        work_dir_path = Path(work_dir)
        zip_output_dir_path = Path(zip_output_dir)

        await asyncio.to_thread(md_output_dir_path.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(work_dir_path.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(zip_output_dir_path.mkdir, parents=True, exist_ok=True)

        pdf_paths = [
            p.resolve()
            for p in pdf_dir_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".pdf"
        ]

        if not pdf_paths:
            logger.warning("目录中没有找到PDF文件: %s", pdf_dir_path)
            return []

        total = len(pdf_paths)
        logger.info("开始批量转换PDF（数据初始化模式），共 %d 个文件", total)

        results: List[PdfToMdResult] = []
        success_count = 0
        fail_count = 0

        semaphore = asyncio.Semaphore(
            self.mineru_converter.max_concurrent_tasks
            or PDF_TO_MD_MAX_CONCURRENT_TASKS
            or 5
        )

        async def process_single_pdf(
            pdf_path: Path, index: int
        ) -> Optional[PdfToMdResult]:
            async with semaphore:
                logger.info("[%d/%d] 处理: %s", index, total, pdf_path.name)
                try:
                    result = await self.async_convert_pdf_to_md_with_layout(
                        pdf_path=pdf_path,
                        md_output_dir=md_output_dir_path,
                        work_dir=work_dir_path,
                        zip_output_dir=zip_output_dir_path,
                        task_timeout=task_timeout,
                        task_interval=task_interval,
                    )
                    return result
                except Exception as e:
                    logger.error("处理失败: %s, 错误: %r", pdf_path.name, e)
                    return None

        tasks = [
            process_single_pdf(pdf_path, i + 1) for i, pdf_path in enumerate(pdf_paths)
        ]
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in task_results:
            if isinstance(r, Exception):
                fail_count += 1
                logger.error("任务异常: %r", r)
            elif r is not None:
                results.append(r)
                success_count += 1
            else:
                fail_count += 1

        logger.info(
            "批量转换完成: 总数=%d, 成功=%d, 失败=%d",
            total,
            success_count,
            fail_count,
        )

        return results


# ---------------------- 独立运行调试示例 ----------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )

    converter = PdfToMdConverter(
        access_key_id=OSS_ACCESS_KEY_ID,
        access_key_secret=OSS_ACCESS_KEY_SECRET,
        endpoint=OSS_ENDPOINT,
        bucket_name=OSS_BUCKET,
        mineru_api_url=MINERU_API_URL,
        mineru_api_key=MINERU_API_KEY,
    )

    async def main():
        pdf_dir = PROJECT_ROOT / "files" / "file_store" / "pdf_store"
        md_dir = PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_md"
        work_dir = PROJECT_ROOT / "files" / "file_store" / "md_store" / "minerU_work"
        zip_dir = PROJECT_ROOT / "files" / "file_store" / "zip_store" / "minerU_zip"

        results = await converter.async_batch_convert_pdfs_with_layout(
            pdf_dir=pdf_dir,
            md_output_dir=md_dir,
            work_dir=work_dir,
            zip_output_dir=zip_dir,
        )

        logger.info("转换完成，共 %d 个结果", len(results))
        for result in results:
            logger.info(
                "MD: %s, Layout: %s",
                result.md_path,
                result.layout_json_path,
            )

    asyncio.run(main())
