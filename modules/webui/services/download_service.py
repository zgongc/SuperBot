"""
Data Download Service

WebUI service for downloading historical data from Binance.
Uses DataDownloader component in background tasks.
"""

import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class DownloadTask:
    """Single download task"""
    id: str
    symbol: str
    timeframe: str
    start_date: str
    end_date: Optional[str]
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = "Waiting..."
    rows_downloaded: int = 0
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "rows_downloaded": self.rows_downloaded,
            "error": self.error
        }


@dataclass
class DownloadJob:
    """Download job containing multiple tasks"""
    id: str
    tasks: List[DownloadTask]
    created_at: datetime = field(default_factory=datetime.now)
    cancelled: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.id,
            "tasks": [t.to_dict() for t in self.tasks],
            "total": len(self.tasks),
            "completed": sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
            "running": sum(1 for t in self.tasks if t.status == TaskStatus.RUNNING),
            "pending": sum(1 for t in self.tasks if t.status == TaskStatus.PENDING),
            "errors": sum(1 for t in self.tasks if t.status == TaskStatus.ERROR),
            "cancelled": self.cancelled
        }


class DownloadService:
    """
    Data Download Service

    Manages background download tasks using DataDownloader.
    """

    def __init__(self, logger=None):
        self.logger = logger
        self._jobs: Dict[str, DownloadJob] = {}
        self._downloader = None
        self._running_tasks: Dict[str, asyncio.Task] = {}

    def _get_downloader(self):
        """Lazy load DataDownloader"""
        if self._downloader is None:
            try:
                from components.data.data_downloader import DataDownloader
                self._downloader = DataDownloader()
            except ImportError as e:
                if self.logger:
                    self.logger.error(f"Failed to import DataDownloader: {e}")
                raise
        return self._downloader

    def create_job(
        self,
        symbols: List[str],
        timeframes: List[str],
        start_date: str,
        end_date: Optional[str] = None
    ) -> DownloadJob:
        """
        Create a new download job

        Args:
            symbols: List of symbols to download
            timeframes: List of timeframes
            start_date: Start date (YYYY-MM-DD)
            end_date: End date or None for current time

        Returns:
            DownloadJob instance
        """
        job_id = str(uuid.uuid4())[:8]
        tasks = []

        for symbol in symbols:
            for timeframe in timeframes:
                task = DownloadTask(
                    id=f"{job_id}-{symbol}-{timeframe}",
                    symbol=symbol.upper(),
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                tasks.append(task)

        job = DownloadJob(id=job_id, tasks=tasks)
        self._jobs[job_id] = job

        if self.logger:
            self.logger.info(f"Created download job {job_id} with {len(tasks)} tasks")

        return job

    def start_job_sync(self, job_id: str) -> bool:
        """
        Start executing a download job (synchronous version for threading)

        Args:
            job_id: Job ID to start

        Returns:
            True if started successfully
        """
        job = self._jobs.get(job_id)
        if not job:
            return False

        # Process in current thread with new event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._process_job(job))
        finally:
            loop.close()

        return True

    async def _process_job(self, job: DownloadJob):
        """Process all tasks in a job sequentially"""
        try:
            downloader = self._get_downloader()

            for task in job.tasks:
                if job.cancelled:
                    task.status = TaskStatus.CANCELLED
                    task.message = "İptal edildi"
                    continue

                await self._process_task(downloader, task)

        except Exception as e:
            if self.logger:
                self.logger.error(f"Job {job.id} failed: {e}")

    async def _process_task(self, downloader, task: DownloadTask):
        """Process a single download task"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        task.message = "Başlatılıyor..."
        task.progress = 5

        try:
            task.message = "İndiriliyor..."
            task.progress = 10

            df = await downloader.download(
                symbol=task.symbol,
                timeframe=task.timeframe,
                start_date=task.start_date,
                end_date=task.end_date,
                output_dir="data/parquets"
            )

            task.progress = 100
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()

            # Get row count from result
            if df is not None and hasattr(df, '__len__'):
                if 'total_rows' in df.columns:
                    task.rows_downloaded = int(df['total_rows'].iloc[0])
                else:
                    task.rows_downloaded = len(df)
            else:
                task.rows_downloaded = 0

            task.message = f"{task.rows_downloaded:,} satır indirildi"

            if self.logger:
                self.logger.info(f"Task {task.id} completed: {task.rows_downloaded} rows")

        except Exception as e:
            task.status = TaskStatus.ERROR
            task.error = str(e)
            task.message = f"Hata: {str(e)[:50]}"
            task.completed_at = datetime.now()

            if self.logger:
                self.logger.error(f"Task {task.id} failed: {e}")

    def get_job(self, job_id: str) -> Optional[DownloadJob]:
        """Get job by ID"""
        return self._jobs.get(job_id)

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job status as dict"""
        job = self._jobs.get(job_id)
        if job:
            return job.to_dict()
        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        job = self._jobs.get(job_id)
        if not job:
            return False

        job.cancelled = True

        # Mark pending tasks as cancelled
        for task in job.tasks:
            if task.status == TaskStatus.PENDING:
                task.status = TaskStatus.CANCELLED
                task.message = "Cancelled"

        if self.logger:
            self.logger.info(f"Job {job_id} cancelled")

        return True

    def cleanup_old_jobs(self, max_age_hours: int = 24):
        """Remove old completed jobs"""
        now = datetime.now()
        to_remove = []

        for job_id, job in self._jobs.items():
            # Check if all tasks are done
            all_done = all(
                t.status in [TaskStatus.COMPLETED, TaskStatus.ERROR, TaskStatus.CANCELLED]
                for t in job.tasks
            )

            if all_done:
                age_hours = (now - job.created_at).total_seconds() / 3600
                if age_hours > max_age_hours:
                    to_remove.append(job_id)

        for job_id in to_remove:
            del self._jobs[job_id]

        if self.logger and to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old jobs")
