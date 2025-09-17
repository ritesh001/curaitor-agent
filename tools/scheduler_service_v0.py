# scheduler_service.py
from __future__ import annotations
import asyncio
import logging
from typing import Optional
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from datetime import datetime
# from pytz import timezone  # if you prefer pytz; otherwise use zoneinfo
from zoneinfo import ZoneInfo

# ---- Singleton scheduler (module-level) ----
_scheduler: Optional[AsyncIOScheduler] = None

def get_scheduler() -> AsyncIOScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = AsyncIOScheduler(timezone=ZoneInfo("Europe/London"))  # set your TZ
        _scheduler.start()  # non-blocking
    elif not _scheduler.running:
        _scheduler.start()
    return _scheduler

# ---- Your job (could be any async function) ----
async def my_job():
    print(f"[{datetime.now()}] Async job executed at the scheduled time!")

# ---- API you’ll call from MCP tools ----
def schedule_daily_job(
    hour: int,
    minute: int,
    job_id: str = "daily_my_job",
    replace_existing: bool = True,
):
    """
    Add/replace a daily job at HH:MM. Returns immediately (non-blocking).
    """
    scheduler = get_scheduler()
    trigger = CronTrigger(hour=hour, minute=minute)  # uses scheduler's timezone
    scheduler.add_job(
        my_job,
        trigger=trigger,
        id=job_id,
        replace_existing=replace_existing,
        coalesce=True,  # collapse missed runs into one
        max_instances=1 # avoid overlapping runs
    )
    return f"Scheduled '{job_id}' daily at {hour:02d}:{minute:02d}."

def remove_job(job_id: str):
    scheduler = get_scheduler()
    scheduler.remove_job(job_id)
    return f"Removed job '{job_id}'."

def list_jobs():
    scheduler = get_scheduler()
    return [str(job) for job in scheduler.get_jobs()]

def shutdown_scheduler():
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)
    _scheduler = None
    return "Scheduler shut down."
