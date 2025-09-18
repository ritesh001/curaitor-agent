# scheduler_service.py
from __future__ import annotations

import asyncio
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore
from apscheduler.executors.pool import ThreadPoolExecutor, ProcessPoolExecutor
from apscheduler.events import (
    EVENT_JOB_EXECUTED,
    EVENT_JOB_ERROR,
    EVENT_JOB_MISSED,
    JobEvent,
)

import arxiv
import json
import os
from typing import List

PAPER_DIR = "data/agent_papers"

# ----------------------------
# Logging (console + file)
# ----------------------------
_logging_configured = False

def _configure_logging(level: int = logging.INFO) -> None:
    global _logging_configured
    if _logging_configured:
        return

    logger = logging.getLogger()
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler("data/tracker/scheduler.log", mode="a")  # append mode
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    _logging_configured = True


# ----------------------------
# Singleton scheduler (module-level)
# ----------------------------
_scheduler: Optional[AsyncIOScheduler] = None

def _job_listener(event: JobEvent) -> None:
    log = logging.getLogger("scheduler.listener")
    if event.code == EVENT_JOB_EXECUTED:
        log.info("Job %s executed", event.job_id)
    elif event.code == EVENT_JOB_ERROR:
        log.exception("Job %s errored", event.job_id)
    elif event.code == EVENT_JOB_MISSED:
        log.warning("Job %s MISSED (scheduler/process may have been paused)", event.job_id)

def get_scheduler() -> AsyncIOScheduler:
    """
    Lazily create and start a single AsyncIOScheduler instance for this process.
    Uses a persistent SQLite job store so jobs are visible across processes / restarts.
    """
    global _scheduler
    if _scheduler is not None and _scheduler.running:
        print("Scheduler already running.")
        return _scheduler

    _configure_logging()

    jobstores = {
        "default": SQLAlchemyJobStore(url="sqlite:///data/tracker/mcp_jobs.sqlite"),
    }
    executors = {
        "default": ThreadPoolExecutor(10),
        "processpool": ProcessPoolExecutor(2),
    }
    job_defaults = {
        "coalesce": True,
        "max_instances": 1,
        "misfire_grace_time": 300,  # seconds
    }

    _scheduler = AsyncIOScheduler(
        jobstores=jobstores,
        executors=executors,
        job_defaults=job_defaults,
        timezone=ZoneInfo("Europe/London"),
    )

    _scheduler.add_listener(
        _job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR | EVENT_JOB_MISSED
    )

    _scheduler.start()
    logging.getLogger("scheduler").info("Scheduler started (tz=Europe/London, persistent=SQLite)")
    return _scheduler


# ----------------------------
# Example job (async)
# ----------------------------
async def my_job_async():
    logging.getLogger("scheduler.job").info(
        "Async job executed at %s", datetime.now().isoformat(timespec="seconds")
    )
    with open("data/tracker/job_run_times.txt", "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')}\n")
    
    return print(f"[{datetime.now()}] Async job executed at the scheduled time!")

# async def send_whatsapp_message(message: str, to_number: str) -> None:
#     return

async def search_papersx(topic: str, max_results: int = 5) -> List[str]:
    """
    Search for papers on arXiv based on a topic and store their information.
    
    Args:
        topic: The topic to search for
        max_results: Maximum number of results to retrieve (default: 5)
        
    Returns:
        List of paper IDs found in the search
    """
    
    # Use arxiv to find the papers 
    client = arxiv.Client()

    # Search for the most relevant articles matching the queried topic
    search = arxiv.Search(
        query = topic,
        max_results = max_results,
        sort_by = arxiv.SortCriterion.Relevance
    )

    papers = client.results(search)
    
    # Create directory for this topic
    path = os.path.join(PAPER_DIR, topic.lower().replace(" ", "_"))
    os.makedirs(path, exist_ok=True)
    
    file_path = os.path.join(path, "papers_info.json")

    # Try to load existing papers info
    try:
        with open(file_path, "r") as json_file:
            papers_info = json.load(json_file)
    except (FileNotFoundError, json.JSONDecodeError):
        papers_info = {}

    # Process each paper and add to papers_info  
    paper_ids = []
    for paper in papers:
        paper_ids.append(paper.get_short_id())
        paper_info = {
            'title': paper.title,
            'authors': [author.name for author in paper.authors],
            'summary': paper.summary,
            'pdf_url': paper.pdf_url,
            'published': str(paper.published.date())
        }
        papers_info[paper.get_short_id()] = paper_info
    
    # Save updated papers_info to json file
    with open(file_path, "w") as json_file:
        json.dump(papers_info, json_file, indent=2)
    
    print(f"Results are saved in: {file_path}")
    
    return paper_ids

def my_job(topic: str = "machine learning") -> str:
    """
    Wrapper to call the async job from a sync context.
    """

    # asyncio.run(my_job_async())
    asyncio.run(search_papersx(topic, 3))
    return "Job executed."

# ----------------------------
# Public API used by MCP tools
# ----------------------------
def schedule_daily_job(
    hour: int,
    minute: int,
    job_id: str = "daily_my_job",
    replace_existing: bool = True,
    topic: str = "machine learning"
) -> Dict[str, Any]:
    scheduler = get_scheduler()
    trigger = CronTrigger(hour=hour, minute=minute)  # uses scheduler's timezone
    scheduler.add_job(
        my_job,
        trigger=trigger,
        id=job_id,
        replace_existing=replace_existing,
    )
    logging.getLogger("scheduler").info(
        "Scheduled job '%s' daily at %02d:%02d", job_id, hour, minute
    )
    return {
        "ok": True,
        "message": f"Scheduled '{job_id}' daily at {hour:02d}:{minute:02d}.",
        "job_id": job_id,
        "hour": hour,
        "minute": minute,
    }

def remove_job(job_id: str) -> Dict[str, Any]:
    scheduler = get_scheduler()
    scheduler.remove_job(job_id)
    logging.getLogger("scheduler").info("Removed job '%s'", job_id)
    return {"ok": True, "removed": job_id}

def list_jobs() -> List[Dict[str, Any]]:
    scheduler = get_scheduler()
    jobs_info: List[Dict[str, Any]] = []
    for j in scheduler.get_jobs():
        nrt = j.next_run_time.isoformat() if j.next_run_time else None
        jobs_info.append(
            {
                "id": j.id,
                "name": j.name,
                "trigger": str(j.trigger),
                "next_run_time": nrt,
                "coalesce": j.coalesce,
                "max_instances": j.max_instances,
                "misfire_grace_time": j.misfire_grace_time,
            }
        )
    return jobs_info

def shutdown_scheduler() -> str:
    global _scheduler
    if _scheduler and _scheduler.running:
        logging.getLogger("scheduler").info("Shutting down scheduler")
        _scheduler.shutdown(wait=False)
    _scheduler = None
    return "Scheduler shut down."
