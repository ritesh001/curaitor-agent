import logging
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from mcp.server.fastmcp import FastMCP

from scheduler_service import (
    schedule_daily_job,
    remove_job,
    list_jobs,
    get_scheduler,  # optional: to ensure scheduler is initialized on demand
)

mcp = FastMCP("scheduler")

# Optional: ensure base logging if your server doesn't configure it elsewhere
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

@mcp.tool()
async def add_daily_job(hour: int, minute: int, job_id: str = "daily_my_job"):
    """
    Start/ensure the scheduler is running and add a daily job at HH:MM.
    Non-blocking: returns immediately.
    """

    return schedule_daily_job(hour, minute, job_id=job_id)

@mcp.tool()
async def delete_job(job_id: str):
    """Remove a scheduled job by its ID."""
    return remove_job(job_id)

@mcp.tool()
async def jobs():
    """List scheduled jobs."""
    return {"jobs": list_jobs()}

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')