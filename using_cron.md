Where to change it

  - Cron (Linux/macOS)
      - Edit your crontab: crontab -e
      - The first 5 fields control schedule: minute hour day month day-of-week
      - Example entries (replace paths):
          - Daily at 07:00:
              - 0 7 * * * cd /abs/path/curaitor-agent && /opt/homebrew/bin/uv run python scripts/run_daily.py --query "plastic recycling" --max-days 7 --db data/curaitor.sqlite >> logs/langraph_daily.log 2>&1
          - Every 30 minutes:
              - */30 * * * * cd /abs/path/curaitor-agent && /opt/homebrew/bin/uv run python scripts/run_daily.py --query "plastic recycling" --max-days 1 --db data/curaitor.sqlite >> logs/langraph_halfhour.log
  2>&1
          - Every 6 hours:
              - 0 */6 * * * cd /abs/path/curaitor-agent && /opt/homebrew/bin/uv run python scripts/run_daily.py --query "plastic recycling" --max-days 1 --db data/curaitor.sqlite >> logs/langraph_6h.log 2>&1
          - Weekdays 06:15, 12:15, 18:15:
              - 15 6,12,18 * * 1-5 cd /abs/path/curaitor-agent && /opt/homebrew/bin/uv run python scripts/run_daily.py --query "plastic recycling" --max-days 1 --db data/curaitor.sqlite >> logs/langraph_wd.log
  2>&1
      - Tips:
          - Use absolute paths for repo and uv.
          - Set PATH in crontab header if needed: PATH=/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin
          - Cron uses system timezone; set TZ=Europe/London in crontab if you need a specific zone.
  - launchd (macOS)
      - Edit the plist you loaded (copy of scripts/launchd/curaitor.langgraph.sample.plist in ~/Library/LaunchAgents/).
      - Change timing under StartCalendarInterval:
          - One time per day:
              - <key>Hour</key><integer>7</integer><key>Minute</key><integer>0</integer>
          - Multiple times: make it an array of dicts, e.g. three runs/day
              - <array><dict><key>Hour</key><integer>6</integer><key>Minute</key><integer>15</integer></dict> ... </array>
          - Every N seconds: use <key>StartInterval</key><integer>3600</integer> (hourly).
      - After editing:
          - launchctl unload ~/Library/LaunchAgents/com.curaitor.langgraph.daily.plist
          - launchctl load ~/Library/LaunchAgents/com.curaitor.langgraph.daily.plist
          - Or launchctl kickstart -k gui/$UID/com.curaitor.langgraph.daily

  What’s in code vs scheduler

  - Code (scripts/run_daily.py) only defines what to run, not when. Change frequency/timing in crontab/plist.
  - You can pass different flags in your cron/plist command (--max-days, --no-embeddings, --db, etc.) without touching code.

  If you want an in-code scheduler instead (single process), I can add a tiny loop using schedule or APScheduler.

▌how do I stop the cron job scheduled?

> Here are the quick ways to stop it.

  Cron (user crontab)

  - List current entries: crontab -l
  - Edit and remove or comment the job: crontab -e
      - Put a # in front of the line or delete it.
  - Remove all your cron entries (dangerous): crontab -r
      - Only do this if you intend to clear your entire crontab.
  - Stop a currently running job (optional):
      - Find it: ps aux | rg 'scripts/run_daily.py|curaitor_agent.langraph_pipeline'
      - Kill it: kill <PID> or pkill -f scripts/run_daily.py

  System cron files (if you used them)

  - Check /etc/crontab or files in /etc/cron.d/ and comment/remove the line (may require sudo).
  - Restart cron if needed (varies by OS; often not required for simple edits).

  launchd (macOS)

  - Unload and stop the LaunchAgent:
      - launchctl unload ~/Library/LaunchAgents/com.curaitor.langgraph.daily.plist
      - Optionally remove the plist: rm ~/Library/LaunchAgents/com.curaitor.langgraph.daily.plist
  - Verify it’s gone: launchctl list | rg curaitor

  Tips

  - Keep the job line commented in crontab if you might re-enable later.
  - Clean up logs if you were redirecting output (e.g., logs/langraph_daily.log).