@echo off
REM Daily NBA player prop line collection
REM Schedule this in Windows Task Scheduler to run at 10:00 AM ET daily during NBA season
REM
REM Setup:
REM   1. Get API key from https://the-odds-api.com/
REM   2. Set system environment variable ODDS_API_KEY=your_key_here
REM   3. Task Scheduler: Create Basic Task -> Daily -> 10:00 AM -> Start a Program
REM      Program: C:\github\youBet\workflows\nba\betting\scripts\collect_props_daily.bat

set PYTHONPATH=C:\github\youBet\src
cd /d C:\github\youBet\workflows\nba

python betting\scripts\collect_prop_lines.py >> betting\data\props\collection.log 2>&1
