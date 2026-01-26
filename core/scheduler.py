"""
Task Scheduler - Cron-Like Task Scheduling
===========================================

Schedules and executes tasks at specific times or intervals:
- Trading schedule (auto start/stop at specific hours)
- Daily backtest
- Weekly reports
- Custom cron jobs

Author: SuperBot Team
Date: 2025-11-07
"""

import asyncio
from typing import Dict, Any, Callable, Optional
from datetime import datetime, time as dt_time
import pytz


class TaskScheduler:
    """
    Task Scheduler - Cron-like scheduling

    Supports:
    - Time-based scheduling (HH:MM format)
    - Cron expressions (simple subset)
    - Day-of-week filtering
    - Timezone-aware scheduling
    """

    def __init__(self, logger, config: Dict, daemon=None):
        """
        Initialize Task Scheduler

        Args:
            logger: Logger instance
            config: Scheduler configuration
            daemon: Reference to daemon (for task execution)
        """
        self.logger = logger
        self.config = config
        self.daemon = daemon

        # Timezone
        self.timezone = pytz.timezone(config.get('timezone', 'UTC'))

        # Scheduled tasks
        self.tasks: Dict[str, Dict] = {}

        # State
        self.running = False
        self.task: Optional[asyncio.Task] = None

        # Load tasks from config
        self._load_tasks_from_config()

    def _load_tasks_from_config(self):
        """Load scheduled tasks from configuration"""
        # Trading schedule
        trading_start = self.config.get('trading_start')
        trading_stop = self.config.get('trading_stop')
        trading_days = self.config.get('trading_days', [1, 2, 3, 4, 5])  # Mon-Fri

        if trading_start:
            self.add_task(
                name='trading_start',
                time_str=trading_start,
                func=self._start_trading,
                days=trading_days
            )

        if trading_stop:
            self.add_task(
                name='trading_stop',
                time_str=trading_stop,
                func=self._stop_trading,
                days=trading_days
            )

        # Daily backtest
        daily_backtest = self.config.get('daily_backtest', {})
        if daily_backtest.get('enabled'):
            cron = daily_backtest.get('cron', '0 2 * * *')  # 2 AM daily
            strategy = daily_backtest.get('strategy')

            self.add_cron_task(
                name='daily_backtest',
                cron=cron,
                func=self._run_daily_backtest,
                params={'strategy': strategy}
            )

        # Weekly report
        weekly_report = self.config.get('weekly_report', {})
        if weekly_report.get('enabled'):
            cron = weekly_report.get('cron', '0 0 * * 0')  # Sunday midnight

            self.add_cron_task(
                name='weekly_report',
                cron=cron,
                func=self._generate_weekly_report
            )

        self.logger.info(f"Loaded {len(self.tasks)} scheduled tasks")

    def add_task(
        self,
        name: str,
        time_str: str,
        func: Callable,
        days: Optional[list] = None,
        params: Optional[Dict] = None
    ):
        """
        Add time-based task (HH:MM format)

        Args:
            name: Task name
            time_str: Time in HH:MM format (e.g., "09:00")
            func: Function to execute
            days: List of days (1=Mon, 7=Sun), None = all days
            params: Function parameters
        """
        # Parse time
        try:
            hour, minute = map(int, time_str.split(':'))
            scheduled_time = dt_time(hour=hour, minute=minute)
        except ValueError:
            self.logger.error(f"Invalid time format: {time_str}")
            return

        self.tasks[name] = {
            'type': 'time',
            'time': scheduled_time,
            'func': func,
            'days': days,
            'params': params or {},
            'last_run': None
        }

        self.logger.info(f"Added task '{name}': runs at {time_str} on days {days}")

    def add_cron_task(
        self,
        name: str,
        cron: str,
        func: Callable,
        params: Optional[Dict] = None
    ):
        """
        Add cron-based task

        Args:
            name: Task name
            cron: Cron expression (simplified: "minute hour day month weekday")
            func: Function to execute
            params: Function parameters
        """
        # Parse cron (simplified parser)
        try:
            cron_parts = self._parse_cron(cron)
        except Exception as e:
            self.logger.error(f"Invalid cron expression '{cron}': {e}")
            return

        self.tasks[name] = {
            'type': 'cron',
            'cron': cron_parts,
            'func': func,
            'params': params or {},
            'last_run': None
        }

        self.logger.info(f"Added cron task '{name}': {cron}")

    def _parse_cron(self, cron: str) -> Dict:
        """
        Parse cron expression (simplified)

        Format: "minute hour day month weekday"
        Examples:
            "0 2 * * *"    -> Every day at 2:00 AM
            "0 0 * * 0"    -> Every Sunday at midnight
            "*/15 * * * *" -> Every 15 minutes

        Returns:
            Dict with parsed cron fields
        """
        parts = cron.split()

        if len(parts) != 5:
            raise ValueError("Cron must have 5 fields: minute hour day month weekday")

        minute, hour, day, month, weekday = parts

        return {
            'minute': self._parse_cron_field(minute, 0, 59),
            'hour': self._parse_cron_field(hour, 0, 23),
            'day': self._parse_cron_field(day, 1, 31),
            'month': self._parse_cron_field(month, 1, 12),
            'weekday': self._parse_cron_field(weekday, 0, 6)  # 0=Sun, 6=Sat
        }

    def _parse_cron_field(self, field: str, min_val: int, max_val: int):
        """Parse single cron field"""
        if field == '*':
            return None  # Any value

        if field.isdigit():
            return int(field)

        # TODO: Support ranges (1-5), lists (1,3,5), steps (*/15)
        # For now, just support * and single values

        raise ValueError(f"Unsupported cron field: {field}")

    async def start(self):
        """Start scheduler"""
        self.running = True
        self.task = asyncio.create_task(self._scheduler_loop())

        self.logger.info("Task Scheduler started")

    async def stop(self):
        """Stop scheduler"""
        if not self.running:
            return

        self.running = False

        if self.task:
            self.task.cancel()
            try:
                await self.task
            except asyncio.CancelledError:
                pass

        self.logger.info("Task Scheduler stopped")

    async def _scheduler_loop(self):
        """Main scheduler loop"""
        self.logger.info("Scheduler loop started")

        try:
            while self.running:
                await self._check_tasks()
                await asyncio.sleep(30)  # Check every 30 seconds

        except asyncio.CancelledError:
            self.logger.info("Scheduler loop cancelled")
        except Exception as e:
            self.logger.error(f"Error in scheduler loop: {e}", exc_info=True)

    async def _check_tasks(self):
        """Check if any tasks should run"""
        now = datetime.now(self.timezone)

        for task_name, task_info in self.tasks.items():
            if task_info['type'] == 'time':
                if self._should_run_time_task(now, task_info):
                    await self._execute_task(task_name, task_info)

            elif task_info['type'] == 'cron':
                if self._should_run_cron_task(now, task_info):
                    await self._execute_task(task_name, task_info)

    def _should_run_time_task(self, now: datetime, task_info: Dict) -> bool:
        """Check if time-based task should run"""
        scheduled_time = task_info['time']
        days = task_info['days']

        # Check day of week (1=Mon, 7=Sun)
        if days is not None:
            current_day = now.isoweekday()
            if current_day not in days:
                return False

        # Check time
        current_time = now.time()

        # Check if we're within the minute window
        if (current_time.hour == scheduled_time.hour and
            current_time.minute == scheduled_time.minute):

            # Check if already ran in last 2 minutes (prevent duplicate runs)
            last_run = task_info['last_run']
            if last_run:
                time_since_run = (now - last_run).total_seconds()
                if time_since_run < 120:  # 2 minutes
                    return False

            return True

        return False

    def _should_run_cron_task(self, now: datetime, task_info: Dict) -> bool:
        """Check if cron task should run"""
        cron = task_info['cron']

        # Check each field
        if cron['minute'] is not None and now.minute != cron['minute']:
            return False

        if cron['hour'] is not None and now.hour != cron['hour']:
            return False

        if cron['day'] is not None and now.day != cron['day']:
            return False

        if cron['month'] is not None and now.month != cron['month']:
            return False

        if cron['weekday'] is not None:
            # Convert: Python weekday (0=Mon) to cron weekday (0=Sun)
            current_weekday = (now.weekday() + 1) % 7
            if current_weekday != cron['weekday']:
                return False

        # Check if already ran
        last_run = task_info['last_run']
        if last_run:
            time_since_run = (now - last_run).total_seconds()
            if time_since_run < 60:  # 1 minute
                return False

        return True

    async def _execute_task(self, task_name: str, task_info: Dict):
        """Execute scheduled task"""
        self.logger.info(f"Executing scheduled task: {task_name}")

        try:
            func = task_info['func']
            params = task_info['params']

            # Execute task
            if asyncio.iscoroutinefunction(func):
                await func(**params)
            else:
                func(**params)

            # Update last run
            task_info['last_run'] = datetime.now(self.timezone)

            self.logger.info(f"Task '{task_name}' executed successfully")

        except Exception as e:
            self.logger.error(f"Error executing task '{task_name}': {e}", exc_info=True)

    # ============================================================
    # SCHEDULED TASK HANDLERS
    # ============================================================

    async def _start_trading(self, **params):
        """Start trading module"""
        self.logger.info("Scheduled task: Starting trading")

        if not self.daemon:
            self.logger.error("Daemon not available")
            return

        await self.daemon.start_module('trading', params)

    async def _stop_trading(self, **params):
        """Stop trading module"""
        self.logger.info("Scheduled task: Stopping trading")

        if not self.daemon:
            self.logger.error("Daemon not available")
            return

        await self.daemon.stop_module('trading')

    async def _run_daily_backtest(self, strategy: str = None, **params):
        """Run daily backtest"""
        self.logger.info(f"Scheduled task: Running daily backtest for {strategy}")

        if not self.daemon:
            self.logger.error("Daemon not available")
            return

        # Start backtest module with strategy
        await self.daemon.start_module('backtest', {'strategy': strategy})

    async def _generate_weekly_report(self, **params):
        """Generate weekly report"""
        self.logger.info("Scheduled task: Generating weekly report")

        # TODO: Implement report generation
        # This would collect data from database and generate report

        pass
