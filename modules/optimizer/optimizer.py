#!/usr/bin/env python3
"""
components/optimizer/optimizer.py
SuperBot - Optimizer Ana Class
Author: SuperBot Team
Date: 2025-11-17
Versiyon: 2.0.0

Stage-by-stage optimizasyon orchestrator.
Parallel backtest execution, automatic stage chaining, resume capability.

Features:
- Optimizer class (ana orchestrator)
- Parameter generation (grid, random, beam search)
- Async backtest execution (BacktestEngine)
- Auto stage chaining (Stage N best params → Stage N+1)
- Resume capability (interrupted optimization)
- Progress tracking (real-time status)
- Multi-metric ranking (weighted score)

Usage:
    from components.optimizer.v2 import Optimizer

    # 1. Create optimizer
    optimizer = Optimizer(
        strategy=strategy,
        backtest_config={
            'symbol': 'BTCUSDT',
            'timeframe': '30m',
            'start_date': '2024-01-01',
            'end_date': '2025-01-01',
        }
    )

    # 2. Tek stage optimize et
    results = await optimizer.optimize_stage(
        stage_name='risk_management',
        stage_number=1,
        method='grid',  # or 'random', 'beam'
        max_trials=100
    )

    # 3. Optimize all stages in order (auto)
    final_results = await optimizer.optimize_all_stages(
        stages=['risk_management', 'exit_strategy', 'indicators'],
        max_trials_per_stage=[100, 500, 200]
    )

Dependencies:
    - python>=3.10
    - asyncio
    - BacktestEngine
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import itertools
import random
from dataclasses import dataclass
import time

try:
    from tqdm.asyncio import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

from core.logger_engine import get_logger
from modules.backtest.backtest_engine import BacktestEngine
from .metrics import MetricsCalculator, BacktestMetrics
from .stage_results import StageResultsManager, StageResult
from components.strategies.base_strategy import (
    ExitMethod,
    StopLossMethod,
    PositionSizeMethod,
    TradingSide
)

logger = get_logger("components.optimizer.optimizer")


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

    # Parallel execution
    max_parallel_backtests: int = 10  # How many backtests to run concurrently

    # Primary metric
    primary_metric: str = 'sharpe_ratio'  # Optimize edilecek ana metrik

    # Weighted score (primary_metric='weighted_score' ise)
    metric_weights: Dict[str, float] = None

    # Minimum thresholds (filter out bad results)
    min_thresholds: Dict[str, float] = None

    # Beam search
    beam_width: int = 10  # How many best parameter sets to keep in each stage

    # Resume
    resume_from: Optional[str] = None  # run_id to resume

    # Risk-free rate (for Sharpe/Sortino)
    risk_free_rate: float = 0.02


class Optimizer:
    """
    Stage-by-stage optimizer orchestrator

    Asynchronous parallel backtesting execution with BacktestEngine,
    automatic stage chaining, resume capability.
    """

    def __init__(
        self,
        strategy: Any,
        backtest_config: Dict[str, Any],
        optimizer_config: Optional[OptimizerConfig] = None,
        results_dir: str = "data/optimization_results"
    ):
        """
        Initialize the optimizer.

        Args:
            strategy: The strategy instance to be optimized.
            backtest_config: Backtest settings (symbol, timeframe, dates).
            optimizer_config: Optimizer settings.
            results_dir: The directory where the results will be saved.
        """
        self.strategy = strategy
        self.backtest_config = backtest_config
        self.config = optimizer_config or OptimizerConfig()

        # Managers
        self.stage_manager = StageResultsManager(
            run_id=self.config.resume_from,
            results_dir=results_dir
        )
        self.metrics_calculator = MetricsCalculator(
            risk_free_rate=self.config.risk_free_rate
        )

        # Stats
        self.total_backtests_run = 0
        self.start_time = None

    # ========================================================================
    # STAGE OPTIMIZATION
    # ========================================================================

    async def optimize_stage(
        self,
        stage_name: str,
        stage_number: int,
        method: str = 'grid',
        max_trials: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Tek bir stage'i optimize et

        Args:
            stage_name: Stage name (e.g., 'risk_management')
            stage_number: Stage number (1, 2, 3, ...)
            method: Optimization method ('grid', 'random', 'beam')
            max_trials: Maximum number of backtests

        Returns:
            Sorted list of results (best -> worst)
        """
        print(f"\n{'='*60}")
        print(f"🚀 Stage {stage_number}: {stage_name}")
        print(f"{'='*60}")

        self.start_time = datetime.now()

        # 1. Load previous stage results
        previous_results = self.stage_manager.load_all_previous_results(
            before_stage=stage_number
        )

        if previous_results:
            print(f"📂 {len(previous_results)} previous stage result loaded")
            # Apply to strategy
            self.stage_manager.apply_results_to_strategy(
                self.strategy,
                previous_results
            )

        # 2. Create parameter combinations for this stage.
        param_combinations = self._generate_param_combinations(
            stage_name=stage_name,
            method=method,
            max_trials=max_trials
        )

        print(f"🔧 {len(param_combinations)} parameter combination created")

        # Debug mode: Show the first 5 combinations
        if self.backtest_config.get('verbose', False) and param_combinations:
            print(f"\n🔍 Debug: First {min(5, len(param_combinations))} parameter combinations:")
            for i, combo in enumerate(param_combinations[:5], 1):
                # Convert NumPy types to Python types
                clean_combo = {k: self._clean_value(v) for k, v in combo.items()}
                print(f"   {i}. {clean_combo}")
            print()

        # 3. Parallel backtest execution
        results = await self._run_backtests_parallel(
            param_combinations=param_combinations,
            stage_name=stage_name
        )

        # 4. Sort the results (best -> worst)
        sorted_results = self._rank_results(results)

        # Save the results of the 5th stage
        self.stage_manager.save_stage_result(
            stage_name=stage_name,
            stage_number=stage_number,
            strategy=self.strategy,
            backtest_period=self.backtest_config,
            optimizer_config={
                'method': method,
                'max_trials': max_trials,
                'beam_width': self.config.beam_width,
            },
            results=sorted_results,
            top_n=self.config.beam_width
        )

        # 6. Summary information
        stage_duration = (datetime.now() - self.start_time).total_seconds()
        self._print_stage_summary(sorted_results, stage_name, stage_duration)

        return sorted_results

    # ========================================================================
    # PARAMETER GENERATION
    # ========================================================================

    def _clean_value(self, value: Any) -> Any:
        """
        Convert Numpy types to Python types (for pretty print).

        Args:
            value: Any value (numpy or Python type)

        Returns:
            Python native type
        """
        import numpy as np
        if isinstance(value, np.integer):
            return int(value)
        elif isinstance(value, np.floating):
            return float(value)
        elif isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _expand_param_ranges(self, stage_params: Dict[str, Any]) -> Dict[str, List[Any]]:
        """
        Parametre range'lerini expand et (tuple → list)

        Nested dict support for indicators stage:
            {'donchian_20': {'period': (10, 30, 5)}}
            → {'donchian_20.period': [10, 15, 20, 25, 30]}

        Args:
            stage_params: Parameter dictionary (may contain tuples, lists, or nested dicts)

        Returns:
            Expanded parameter dictionary (only a list).
        """
        import numpy as np

        expanded = {}
        for key, value in stage_params.items():
            if isinstance(value, dict):
                # Nested dict (e.g., indicator params): flatten with dot notation
                for sub_key, sub_value in value.items():
                    flat_key = f"{key}.{sub_key}"
                    if isinstance(sub_value, tuple) and len(sub_value) == 3:
                        min_val, max_val, step = sub_value
                        if isinstance(min_val, (int, float)):
                            expanded[flat_key] = list(np.arange(min_val, max_val + step, step))
                        else:
                            expanded[flat_key] = list(sub_value)
                    elif isinstance(sub_value, list):
                        expanded[flat_key] = sub_value
                    else:
                        expanded[flat_key] = [sub_value]
            elif isinstance(value, tuple) and len(value) == 3:
                # (min, max, step) → [min, min+step, ..., max]
                min_val, max_val, step = value
                if isinstance(min_val, (int, float)):
                    # Numeric range
                    expanded[key] = list(np.arange(min_val, max_val + step, step))
                else:
                    # Tuple but not numeric, convert to list
                    expanded[key] = list(value)
            elif isinstance(value, list):
                # Already a list
                expanded[key] = value
            else:
                # Single value, convert to list
                expanded[key] = [value]

        return expanded

    def _get_default_stage_params(self, stage_name: str) -> Dict[str, List[Any]]:
        """
        Returns the default stage parameters.

        Args:
            stage_name: Stage name

        Returns:
            Default parameter ranges.
        """
        default_params = {
            'risk_management': {
                'max_risk_per_trade': [0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
                'max_portfolio_risk': [5.0, 10.0, 15.0, 20.0],
                'max_correlation': [0.5, 0.6, 0.7, 0.8],
                'max_drawdown_limit': [10.0, 15.0, 20.0, 25.0],
            },
            'exit_strategy': {
                'tp_multiplier': [1.0, 1.5, 2.0, 2.5, 3.0],
                'sl_multiplier': [0.8, 1.0, 1.2, 1.5],
                'trailing_stop': [True, False],
                'trailing_distance': [0.5, 1.0, 1.5, 2.0],
            },
            'indicators': {
                'rsi_period': [7, 14, 21, 28],
                'rsi_overbought': [65, 70, 75, 80],
                'rsi_oversold': [20, 25, 30, 35],
                'ema_fast': [5, 8, 12, 21],
                'ema_slow': [21, 34, 50, 89],
            },
            'entry_conditions': {
                'min_volume_ratio': [1.0, 1.5, 2.0, 2.5],
                'min_trend_strength': [0.5, 0.6, 0.7, 0.8],
                'confirmation_candles': [1, 2, 3],
            },
            'position_management': {
                'sizing_method': ['RISK_BASED', 'FIXED', 'KELLY'],
                'max_position_size': [0.1, 0.2, 0.3, 0.5],
                'scale_in_levels': [1, 2, 3],
            },
            'market_filters': {
                'min_volatility': [0.01, 0.02, 0.03, 0.05],
                'max_volatility': [0.1, 0.15, 0.2, 0.25],
                'trend_filter': [True, False],
            }
        }

        if stage_name not in default_params:
            logger.warning(f"⚠️ Default parameters are not defined for stage '{stage_name}', returning an empty dict")
            return {}

        return default_params[stage_name]

    def _generate_param_combinations(
        self,
        stage_name: str,
        method: str,
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """
        Create parameter combinations.

        Args:
            stage_name: Stage name
            method: 'grid', 'random', 'beam'
            max_trials: Maximum number of combinations.

        Returns:
            List of parameter combinations.
        """
        # Strategy'den optimizer_parameters al
        optimizer_params = getattr(self.strategy, 'optimizer_parameters', {})

        if stage_name not in optimizer_params:
            # If optimizer_parameters is not provided, use the default parameter ranges.
            logger.warning(f"⚠️ Stage '{stage_name}' optimizer_parameters' could not be found, default parameters will be used")
            stage_params = self._get_default_stage_params(stage_name)
        else:
            stage_params = optimizer_params[stage_name]

        # Remove the 'enabled' parameter (for the strategy, not for the optimizer)
        stage_params = {k: v for k, v in stage_params.items() if k != 'enabled'}

        # Tuple range'leri expand et: (min, max, step) → [val1, val2, ...]
        stage_params = self._expand_param_ranges(stage_params)

        if method == 'grid':
            return self._grid_search(stage_params, max_trials)
        elif method == 'random':
            return self._random_search(stage_params, max_trials)
        elif method == 'beam':
            # Beam search: Use the top N results from the previous stage.
            return self._beam_search(stage_params, max_trials)
        else:
            raise ValueError(f"Bilinmeyen method: {method}")

    def _grid_search(
        self,
        stage_params: Dict[str, List[Any]],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """
        Grid search: All combinations

        Specifically for risk management: conditional parameters according to the sizing_method.

        Args:
            stage_params: {'param1': [val1, val2], 'param2': [val3, val4]}
            max_trials: Maximum number of combinations.

        Returns:
            Parameter combinations
        """
        # Special handling for risk management: if sizing_method exists, use conditional logic.
        if 'sizing_method' in stage_params:
            return self._grid_search_conditional_sizing(stage_params, max_trials)

        # Normal grid search (for other stages)
        # Separate parameter names and values
        param_names = list(stage_params.keys())
        param_values = []
        for name in param_names:
            val = stage_params[name]
            # If it's not a list, convert it to a list.
            if not isinstance(val, list):
                val = [val]
            param_values.append(val)

        # Create all combinations
        all_combinations = list(itertools.product(*param_values))

        # Take up to max_trials (sample if more)
        if len(all_combinations) > max_trials:
            print(f"⚠️  Grid search: There are {len(all_combinations)} combinations, sampling {max_trials} of them")
            all_combinations = random.sample(all_combinations, max_trials)

        # Convert to dictionary format
        combinations = []
        for combo in all_combinations:
            param_dict = dict(zip(param_names, combo))
            combinations.append(param_dict)

        return combinations

    def _grid_search_conditional_sizing(
        self,
        stage_params: Dict[str, List[Any]],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """
        Conditional grid search for sizing_method

        For each sizing_method, optimize only the relevant parameters:
        - FIXED_USD: position_usd_size
        - RISK_BASED: max_risk_per_trade
        - FIXED_PERCENT: position_percent_size

        Args:
            stage_params: Risk management parameters
            max_trials: Maximum number of combinations

        Returns:
            Conditional parameter combinations.
        """
        sizing_methods = stage_params.get('sizing_method', ['FIXED_PERCENT'])
        if not isinstance(sizing_methods, list):
            sizing_methods = [sizing_methods]

        # Parameter mapping: which method uses which parameters
        method_params_map = {
            'FIXED_USD': ['position_usd_size'],
            'RISK_BASED': ['max_risk_per_trade'],
            'FIXED_PERCENT': ['position_percent_size'],
            'FIXED_QUANTITY': ['position_quantity_size'],
            'KELLY': ['kelly_fraction', 'max_risk_per_trade'],
        }

        all_combinations = []

        # Create separate combinations for each sizing_method
        for method in sizing_methods:
            # This method requires the following parameters
            required_params = method_params_map.get(method, [])

            # Get only the relevant parameters
            relevant_params = {}
            for param_name in required_params:
                if param_name in stage_params:
                    relevant_params[param_name] = stage_params[param_name]

            # Create combinations for this method
            if relevant_params:
                param_names = list(relevant_params.keys())
                param_values = [relevant_params[name] for name in param_names]

                # Create grid
                method_combos = list(itertools.product(*param_values))

                # sizing_method ekle
                for combo in method_combos:
                    param_dict = {'sizing_method': method}
                    param_dict.update(dict(zip(param_names, combo)))
                    all_combinations.append(param_dict)
            # else: SKIP this method if there are no parameters (user does not want to test)

        # Sample up to max_trials times
        if len(all_combinations) > max_trials:
            print(f"⚠️ Grid search: There are {len(all_combinations)} combinations, sampling {max_trials} of them")
            all_combinations = random.sample(all_combinations, max_trials)

        return all_combinations

    def _random_search(
        self,
        stage_params: Dict[str, List[Any]],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """
        Random search: Rastgele kombinasyonlar

        Args:
            stage_params: {'param1': [val1, val2], 'param2': [val3, val4]}
            max_trials: How many combinations

        Returns:
            Parameter combinations
        """
        param_names = list(stage_params.keys())
        param_values = [stage_params[name] for name in param_names]

        combinations = []
        for _ in range(max_trials):
            # Select a random value for each parameter
            random_values = [random.choice(values) for values in param_values]
            param_dict = dict(zip(param_names, random_values))
            combinations.append(param_dict)

        return combinations

    def _beam_search(
        self,
        stage_params: Dict[str, List[Any]],
        max_trials: int
    ) -> List[Dict[str, Any]]:
        """
        Beam search: Expand the top N results from the previous stage.

        TODO: Implement beam search logic
        """
        # For now, let's use grid search.
        return self._grid_search(stage_params, max_trials)

    # ========================================================================
    # BACKTEST EXECUTION
    # ========================================================================

    async def _run_backtests_parallel(
        self,
        param_combinations: List[Dict[str, Any]],
        stage_name: str
    ) -> List[Dict[str, Any]]:
        """
        Parallel backtest execution (async) with progress bar

        Args:
            param_combinations: Parameter combinations
            stage_name: Stage name

        Returns:
            Backtest results (parameters + metrics)
        """
        total = len(param_combinations)
        print(f"\n🔄 {total} backtests are running in parallel...")
        print(f"   Max paralel: {self.config.max_parallel_backtests}")

        # Track the start time
        stage_start_time = time.time()

        # Limit the number of parallel processes with a semaphore.
        semaphore = asyncio.Semaphore(self.config.max_parallel_backtests)

        # For progress tracking
        self.completed_tasks = 0
        self.total_tasks = total
        self.stage_start = stage_start_time

        # Run all backtests
        tasks = []
        for i, params in enumerate(param_combinations):
            task = self._run_single_backtest_with_progress(
                params=params,
                stage_name=stage_name,
                trial_number=i + 1,
                total_trials=total,
                semaphore=semaphore
            )
            tasks.append(task)

        # Wait with a progress bar
        if TQDM_AVAILABLE:
            results = []
            with tqdm(total=total, desc=f"⚙️  {stage_name}",
                     unit="test", ncols=100, colour='green') as pbar:
                for coro in asyncio.as_completed(tasks):
                    result = await coro
                    results.append(result)
                    pbar.update(1)

                    # ETA and speed information is automatically displayed by tqdm.
                    if result:
                        # Update description with success rate
                        success_count = len([r for r in results if r is not None])
                        pbar.set_postfix(success=f"{success_count}/{len(results)}")
        else:
            # tqdm yoksa basit output
            results = await asyncio.gather(*tasks)

        # Filter out non-null results (failed backtests)
        valid_results = [r for r in results if r is not None]

        # Calculate total duration
        total_time = time.time() - stage_start_time
        avg_time = total_time / total if total > 0 else 0

        print(f"\n✅ {len(valid_results)}/{total} backtest successful")
        print(f"⏱️ Total time: {total_time:.1f}s | Average: {avg_time:.2f}s/test")

        return valid_results

    async def _run_single_backtest_with_progress(
        self,
        params: Dict[str, Any],
        stage_name: str,
        trial_number: int,
        total_trials: int,
        semaphore: asyncio.Semaphore
    ) -> Optional[Dict[str, Any]]:
        """
        Run a single backtest (with progress tracking)

        Args:
            params: Parametre seti
            stage_name: Stage name
            trial_number: Trial number
            total_trials: Total number of trials
            semaphore: Parallel count limiter

        Returns:
            {'params': {...}, 'metrics': BacktestMetrics} or None (failed)
        """
        async with semaphore:
            try:
                # Show progress for debug mode (when tqdm is not available)
                if not TQDM_AVAILABLE and self.backtest_config.get('verbose', False):
                    if trial_number % 10 == 0 or trial_number == 1:
                        elapsed = time.time() - self.stage_start
                        eta = (elapsed / trial_number) * (total_trials - trial_number)
                        clean_params = {k: self._clean_value(v) for k, v in params.items()}
                        print(f"   [{trial_number}/{total_trials}] "
                              f"ETA: {int(eta)}s | "
                              f"Params: {clean_params}")

                # Apply parameters to the strategy
                self._apply_params_to_strategy(params, stage_name)

                # Apply the backtest configuration to the Strategy (for CLI override)
                if 'symbol' in self.backtest_config:
                    # Update the symbol list (first symbol override)
                    if hasattr(self.strategy, 'symbols') and self.strategy.symbols:
                        # Parse symbol (BTCUSDT → BTC, USDT)
                        symbol_str = self.backtest_config['symbol']
                        # Simple parsing (assume USDT quote for now)
                        if 'USDT' in symbol_str:
                            base = symbol_str.replace('USDT', '')
                            self.strategy.symbols[0].symbol = [base]
                            self.strategy.symbols[0].quote = 'USDT'

                if 'timeframe' in self.backtest_config:
                    self.strategy.primary_timeframe = self.backtest_config['timeframe']

                if 'start_date' in self.backtest_config:
                    # Format: YYYY-MM-DD → YYYY-MM-DDT00:00
                    start_date = self.backtest_config['start_date']
                    if 'T' not in start_date:
                        start_date = f"{start_date}T00:00"
                    self.strategy.backtest_start_date = start_date

                if 'end_date' in self.backtest_config:
                    # Format: YYYY-MM-DD → YYYY-MM-DDT23:59
                    end_date = self.backtest_config['end_date']
                    if 'T' not in end_date:
                        end_date = f"{end_date}T23:59"
                    self.strategy.backtest_end_date = end_date

                if 'initial_balance' in self.backtest_config:
                    self.strategy.initial_balance = self.backtest_config['initial_balance']

                # Debug mode'u strategy'ye uygula (BacktestEngine bunu okur)
                if 'verbose' in self.backtest_config:
                    self.strategy.debug = self.backtest_config['verbose']

                # Run backtest (async) - All config is read from Strategy.
                # BacktestEngine strategy.debug attribute'unu okur
                engine = BacktestEngine()
                result = await engine.run(strategy=self.strategy, use_cache=True)

                # Trades are generated
                trades = result.trades if hasattr(result, 'trades') else []

                # Metrikleri hesapla
                backtest_days = (
                    datetime.fromisoformat(self.backtest_config['end_date']) -
                    datetime.fromisoformat(self.backtest_config['start_date'])
                ).days

                metrics = self.metrics_calculator.calculate_all_metrics(
                    trades=trades,
                    initial_balance=self.backtest_config.get('initial_balance', 10000),
                    backtest_days=backtest_days
                )

                self.total_backtests_run += 1

                return {
                    'params': params,
                    'metrics': metrics
                }

            except Exception as e:
                print(f"   ❌ Backtest failed: {e}")
                # Debug mode: Full traceback
                if self.backtest_config.get('verbose', False):
                    import traceback
                    traceback.print_exc()
                return None

    def _apply_params_to_strategy(
        self,
        params: Dict[str, Any],
        stage_name: str
    ):
        """
        Apply the parameters to the strategy.

        Args:
            params: Parametre dict'i
            stage_name: Stage name (risk_management, exit_strategy, etc.)
        """
        # Find the relevant strategy attribute based on the stage name.
        if stage_name == 'main_strategy':
            # Main strategy parameters (side_method, leverage, etc.)
            for key, value in params.items():
                # Enum conversion
                if key == 'side_method' and isinstance(value, str):
                    value = TradingSide(value)
                setattr(self.strategy, key, value)

        elif stage_name == 'risk_management':
            for key, value in params.items():
                # Enum conversion
                if key == 'sizing_method' and isinstance(value, str):
                    value = PositionSizeMethod(value)
                setattr(self.strategy.risk_management, key, value)

        elif stage_name == 'exit_strategy':
            for key, value in params.items():
                # Enum conversion (optimizer passes a string, strategy expects an enum)
                if key == 'stop_loss_method' and isinstance(value, str):
                    value = StopLossMethod(value)
                elif key == 'take_profit_method' and isinstance(value, str):
                    value = ExitMethod(value)
                setattr(self.strategy.exit_strategy, key, value)

            # Validation: If ATR_BASED is selected, add the ATR indicator.
            self._validate_and_fix_atr()

        elif stage_name == 'position_management':
            for key, value in params.items():
                setattr(self.strategy.position_management, key, value)

        elif stage_name == 'indicators':
            # Apply indicator parameters to technical_parameters.indicators
            # Supports dot notation from _expand_param_ranges: 'donchian_20.period' → indicators['donchian_20']['period']
            for key, value in params.items():
                if '.' in key:
                    indicator_name, param_name = key.split('.', 1)
                    if indicator_name in self.strategy.technical_parameters.indicators:
                        self.strategy.technical_parameters.indicators[indicator_name][param_name] = value
                else:
                    # Flat key: try indicator_config fallback
                    if hasattr(self.strategy, 'indicator_config'):
                        self.strategy.indicator_config[key] = value

        elif stage_name == 'entry_conditions':
            # Entry condition parameters
            # TODO: Implement entry condition parameter application
            pass

        elif stage_name == 'market_filters':
            # Market filter parameters
            # TODO: Implement market filter parameter application
            pass

    def _validate_and_fix_atr(self):
        """
        If ATR_BASED is selected, add the ATR indicator if it doesn't exist
        (Same logic as StrategyValidator)
        """
        import re

        # Exit strategy check
        sl_method = self.strategy.exit_strategy.stop_loss_method
        tp_method = self.strategy.exit_strategy.take_profit_method

        sl_name = sl_method.name if hasattr(sl_method, 'name') else str(sl_method)
        tp_name = tp_method.name if hasattr(tp_method, 'name') else str(tp_method)

        # Is ATR_BASED selected?
        if sl_name == 'ATR_BASED' or tp_name == 'ATR_BASED':
            # Is there an ATR?
            indicators = self.strategy.technical_parameters.indicators
            atr_pattern = re.compile(r'^atr(_\d+)?$')
            has_atr = any(atr_pattern.match(key) for key in indicators.keys())

            if not has_atr:
                # Automatically add
                indicators['atr_14'] = {'period': 14}
                # Cache invalidate
                self.strategy._cache_invalidated = True
                logger.info(f"💡 ATR_BASED is selected, but ATR is not available -> Automatically added atr_14")

    # ========================================================================
    # RESULT RANKING
    # ========================================================================

    def _rank_results(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort the results based on the metric value (best -> worst).

        Args:
            results: Backtest results

        Returns:
            Sorted results
        """
        # First, filter those that exceed the minimum thresholds.
        filtered_results = self._filter_by_thresholds(results)

        # Sort by primary metric
        if self.config.primary_metric == 'weighted_score':
            # Weighted score hesapla
            for result in filtered_results:
                result['score'] = self._calculate_weighted_score(result['metrics'])

            # Sort by score
            sorted_results = sorted(
                filtered_results,
                key=lambda x: x['score'],
                reverse=True
            )
        else:
            # Sort by single metric
            sorted_results = sorted(
                filtered_results,
                key=lambda x: getattr(x['metrics'], self.config.primary_metric),
                reverse=True  # High = good (sharpe, profit_factor, etc.)
            )

        return sorted_results

    def _filter_by_thresholds(
        self,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Filter results that do not exceed the minimum thresholds.

        Args:
            results: All results

        Returns:
            Filtered results
        """
        if not self.config.min_thresholds:
            return results

        filtered = []
        for result in results:
            metrics = result['metrics']

            # Check all thresholds
            passes = True
            for metric_name, min_value in self.config.min_thresholds.items():
                actual_value = getattr(metrics, metric_name, None)

                if actual_value is None:
                    passes = False
                    break

                # inverse for max_drawdown (lower is better)
                if metric_name == 'max_drawdown':
                    if actual_value > min_value:
                        passes = False
                        break
                else:
                    if actual_value < min_value:
                        passes = False
                        break

            if passes:
                filtered.append(result)

        print(f"   🔍 Threshold filter: {len(filtered)}/{len(results)} results passed")

        return filtered

    def _calculate_weighted_score(
        self,
        metrics: BacktestMetrics
    ) -> float:
        """
        Calculate the weighted score (weighted sum of multiple metrics).

        Args:
            metrics: Backtest metrikleri

        Returns:
            Weighted score
        """
        if not self.config.metric_weights:
            raise ValueError("metric_weights is not defined, but primary_metric='weighted_score'")

        score = 0.0
        for metric_name, weight in self.config.metric_weights.items():
            value = getattr(metrics, metric_name, 0)

            # Negative weight = minimize (e.g., max_drawdown)
            score += weight * value

        return score

    # ========================================================================
    # SUMMARY & REPORTING
    # ========================================================================

    def _print_stage_summary(
        self,
        results: List[Dict[str, Any]],
        stage_name: str,
        stage_duration: float
    ):
        """
        Print stage summary (with timing info)

        Args:
            results: Sorted results
            stage_name: Stage name
            stage_duration: Stage duration (seconds)
        """
        print(f"\n{'='*60}")
        print(f"📊 Stage Summary: {stage_name}")
        print(f"{'='*60}")

        if not results:
            print("❌ No valid results found!")
            return

        # Top 3 results
        print("\n🏆 Top 3 Results:")
        for i, result in enumerate(results[:3]):
            metrics = result['metrics']
            # Clear Numpy types
            clean_params = {k: self._clean_value(v) for k, v in result['params'].items()}
            print(f"\n   #{i+1}")
            print(f"      Params: {clean_params}")
            print(f"      Sharpe: {metrics.sharpe_ratio:.2f}")
            print(f"      Profit Factor: {metrics.profit_factor:.2f}")
            print(f"      Win Rate: {metrics.win_rate:.1f}%")
            print(f"      Total Return: {metrics.total_return:.2f}%")
            print(f"      Max DD: {metrics.max_drawdown:.2f}%")

        # Statistics - advanced time format
        avg_time = stage_duration / len(results) if len(results) > 0 else 0

        # Convert the duration to minutes:seconds format
        minutes = int(stage_duration // 60)
        seconds = int(stage_duration % 60)
        time_str = f"{minutes}m {seconds}s" if minutes > 0 else f"{seconds}s"

        print(f"\n📈 Statistics:")
        print(f"   Total Backtests: {len(results)}")
        print(f"   ⏱️  Total Time: {time_str} ({stage_duration:.1f}s)")
        print(f"   📊 Avg Time/Backtest: {avg_time:.2f}s")
        print(f"   ⚡ Throughput: {len(results)/stage_duration:.2f} tests/sec")

        # Debug mode: Show all results
        if self.backtest_config.get('verbose', False):
            print(f"\n🔍 Debug: All {len(results)} Results (Best -> Worst):")
            for i, result in enumerate(results, 1):
                metrics = result['metrics']
                clean_params = {k: self._clean_value(v) for k, v in result['params'].items()}
                print(f"\n   #{i}")
                print(f"      Params: {clean_params}")
                print(f"      Sharpe: {metrics.sharpe_ratio:.2f}, PF: {metrics.profit_factor:.2f}, "
                      f"WR: {metrics.win_rate:.1f}%, Return: {metrics.total_return:.2f}%, "
                      f"DD: {metrics.max_drawdown:.2f}%")
                print(f"      Trades: {metrics.total_trades}, Win: {metrics.winning_trades}, "
                      f"Loss: {metrics.losing_trades}, Avg: ${metrics.avg_trade:.2f}")


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'Optimizer',
    'OptimizerConfig',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 Optimizer Test")
    print("=" * 60)

    # Test 1: Create OptimizerConfig
    print("\n📊 Test 1: OptimizerConfig")
    config = OptimizerConfig(
        max_parallel_backtests=10,
        primary_metric='sharpe_ratio',
        beam_width=10
    )
    print(f"   Max parallel: {config.max_parallel_backtests}")
    print(f"   Primary metric: {config.primary_metric}")
    print("   ✅ Test successful")

    # Test 2: Grid search parameter generation (mock)
    print("\n📊 Test 2: Grid search parameter generation")
    stage_params = {
        'sizing_method': ['RISK_BASED', 'FIXED'],
        'max_risk_per_trade': [1.0, 2.0, 3.0]
    }

    # Mock optimizer
    class MockStrategy:
        optimizer_parameters = {
            'risk_management': stage_params
        }

    mock_optimizer = Optimizer(
        strategy=MockStrategy(),
        backtest_config={'symbol': 'BTCUSDT', 'timeframe': '30m'}
    )

    combinations = mock_optimizer._grid_search(stage_params, max_trials=10)
    print(f"   Number of combinations: {len(combinations)}")
    print(f"   First 3: {combinations[:3]}")
    print("   ✅ Test successful")

    print("\n✅ All tests completed!")
    print("=" * 60)
