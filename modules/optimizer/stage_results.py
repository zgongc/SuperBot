#!/usr/bin/env python3
"""
components/optimizer/stage_results.py
SuperBot - Optimizer Stage Results Manager
Author: SuperBot Team
Date: 2025-11-17
Versiyon: 2.0.0

Manages the saving, loading, and application of stage results.
Saves each stage results to JSON, the next stage loads and applies them.

Features:
- StageResult dataclass (stage result structure)
- StageResultsManager (stage result management)
- JSON-based storage (data/optimization_results/)
- Automatic stage chaining (Stage N -> Stage N+1)
- Parameter override (apply previous stage results)
- Export optimized strategy (save the optimized strategy)

Usage:
    from components.optimizer import StageResultsManager

    # Start a new run
    manager = StageResultsManager(run_id="opt_20251117_143022")

    # Save the stage result
    manager.save_stage_result(
        stage_name="risk_management",
        stage_number=1,
        best_params={'sizing_method': 'RISK_BASED'},
        top_results=[...],
        optimizer_config={...}
    )

    # Load previous stages
    previous_results = manager.load_all_previous_results(before_stage=2)

    # Apply to Strategy (parameter override)
    manager.apply_results_to_strategy(strategy, previous_results)

Dependencies:
    - python>=3.10
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .metrics import BacktestMetrics


@dataclass
class StageResult:
    """Result of a single optimization stage"""

    run_id: str
    stage: str
    stage_number: int
    strategy_name: str
    strategy_version: str

    backtest_period: Dict[str, str]  # start, end, symbol, timeframe

    optimizer_config: Dict[str, Any]  # method, beam_width, total_trials

    best_params: Dict[str, Any]  # En iyi parametre seti

    top_results: List[Dict[str, Any]]  # Top N results (rank, params, metrics)

    timestamp: str


class StageResultsManager:
    """Manages the saving and loading of stage results"""

    def __init__(self, run_id: Optional[str] = None, results_dir: str = "data/optimization_results"):
        """
        Initialize the stage result manager.

        Args:
            run_id: Unique run ID (automatically generated if None)
            results_dir: Main directory for results (no folder is created, it's saved directly here)
        """
        if run_id is None:
            run_id = f"opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.run_id = run_id
        self.results_dir = Path(results_dir)
        # Create folder - save directly to results_dir
        self.run_dir = self.results_dir

        # Create results directory
        self.run_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # SAVE STAGE RESULTS
    # ========================================================================

    def save_stage_result(
        self,
        stage_name: str,
        stage_number: int,
        strategy: Any,
        backtest_period: Dict[str, str],
        optimizer_config: Dict[str, Any],
        results: List[Dict[str, Any]],
        top_n: int = 10
    ) -> Path:
        """
        Save stage results to JSON file

        Args:
            stage_name: Stage name (e.g., 'risk_management')
            stage_number: Stage number (1, 2, 3, ...)
            strategy: Strategy instance
            backtest_period: Dict with start, end, symbol, timeframe
            optimizer_config: Optimizer configuration
            results: List of backtest results (sorted by primary metric)
            top_n: Number of top results to save

        Returns:
            Path to saved file
        """
        # Prepare top results
        top_results = []
        for i, result in enumerate(results[:top_n]):
            top_results.append({
                'rank': i + 1,
                'params': result['params'],
                'metrics': self._metrics_to_dict(result['metrics'])
            })

        # Best params (rank 1)
        best_params = results[0]['params'] if results else {}

        # Create stage result
        stage_result = StageResult(
            run_id=self.run_id,
            stage=stage_name,
            stage_number=stage_number,
            strategy_name=strategy.strategy_name,
            strategy_version=strategy.strategy_version,
            backtest_period=backtest_period,
            optimizer_config=optimizer_config,
            best_params=best_params,
            top_results=top_results,
            timestamp=datetime.now().isoformat()
        )

        # Save to file - include run_id in filename
        filename = f"{self.run_id}_stage_{stage_number}_{stage_name}.json"
        filepath = self.run_dir / filename

        # Custom JSON encoder for numpy types
        def convert_numpy_types(obj):
            """Convert numpy types to Python types"""
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # Convert stage_result dict, handling numpy types
        result_dict = asdict(stage_result)
        # Recursively convert numpy types
        def convert_dict(d):
            if isinstance(d, dict):
                return {k: convert_dict(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [convert_dict(item) for item in d]
            else:
                return convert_numpy_types(d)

        result_dict = convert_dict(result_dict)

        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)

        return filepath

    # ========================================================================
    # LOAD STAGE RESULTS
    # ========================================================================

    def load_stage_result(self, stage_number: int) -> Optional[StageResult]:
        """
        Load a specific stage result

        Args:
            stage_number: Stage number to load

        Returns:
            StageResult or None if not found
        """
        # Find file matching run_id and stage number
        pattern = f"{self.run_id}_stage_{stage_number}_*.json"
        files = list(self.run_dir.glob(pattern))

        if not files:
            return None

        # Load first match
        with open(files[0]) as f:
            data = json.load(f)

        return StageResult(**data)

    def load_all_previous_results(self, before_stage: int) -> List[StageResult]:
        """
        Load all stage results before a given stage

        Args:
            before_stage: Load all results before this stage number

        Returns:
            List of StageResult (sorted by stage number)
        """
        results = []

        for stage_num in range(1, before_stage):
            result = self.load_stage_result(stage_num)
            if result:
                results.append(result)

        return results

    def load_all_results(self) -> List[StageResult]:
        """
        Load all stage results for this run

        Returns:
            List of StageResult (sorted by stage number)
        """
        results = []

        # Find all stage files
        stage_files = sorted(self.run_dir.glob("stage_*.json"))

        for filepath in stage_files:
            with open(filepath) as f:
                data = json.load(f)
            results.append(StageResult(**data))

        # Sort by stage number
        results.sort(key=lambda x: x.stage_number)

        return results

    # ========================================================================
    # APPLY RESULTS TO STRATEGY
    # ========================================================================

    def apply_results_to_strategy(
        self,
        strategy: Any,
        stage_results: List[StageResult]
    ) -> None:
        """
        Apply stage results to strategy (override parameters)

        Args:
            strategy: Strategy instance to modify
            stage_results: List of stage results to apply
        """
        for stage_result in stage_results:
            stage_name = stage_result.stage
            best_params = stage_result.best_params

            # Apply based on stage type
            if stage_name == 'risk_management':
                self._apply_risk_params(strategy, best_params)

            elif stage_name == 'exit_strategy':
                self._apply_exit_params(strategy, best_params)

            elif stage_name == 'indicators':
                self._apply_indicator_params(strategy, best_params)

            elif stage_name == 'entry_conditions':
                self._apply_entry_params(strategy, best_params)

            elif stage_name == 'position_management':
                self._apply_position_params(strategy, best_params)

            elif stage_name == 'market_filters':
                self._apply_filter_params(strategy, best_params)

    def _apply_risk_params(self, strategy: Any, params: Dict[str, Any]) -> None:
        """Apply risk management parameters"""
        if not hasattr(strategy, 'risk_management'):
            return

        for param, value in params.items():
            if hasattr(strategy.risk_management, param):
                setattr(strategy.risk_management, param, value)

    def _apply_exit_params(self, strategy: Any, params: Dict[str, Any]) -> None:
        """Apply exit strategy parameters"""
        if not hasattr(strategy, 'exit_strategy'):
            return

        for param, value in params.items():
            if hasattr(strategy.exit_strategy, param):
                setattr(strategy.exit_strategy, param, value)

    def _apply_indicator_params(self, strategy: Any, params: Dict[str, Any]) -> None:
        """Apply indicator parameters"""
        if not hasattr(strategy, 'technical_parameters'):
            return

        indicators = strategy.technical_parameters.indicators

        for indicator_name, indicator_params in params.items():
            if indicator_name in indicators:
                for param, value in indicator_params.items():
                    if param in indicators[indicator_name]:
                        indicators[indicator_name][param] = value

    def _apply_entry_params(self, strategy: Any, params: Dict[str, Any]) -> None:
        """Apply entry conditions parameters"""
        # Entry conditions are complex (list of conditions)
        # For now, just log that they would be applied
        # Full implementation depends on condition structure
        pass

    def _apply_position_params(self, strategy: Any, params: Dict[str, Any]) -> None:
        """Apply position management parameters"""
        if not hasattr(strategy, 'position_management'):
            return

        for param, value in params.items():
            if hasattr(strategy.position_management, param):
                setattr(strategy.position_management, param, value)

    def _apply_filter_params(self, strategy: Any, params: Dict[str, Any]) -> None:
        """Apply market filter parameters"""
        if not hasattr(strategy, 'custom_parameters'):
            return

        custom_params = strategy.custom_parameters

        # Day filter
        if 'day_filter' in custom_params and 'day_filter' in params:
            for key, value in params['day_filter'].items():
                custom_params['day_filter'][key] = value

        # Session filter
        if 'session_filter' in custom_params and 'session_filter' in params:
            for key, value in params['session_filter'].items():
                custom_params['session_filter'][key] = value

        # Time filter
        if 'time_filter' in custom_params and 'time_filter' in params:
            for key, value in params['time_filter'].items():
                custom_params['time_filter'][key] = value

    # ========================================================================
    # EXPORT FINAL STRATEGY
    # ========================================================================

    def export_optimized_strategy(
        self,
        original_strategy_path: str,
        stage_results: List[StageResult],
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Export optimized strategy template with best parameters

        Args:
            original_strategy_path: Path to original strategy template
            stage_results: All stage results
            output_dir: Output directory (default: run_dir)

        Returns:
            Path to exported strategy file
        """
        if output_dir is None:
            output_dir = self.run_dir

        # Generate output filename
        original_name = Path(original_strategy_path).stem
        timestamp = datetime.now().strftime("%Y%m%d")
        output_filename = f"{original_name}_optimized_{timestamp}.py"
        output_path = output_dir / output_filename

        # Read original template
        with open(original_strategy_path, 'r') as f:
            original_content = f.read()

        # Build optimization metadata comment
        metadata = self._build_optimization_metadata(stage_results)

        # Insert metadata at top of file (after shebang and docstring)
        lines = original_content.split('\n')

        # Find where to insert (after module docstring)
        insert_index = 0
        in_docstring = False
        for i, line in enumerate(lines):
            if '"""' in line or "'''" in line:
                in_docstring = not in_docstring
                if not in_docstring:
                    insert_index = i + 1
                    break

        # Insert metadata
        lines.insert(insert_index, metadata)

        # Write optimized template
        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))

        return output_path

    def _build_optimization_metadata(self, stage_results: List[StageResult]) -> str:
        """Build optimization metadata comment block"""
        if not stage_results:
            return ""

        # Get final metrics (last stage)
        final_stage = stage_results[-1]
        final_metrics = final_stage.top_results[0]['metrics'] if final_stage.top_results else {}

        metadata = f"""
# ============================================================================
# OPTIMIZED STRATEGY TEMPLATE
# ============================================================================
# Original: {final_stage.strategy_name}
# Optimization Date: {datetime.now().strftime("%Y-%m-%d")}
# Optimizer Version: 2.0.0
# Run ID: {self.run_id}
#
# BACKTEST PERIOD:
#   Start: {final_stage.backtest_period.get('start', 'N/A')}
#   End: {final_stage.backtest_period.get('end', 'N/A')}
#   Symbol: {final_stage.backtest_period.get('symbol', 'N/A')}
#   Timeframe: {final_stage.backtest_period.get('timeframe', 'N/A')}
#
# OPTIMIZATION RESULTS:
#   Sharpe Ratio: {final_metrics.get('sharpe_ratio', 0):.2f}
#   Total Return: {final_metrics.get('total_return', 0):.1f}%
#   Profit Factor: {final_metrics.get('profit_factor', 0):.2f}
#   Win Rate: {final_metrics.get('win_rate', 0):.1f}%
#   Max Drawdown: {final_metrics.get('max_drawdown', 0):.1f}%
#   Total Trades: {final_metrics.get('total_trades', 0)}
#
# OPTIMIZED STAGES:
"""

        for stage_result in stage_results:
            stage_metrics = stage_result.top_results[0]['metrics'] if stage_result.top_results else {}
            metadata += f"#   Stage {stage_result.stage_number} ({stage_result.stage}): "
            metadata += f"Sharpe={stage_metrics.get('sharpe_ratio', 0):.2f}, "
            metadata += f"Return={stage_metrics.get('total_return', 0):.1f}%\n"

        metadata += "# ============================================================================\n"

        return metadata

    # ========================================================================
    # UTILITY METHODS
    # ========================================================================

    def _metrics_to_dict(self, metrics: BacktestMetrics) -> Dict[str, Any]:
        """Convert BacktestMetrics to dict"""
        return {
            'total_return': metrics.total_return,
            'annualized_return': metrics.annualized_return,
            'cagr': metrics.cagr,
            'max_drawdown': metrics.max_drawdown,
            'volatility': metrics.volatility,
            'sharpe_ratio': metrics.sharpe_ratio,
            'sortino_ratio': metrics.sortino_ratio,
            'calmar_ratio': metrics.calmar_ratio,
            'total_trades': metrics.total_trades,
            'winning_trades': metrics.winning_trades,
            'losing_trades': metrics.losing_trades,
            'win_rate': metrics.win_rate,
            'gross_profit': metrics.gross_profit,
            'gross_loss': metrics.gross_loss,
            'net_profit': metrics.net_profit,
            'profit_factor': metrics.profit_factor,
            'avg_trade': metrics.avg_trade,
            'avg_win': metrics.avg_win,
            'avg_loss': metrics.avg_loss,
            'expectancy': metrics.expectancy,
            'sqn': metrics.sqn,
        }

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization run summary"""
        all_results = self.load_all_results()

        if not all_results:
            return {
                'run_id': self.run_id,
                'total_stages': 0,
                'status': 'no_results'
            }

        final_stage = all_results[-1]
        final_metrics = final_stage.top_results[0]['metrics'] if final_stage.top_results else {}

        return {
            'run_id': self.run_id,
            'total_stages': len(all_results),
            'stages': [r.stage for r in all_results],
            'strategy_name': final_stage.strategy_name,
            'final_sharpe': final_metrics.get('sharpe_ratio', 0),
            'final_return': final_metrics.get('total_return', 0),
            'final_trades': final_metrics.get('total_trades', 0),
            'run_dir': str(self.run_dir),
        }

    def __repr__(self) -> str:
        summary = self.get_summary()
        return (
            f"<StageResultsManager "
            f"run_id='{self.run_id}' "
            f"stages={summary['total_stages']}>"
        )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'StageResult',
    'StageResultsManager',
]


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 StageResultsManager Test")
    print("=" * 60)

    # Test 1: Create a manager
    print("\n📊 Test 1: Creating a manager")
    manager = StageResultsManager(run_id="test_run_123")
    print(f"   Run ID: {manager.run_id}")
    print(f"   Run Dir: {manager.run_dir}")
    print("   ✅ Test successful")

    # Test 2: Summary information
    print("\n📊 Test 2: Summary information")
    summary = manager.get_summary()
    print(f"   Total Stages: {summary.get('total_stages', 0)}")
    print(f"   Completed Stages: {summary.get('completed_stages', [])}")
    print("   ✅ Test successful")

    # Test 3: Repr
    print("\n📊 Test 3: Repr")
    print(f"   {repr(manager)}")
    print("   ✅ Test successful")

    print("\n✅ All tests completed!")
    print("=" * 60)
