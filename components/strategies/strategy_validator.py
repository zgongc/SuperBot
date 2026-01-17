#!/usr/bin/env python3
"""
Strategy Validator - Strateji Parametrelerini Doğrular

Tüm strateji parametrelerinin (RiskManagement, ExitStrategy, entry/exit conditions, vs.)
doğru formatta ve mantıklı değerlerde olup olmadığını kontrol eder.

Bu dosya strategies/ klasöründe çünkü STRATEJI PARAMETRELERI ile ilgilidir.

Version: 2.0.0 (helpers/validation.py'deki tüm kontrollerle birleştirildi)
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import enums for validation
try:
    from components.strategies.base_strategy import (
        PositionSizeMethod,
        ExitMethod,
        StopLossMethod,
    )
except ImportError:
    # Fallback if imports fail
    PositionSizeMethod = None
    ExitMethod = None
    StopLossMethod = None


@dataclass
class ValidationError:
    """
    Validasyon hatası
    
    Attributes:
        field: Hatalı alan adı
        value: Hatalı değer
        message: Hata mesajı (Türkçe)
        severity: Hata ciddiyeti ('error', 'warning')
    """
    field: str
    value: Any
    message: str
    severity: str = 'error'  # 'error' veya 'warning'


@dataclass
class StrategyValidationResult:
    """
    Strateji validasyon sonucu
    
    Attributes:
        valid: Strateji geçerli mi?
        errors: Hata listesi
        warnings: Uyarı listesi
    """
    valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    
    def __str__(self) -> str:
        """İnsan okunabilir format"""
        lines = []
        if self.valid:
            lines.append("Strateji gecerli")
        else:
            lines.append("Strateji gecersiz")
        
        if self.errors:
            lines.append(f"\nHatalar ({len(self.errors)}):")
            for err in self.errors:
                lines.append(f"  - {err.field}: {err.message}")
        
        if self.warnings:
            lines.append(f"\nUyarilar ({len(self.warnings)}):")
            for warn in self.warnings:
                lines.append(f"  - {warn.field}: {warn.message}")
        
        return "\n".join(lines)


class StrategyValidator:
    """
    Kapsamlı Strateji Parametrelerini Validate Eder
    
    Kontrol Edilen Parametreler:
    - Metadata: strategy_name, version, author
    - Symbols: Symbol configs ve format
    - Timeframes: MTF timeframes ve primary timeframe
    - Indicators: Technical indicators ve parametreleri
    - RiskManagement: Risk yönetimi ayarları (ENUM + değerler)
    - ExitStrategy: Çıkış stratejisi ayarları (ENUM + değerler)
    - PositionManagement: Pozisyon yönetimi ayarları
    - entry_conditions: Giriş koşulları (syntax, operators, timeframes)
    - exit_conditions: Çıkış koşulları (syntax, operators, timeframes)
    - custom_parameters: Özel parametreler
    - optimizer_parameters: Optimizasyon parametreleri
    - account_management: Hesap yönetimi
    - backtest_parameters: Backtest parametreleri
    
    Validation Türleri:
    - ✅ Zorunlu alanlar (required fields)
    - ✅ Değer aralıkları (ranges)
    - ✅ Mantıksal tutarlılık (TP > SL vs.)
    - ✅ Format kontrolü (dict, list, vs.)
    - ✅ Enum validations
    - ✅ Operator validations
    - ✅ Timeframe validations
    - ✅ Symbol format validations
    - ✅ Condition syntax validations
    """
    
    # Required fields (minimum)
    REQUIRED_FIELDS = [
        'strategy_name',
        'strategy_version',
        'symbols',
        'technical_parameters',
        'entry_conditions',
        'exit_strategy',
        'risk_management',
        'position_management',
    ]
    
    # Geçerli timeframe'ler
    VALID_TIMEFRAMES = {
        '1m', '3m', '5m', '15m', '30m',
        '1h', '2h', '4h', '6h', '12h',
        '1d', '3d', '1w', '1M'
    }
    
    # Geçerli operatörler
    VALID_OPERATORS = {
        '>', '<', '>=', '<=', '==', '!=',
        'crossover', 'crossunder', 'cross_over', 'cross_under',
        'crossabove', 'crossbelow',
        'rising', 'falling',
        'between', 'outside', 'near',
        'is', 'is_not',
        'above', 'below',
        'gte', 'lte', 'greater_equal', 'less_equal',
        'equals', 'not_equals',
    }
    
    def __init__(self, logger: Optional[Any] = None):
        """
        Validator'ı başlat
        
        Args:
            logger: Opsiyonel logger
        """
        self.logger = logger
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
    
    def validate_strategy(self, strategy: Any) -> StrategyValidationResult:
        """
        Stratejiyi tamamen validate et (KAPSAMLI)

        Args:
            strategy: Strateji objesi (BaseStrategy türevi)

        Returns:
            StrategyValidationResult
        """
        self.errors = []
        self.warnings = []

        # Store current strategy for indicator checks
        self._current_strategy = strategy
        
        # 0. Required fields
        self._validate_required_fields(strategy)
        
        # 1. Metadata
        self._validate_metadata(strategy)
        
        # 2. Symbols
        if hasattr(strategy, 'symbols'):
            self._validate_symbols(strategy)
        
        # 3. Timeframes
        if hasattr(strategy, 'mtf_timeframes'):
            self._validate_timeframes(strategy)
        
        # 4. Indicators
        if hasattr(strategy, 'technical_parameters'):
            self._validate_indicators(strategy)
        
        # 5. RiskManagement validasyonu
        if hasattr(strategy, 'risk_management'):
            self._validate_risk_management(strategy.risk_management)
        else:
            self.errors.append(ValidationError(
                field='risk_management',
                value=None,
                message='RiskManagement parametresi eksik!'
            ))
        
        # 6. ExitStrategy validasyonu
        if hasattr(strategy, 'exit_strategy'):
            self._validate_exit_strategy(strategy.exit_strategy)
        else:
            self.errors.append(ValidationError(
                field='exit_strategy',
                value=None,
                message='ExitStrategy parametresi eksik!'
            ))
        
        # 7. PositionManagement validasyonu
        if hasattr(strategy, 'position_management'):
            self._validate_position_management(strategy.position_management)
        else:
            self.errors.append(ValidationError(
                field='position_management',
                value=None,
                message='PositionManagement parametresi eksik!'
            ))
        
        # 8. Entry/Exit Conditions validasyonu
        if hasattr(strategy, 'entry_conditions'):
            self._validate_conditions(strategy, 'entry_conditions')
        
        if hasattr(strategy, 'exit_conditions'):
            self._validate_conditions(strategy, 'exit_conditions')
        
        # 9. Custom Parameters validasyonu
        if hasattr(strategy, 'custom_parameters'):
            self._validate_custom_parameters(strategy.custom_parameters)
        
        # 10. Optimizer Parameters validasyonu
        if hasattr(strategy, 'optimizer_parameters'):
            self._validate_optimizer_parameters(strategy.optimizer_parameters)
        
        # 11. Account Management (optional)
        if hasattr(strategy, 'account_management') and strategy.account_management:
            self._validate_account_management(strategy.account_management)
        
        # 12. Backtest Parameters (optional)
        if hasattr(strategy, 'backtest_parameters') and strategy.backtest_parameters:
            self._validate_backtest_parameters(strategy)
        
        # Sonuç oluştur
        result = StrategyValidationResult(
            valid=len(self.errors) == 0,
            errors=self.errors,
            warnings=self.warnings
        )
        
        if self.logger:
            if result.valid:
                self.logger.info(f"Strateji validasyonu basarili")
                if result.warnings:
                    self.logger.warning(f"{len(result.warnings)} uyari var")
            else:
                self.logger.error(f"Strateji validasyonu basarisiz: {len(result.errors)} hata")
        
        return result
    
    def _validate_required_fields(self, strategy: Any):
        """Required field'ların varlığını kontrol et"""
        missing = []
        
        for field in self.REQUIRED_FIELDS:
            if not hasattr(strategy, field):
                missing.append(field)
            elif getattr(strategy, field) is None:
                missing.append(field)
        
        if missing:
            self.errors.append(ValidationError(
                field='required_fields',
                value=missing,
                message=f"Eksik required field'lar: {missing}"
            ))
    
    def _validate_metadata(self, strategy: Any):
        """Metadata doğrula"""
        if hasattr(strategy, 'strategy_name'):
            if not strategy.strategy_name:
                self.errors.append(ValidationError(
                    field='strategy_name',
                    value=strategy.strategy_name,
                    message='strategy_name bos olamaz'
                ))
        
        if hasattr(strategy, 'strategy_version'):
            if not strategy.strategy_version:
                self.errors.append(ValidationError(
                    field='strategy_version',
                    value=strategy.strategy_version,
                    message='strategy_version bos olamaz'
                ))
        
        # Optional: author validation
        if hasattr(strategy, 'author') and strategy.author:
            if len(strategy.author) > 100:
                self.warnings.append(ValidationError(
                    field='author',
                    value=strategy.author,
                    message='author cok uzun (max 100 karakter)',
                    severity='warning'
                ))
    
    def _validate_symbols(self, strategy: Any):
        """Symbol config'leri doğrula"""
        if not strategy.symbols:
            self.errors.append(ValidationError(
                field='symbols',
                value=None,
                message='En az 1 symbol gerekli'
            ))
            return
        
        for i, symbol_config in enumerate(strategy.symbols):
            if not hasattr(symbol_config, 'symbol') or not symbol_config.symbol:
                self.errors.append(ValidationError(
                    field=f'symbols[{i}].symbol',
                    value=None,
                    message='Symbol listesi bos'
                ))
            
            if not hasattr(symbol_config, 'quote') or not symbol_config.quote:
                self.errors.append(ValidationError(
                    field=f'symbols[{i}].quote',
                    value=None,
                    message='Quote bos'
                ))
            
            # Check symbol format
            if hasattr(symbol_config, 'symbol'):
                for sym in symbol_config.symbol:
                    if not sym or not isinstance(sym, str):
                        self.errors.append(ValidationError(
                            field=f'symbols[{i}].symbol',
                            value=sym,
                            message=f'Gecersiz symbol: {sym}'
                        ))
    
    def _validate_timeframes(self, strategy: Any):
        """Timeframe'leri doğrula"""
        # MTF timeframes
        # MTF is optional - allow empty list
        if not strategy.mtf_timeframes:
            return  # Skip MTF validation if not used
        
        for tf in strategy.mtf_timeframes:
            if tf not in self.VALID_TIMEFRAMES:
                self.errors.append(ValidationError(
                    field='mtf_timeframes',
                    value=tf,
                    message=f"Gecersiz timeframe: '{tf}'. Gecerli: {sorted(self.VALID_TIMEFRAMES)}"
                ))
        
        # Primary timeframe mtf_timeframes içinde olmalı
        if hasattr(strategy, 'primary_timeframe'):
            if strategy.primary_timeframe not in strategy.mtf_timeframes:
                self.errors.append(ValidationError(
                    field='primary_timeframe',
                    value=strategy.primary_timeframe,
                    message=f"primary_timeframe ('{strategy.primary_timeframe}') mtf_timeframes icinde olmali"
                ))
    
    def _validate_indicators(self, strategy: Any):
        """Indikatör config'lerini doğrula"""
        indicators = strategy.technical_parameters.indicators

        if not indicators:
            self.errors.append(ValidationError(
                field='technical_parameters.indicators',
                value=None,
                message='En az 1 indikatör gerekli'
            ))
            return

        for name, params in indicators.items():
            if not isinstance(params, dict):
                self.errors.append(ValidationError(
                    field=f'technical_parameters.indicators.{name}',
                    value=type(params).__name__,
                    message=f"Parametreler dict olmali, {type(params).__name__} verildi"
                ))

        # ATR_BASED kontrolü: ATR indicator gerekli mi?
        if hasattr(strategy, 'exit_strategy'):
            sl_method = getattr(strategy.exit_strategy, 'stop_loss_method', None)
            tp_method = getattr(strategy.exit_strategy, 'take_profit_method', None)

            sl_name = sl_method.name if hasattr(sl_method, 'name') else str(sl_method) if sl_method else None
            tp_name = tp_method.name if hasattr(tp_method, 'name') else str(tp_method) if tp_method else None

            # ATR_BASED seçilmiş mi?
            if sl_name == 'ATR_BASED' or tp_name == 'ATR_BASED':
                # ATR var mı?
                import re
                atr_pattern = re.compile(r'^atr(_\d+)?$')
                has_atr = any(atr_pattern.match(key) for key in indicators.keys())

                if not has_atr:
                    # Otomatik ekle
                    indicators['atr_14'] = {'period': 14}
                    # Cache invalidate flag (backtest engine kontrol eder)
                    strategy._cache_invalidated = True
                    self.warnings.append(ValidationError(
                        field='technical_parameters.indicators',
                        value='atr_14',
                        message='ATR_BASED seçili ama ATR yok → Otomatik atr_14 eklendi'
                    ))
    
    def _validate_risk_management(self, rm: Any):
        """RiskManagement parametrelerini validate et (ENUM + değerler)"""
        # Enum validation (sadece enum import edildiyse)
        if PositionSizeMethod is not None and hasattr(rm, 'sizing_method'):
            if not isinstance(rm.sizing_method, PositionSizeMethod):
                self.errors.append(ValidationError(
                    field='risk_management.sizing_method',
                    value=type(rm.sizing_method).__name__,
                    message='sizing_method PositionSizeMethod enum olmali'
                ))
        
        # Validate position sizing parameters based on sizing_method
        if hasattr(rm, 'sizing_method'):
            if rm.sizing_method == PositionSizeMethod.FIXED_PERCENT:
                # FIXED_PERCENT requires position_percent_size
                if not hasattr(rm, 'position_percent_size') or rm.position_percent_size <= 0:
                    self.errors.append(ValidationError(
                        field='risk_management.position_percent_size',
                        value=getattr(rm, 'position_percent_size', None),
                        message='FIXED_PERCENT metodu için position_percent_size pozitif olmali'
                    ))
            elif rm.sizing_method == PositionSizeMethod.FIXED_USD:
                # FIXED_USD requires position_usd_size
                if not hasattr(rm, 'position_usd_size') or rm.position_usd_size <= 0:
                    self.errors.append(ValidationError(
                        field='risk_management.position_usd_size',
                        value=getattr(rm, 'position_usd_size', None),
                        message='FIXED_USD metodu için position_usd_size pozitif olmali'
                    ))
            elif rm.sizing_method == PositionSizeMethod.RISK_BASED:
                # RISK_BASED requires max_risk_per_trade
                if not hasattr(rm, 'max_risk_per_trade') or rm.max_risk_per_trade <= 0:
                    self.errors.append(ValidationError(
                        field='risk_management.max_risk_per_trade',
                        value=getattr(rm, 'max_risk_per_trade', None),
                        message='RISK_BASED metodu için max_risk_per_trade pozitif olmali'
                    ))
                elif rm.max_risk_per_trade > 10:
                    self.warnings.append(ValidationError(
                        field='risk_management.max_risk_per_trade',
                        value=rm.max_risk_per_trade,
                        message='Trade basina %10+ risk cok yuksek!',
                        severity='warning'
                    ))
        
        # Portfolio risk kontrolü
        if hasattr(rm, 'max_portfolio_risk'):
            if rm.max_portfolio_risk <= 0:
                self.errors.append(ValidationError(
                    field='risk_management.max_portfolio_risk',
                    value=rm.max_portfolio_risk,
                    message='max_portfolio_risk pozitif olmali'
                ))
        
        # Max drawdown kontrolü
        if hasattr(rm, 'max_drawdown'):
            if rm.max_drawdown < 5:
                self.warnings.append(ValidationError(
                    field='risk_management.max_drawdown',
                    value=rm.max_drawdown,
                    message='Max drawdown cok dusuk (<5%), sik durma olabilir!',
                    severity='warning'
                ))
    
    def _validate_exit_strategy(self, es: Any):
        """ExitStrategy parametrelerini validate et (ENUM + değerler)"""
        # Enum validations (sadece enum import edildiyse)
        if ExitMethod is not None and hasattr(es, 'take_profit_method'):
            if not isinstance(es.take_profit_method, ExitMethod):
                self.errors.append(ValidationError(
                    field='exit_strategy.take_profit_method',
                    value=type(es.take_profit_method).__name__,
                    message='take_profit_method ExitMethod enum olmali'
                ))
        
        if StopLossMethod is not None and hasattr(es, 'stop_loss_method'):
            if not isinstance(es.stop_loss_method, StopLossMethod):
                self.errors.append(ValidationError(
                    field='exit_strategy.stop_loss_method',
                    value=type(es.stop_loss_method).__name__,
                    message='stop_loss_method StopLossMethod enum olmali'
                ))

        # Take profit kontrolü
        if hasattr(es, 'take_profit_value'):
            if es.take_profit_value <= 0:
                self.errors.append(ValidationError(
                    field='exit_strategy.take_profit_value',
                    value=es.take_profit_value,
                    message='take_profit_value pozitif olmali'
                ))
        
        # Stop loss kontrolü
        if hasattr(es, 'stop_loss_value'):
            if es.stop_loss_value <= 0:
                self.errors.append(ValidationError(
                    field='exit_strategy.stop_loss_value',
                    value=es.stop_loss_value,
                    message='stop_loss_value pozitif olmali'
                ))
        
        # TP/SL oranı kontrolü (risk/reward)
        if hasattr(es, 'take_profit_value') and hasattr(es, 'stop_loss_value'):
            if es.take_profit_value > 0 and es.stop_loss_value > 0:
                ratio = es.take_profit_value / es.stop_loss_value
                if ratio < 1.0:
                    self.warnings.append(ValidationError(
                        field='exit_strategy',
                        value=f'TP/SL={ratio:.2f}',
                        message=f'TP/SL orani dusuk ({ratio:.2f}), risk/reward olumsuz!',
                        severity='warning'
                    ))
        
        # Trailing stop kontrolü
        if hasattr(es, 'trailing_stop_enabled') and es.trailing_stop_enabled:
            if not hasattr(es, 'trailing_activation_profit_percent'):
                self.errors.append(ValidationError(
                    field='exit_strategy.trailing_activation_profit_percent',
                    value=None,
                    message='Trailing stop aktif ama aktivasyon seviyesi yok!'
                ))
        
        # Partial exit kontrolü
        if hasattr(es, 'partial_exit_enabled') and es.partial_exit_enabled:
            if not hasattr(es, 'partial_exit_levels') or not es.partial_exit_levels:
                self.errors.append(ValidationError(
                    field='exit_strategy.partial_exit_levels',
                    value=None,
                    message='partial_exit_enabled=True ama partial_exit_levels bos'
                ))
            
            if not hasattr(es, 'partial_exit_sizes') or not es.partial_exit_sizes:
                self.errors.append(ValidationError(
                    field='exit_strategy.partial_exit_sizes',
                    value=None,
                    message='partial_exit_enabled=True ama partial_exit_sizes bos'
                ))
            
            if hasattr(es, 'partial_exit_levels') and hasattr(es, 'partial_exit_sizes'):
                if len(es.partial_exit_levels) != len(es.partial_exit_sizes):
                    self.errors.append(ValidationError(
                        field='exit_strategy.partial_exit',
                        value=f'levels:{len(es.partial_exit_levels)}, sizes:{len(es.partial_exit_sizes)}',
                        message='partial_exit_levels ve partial_exit_sizes ayni uzunlukta olmali'
                    ))
    
    def _validate_position_management(self, pm: Any):
        """PositionManagement parametrelerini validate et"""
        # Pozisyon limit kontrolü
        if hasattr(pm, 'max_total_positions'):
            if pm.max_total_positions <= 0:
                self.errors.append(ValidationError(
                    field='position_management.max_total_positions',
                    value=pm.max_total_positions,
                    message='max_total_positions pozitif olmali'
                ))
            elif pm.max_total_positions > 20:
                self.warnings.append(ValidationError(
                    field='position_management.max_total_positions',
                    value=pm.max_total_positions,
                    message='Cok fazla es zamanli pozisyon (>20), riskli olabilir!',
                    severity='warning'
                ))
        
        if hasattr(pm, 'max_positions_per_symbol'):
            if pm.max_positions_per_symbol <= 0:
                self.errors.append(ValidationError(
                    field='position_management.max_positions_per_symbol',
                    value=pm.max_positions_per_symbol,
                    message='max_positions_per_symbol pozitif olmali'
                ))
        
        # Mantıksal kontrol
        if hasattr(pm, 'max_positions_per_symbol') and hasattr(pm, 'max_total_positions'):
            if pm.max_positions_per_symbol > pm.max_total_positions:
                self.errors.append(ValidationError(
                    field='position_management',
                    value=f'per_symbol:{pm.max_positions_per_symbol}, total:{pm.max_total_positions}',
                    message='max_positions_per_symbol max_total_positions\'dan buyuk olamaz'
                ))
        
        # Pyramiding kontrolü
        if hasattr(pm, 'pyramiding_enabled') and pm.pyramiding_enabled:
            if not hasattr(pm, 'pyramiding_max_entries'):
                self.errors.append(ValidationError(
                    field='position_management.pyramiding_max_entries',
                    value=None,
                    message='Pyramiding aktif ama max_entries yok!'
                ))
            elif pm.pyramiding_max_entries > 10:
                self.warnings.append(ValidationError(
                    field='position_management.pyramiding_max_entries',
                    value=pm.pyramiding_max_entries,
                    message='Cok fazla pyramiding entry (>10), cok riskli!',
                    severity='warning'
                ))
        
        # Timeout kontrolü
        if hasattr(pm, 'position_timeout_enabled') and pm.position_timeout_enabled:
            if not hasattr(pm, 'position_timeout'):
                self.errors.append(ValidationError(
                    field='position_management.position_timeout',
                    value=None,
                    message='Timeout aktif ama timeout suresi yok!'
                ))
    
    def _validate_conditions(self, strategy: Any, field_name: str):
        """Entry/Exit conditions'ları validate et (DETAYLI)"""
        conditions = getattr(strategy, field_name, {})
        
        if not isinstance(conditions, dict):
            self.errors.append(ValidationError(
                field=field_name,
                value=type(conditions).__name__,
                message=f'{field_name} dict olmali!'
            ))
            return
        
        # En az long veya short olmalı (entry için)
        if field_name == 'entry_conditions':
            if 'long' not in conditions and 'short' not in conditions:
                self.errors.append(ValidationError(
                    field=field_name,
                    value=conditions.keys(),
                    message="entry_conditions'da en az 'long' veya 'short' olmali"
                ))
        
        # Her bir koşul grubunu kontrol et
        for side, cond_list in conditions.items():
            if not isinstance(cond_list, list):
                self.errors.append(ValidationError(
                    field=f'{field_name}.{side}',
                    value=type(cond_list).__name__,
                    message=f'{side} kosullari list olmali!'
                ))
                continue
            
            # Her bir koşulu kontrol et
            for idx, condition in enumerate(cond_list):
                self._validate_single_condition(condition, f'{field_name}.{side}[{idx}]', strategy)
    
    def _validate_single_condition(self, condition: Any, context: str, strategy: Any):
        """Tek bir koşulu doğrula (DETAYLI: syntax, operator, timeframe)"""
        if not isinstance(condition, (list, tuple)):
            self.errors.append(ValidationError(
                field=context,
                value=type(condition).__name__,
                message='Kosul list/tuple olmali!'
            ))
            return
        
        if len(condition) < 3:
            self.errors.append(ValidationError(
                field=context,
                value=condition,
                message='Kosul en az 3 eleman icermeli: [indicator, operator, value]'
            ))
            return
        
        if len(condition) > 4:
            self.errors.append(ValidationError(
                field=context,
                value=condition,
                message='Kosul en fazla 4 eleman icermeli: [indicator, operator, value, timeframe]'
            ))
            return
        
        # Operator check
        operator = condition[1]
        if operator not in self.VALID_OPERATORS:
            self.errors.append(ValidationError(
                field=f'{context}.operator',
                value=operator,
                message=f"Gecersiz operator: '{operator}'. Gecerli: {sorted(self.VALID_OPERATORS)}"
            ))
        
        # Timeframe check (if exists)
        if len(condition) == 4:
            timeframe = condition[3]
            if timeframe and timeframe not in self.VALID_TIMEFRAMES:
                self.errors.append(ValidationError(
                    field=f'{context}.timeframe',
                    value=timeframe,
                    message=f"Gecersiz timeframe: '{timeframe}'. Gecerli: {sorted(self.VALID_TIMEFRAMES)}"
                ))
            
            # Timeframe, mtf_timeframes içinde olmalı
            if hasattr(strategy, 'mtf_timeframes') and timeframe:
                if timeframe not in strategy.mtf_timeframes:
                    self.warnings.append(ValidationError(
                        field=f'{context}.timeframe',
                        value=timeframe,
                        message=f"Timeframe '{timeframe}' mtf_timeframes icinde degil!",
                        severity='warning'
                    ))
    
    def _validate_custom_parameters(self, params: Dict):
        """Custom parameters'ı validate et"""
        if not isinstance(params, dict):
            self.errors.append(ValidationError(
                field='custom_parameters',
                value=type(params).__name__,
                message='custom_parameters dict olmali!'
            ))
    
    def _validate_optimizer_parameters(self, params: Any):
        """Optimizer parameters'ı validate et"""
        # Optimizer parametreleri opsiyonel, sadece varsa kontrol et
        if params is None:
            return
        
        if not isinstance(params, dict):
            self.errors.append(ValidationError(
                field='optimizer_parameters',
                value=type(params).__name__,
                message='optimizer_parameters dict olmali!'
            ))
    
    def _validate_account_management(self, acc_mgmt: Any):
        """Account management config'ini doğrula"""
        # Leverage validation
        if hasattr(acc_mgmt, 'leverage'):
            if acc_mgmt.leverage < 1 or acc_mgmt.leverage > 125:
                self.errors.append(ValidationError(
                    field='account_management.leverage',
                    value=acc_mgmt.leverage,
                    message='leverage 1-125 arasinda olmali'
                ))
        
        # Initial balance validation
        if hasattr(acc_mgmt, 'initial_balance'):
            if acc_mgmt.initial_balance <= 0:
                self.errors.append(ValidationError(
                    field='account_management.initial_balance',
                    value=acc_mgmt.initial_balance,
                    message='initial_balance pozitif olmali'
                ))
    
    def _validate_backtest_parameters(self, strategy: Any):
        """Backtest parameters config'ini doğrula"""
        bt_params = strategy.backtest_parameters
        
        # Date validation
        if hasattr(strategy, 'backtest_start_date') and hasattr(strategy, 'backtest_end_date'):
            if strategy.backtest_start_date and strategy.backtest_end_date:
                if strategy.backtest_start_date >= strategy.backtest_end_date:
                    self.errors.append(ValidationError(
                        field='backtest_dates',
                        value=f'start:{strategy.backtest_start_date}, end:{strategy.backtest_end_date}',
                        message='backtest_start_date backtest_end_date\'den once olmali'
                    ))
        
        # Commission validation
        if hasattr(bt_params, 'commission'):
            if bt_params.commission < 0:
                self.errors.append(ValidationError(
                    field='backtest_parameters.commission',
                    value=bt_params.commission,
                    message='commission negatif olamaz'
                ))
        
        # Slippage validation
        if hasattr(bt_params, 'max_slippage'):
            if bt_params.max_slippage < 0:
                self.errors.append(ValidationError(
                    field='backtest_parameters.max_slippage',
                    value=bt_params.max_slippage,
                    message='max_slippage negatif olamaz'
                ))


def validate_strategy(strategy: Any, logger: Optional[Any] = None) -> StrategyValidationResult:
    """
    Stratejiyi validate et (helper function)
    
    Args:
        strategy: Strateji objesi
        logger: Opsiyonel logger
    
    Returns:
        StrategyValidationResult
    
    Örnek:
        >>> from components.strategies.templates.Golden_Cross_Trend import Strategy
        >>> strategy = Strategy()
        >>> result = validate_strategy(strategy)
        >>> if result.valid:
        >>>     print("Strateji kullanilabilir!")
        >>> else:
        >>>     for error in result.errors:
        >>>         print(f"Hata: {error.message}")
    """
    validator = StrategyValidator(logger=logger)
    return validator.validate_strategy(strategy)


# Geriye uyumluluk için eski exception-based function
class ValidationException(Exception):
    """Validation exception (geriye uyumluluk için)"""
    pass


def validate_strategy_strict(strategy: Any) -> None:
    """
    Stratejiyi validate et ve hata varsa exception fırlat
    
    (Geriye uyumluluk için: helpers/validation.py'deki eski davranış)
    
    Args:
        strategy: Strateji objesi
    
    Raises:
        ValidationException: Validation hatası varsa
    """
    result = validate_strategy(strategy)
    if not result.valid:
        error_messages = [f"{err.field}: {err.message}" for err in result.errors]
        raise ValidationException("\n".join(error_messages))


__all__ = [
    'StrategyValidator',
    'ValidationError',
    'StrategyValidationResult',
    'ValidationException',
    'validate_strategy',
    'validate_strategy_strict',
]
