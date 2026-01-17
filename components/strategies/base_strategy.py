#!/usr/bin/env python3
"""
components/strategies/base_strategy.py
SuperBot - Base Strategy Class & Config Types

Version: 1.0.0
Date: 2025-11-13
Author: SuperBot Team

Description:
    Base strategy class ve tüm config type'ları.
    Her strategy template bu class'ı inherit eder.

Kullanım:
    from components.strategies.base_strategy import BaseStrategy, TradingSide
    
    class MyStrategy(BaseStrategy):
        def __init__(self):
            super().__init__()
            self.strategy_name = "My Strategy"
            self.symbols = [...]
            self.entry_conditions = {...}
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


# ============================================================================
# ENUMS
# ============================================================================

class TradingSide(str, Enum):
    """Trading yönü - Strateji hangi yönde pozisyon açabilir"""
    LONG = "LONG"      # Sadece LONG pozisyon aç
    SHORT = "SHORT"    # Sadece SHORT pozisyon aç
    BOTH = "BOTH"      # Hem LONG hem SHORT açabilir
    FLAT = "FLAT"      # Hiçbir pozisyon açma (pause)


class PositionSizeMethod(str, Enum):
    """Pozisyon boyutu hesaplama metodları"""
    FIXED_USD = "FIXED_USD"                  # Sabit dolar ($100)
    FIXED_PERCENT = "FIXED_PERCENT"          # Sabit yüzde (%5 of capital)
    FIXED_QUANTITY = "FIXED_QUANTITY"        # Sabit miktar (0.01 BTC)
    RISK_BASED = "RISK_BASED"                # Risk bazlı (stop loss'a göre)
    KELLY_CRITERION = "KELLY_CRITERION"      # Kelly formülü
    VOLATILITY_SCALED = "VOLATILITY_SCALED"  # ATR/volatility bazlı
    DYNAMIC_AI = "DYNAMIC_AI"                # AI-based dynamic sizing


class ExitMethod(str, Enum):
    """Take profit metodları"""
    FIXED_PERCENT = "FIXED_PERCENT"          # Sabit yüzde (%2)
    FIXED_PRICE = "FIXED_PRICE"              # Sabit fiyat ($45000)
    RISK_REWARD = "RISK_REWARD"              # Risk/Reward ratio (1:2)
    ATR_BASED = "ATR_BASED"                  # ATR çarpanı (2x ATR)
    FIBONACCI = "FIBONACCI"                  # Fibonacci seviyeleri
    DYNAMIC_AI = "DYNAMIC_AI"                # AI-based dynamic exit


class StopLossMethod(str, Enum):
    """Stop loss metodları"""
    FIXED_PERCENT = "FIXED_PERCENT"          # Sabit yüzde (%1)
    FIXED_PRICE = "FIXED_PRICE"              # Sabit fiyat ($95000)
    ATR_BASED = "ATR_BASED"                  # ATR çarpanı (1.5x ATR)
    SWING_POINTS = "SWING_POINTS"            # Swing low/high
    FIBONACCI = "FIBONACCI"                  # Fibonacci retracement
    DYNAMIC_AI = "DYNAMIC_AI"                # AI-based adaptive SL


# ============================================================================
# CONFIG DATACLASSES
# ============================================================================

@dataclass
class SymbolConfig:
    """
    Symbol konfigürasyonu
    
    Attributes:
        symbol: Base asset listesi (örn: ['BTC', 'ETH'])
        quote: Quote currency (örn: 'USDT')
        enabled: Bu semboller trade edilsin mi?
    """
    symbol: List[str]
    quote: str
    enabled: bool = True


@dataclass
class TechnicalParameters:
    """
    Teknik indikatör parametreleri
    
    Attributes:
        indicators: Indikatör dict'i
            Format: {
                "rsi_14": {"period": 14, "overbought": 70, "oversold": 30},
                "ema_50": {"period": 50},
                ...
            }
    """
    indicators: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RiskManagement:
    """
    Risk yönetimi parametreleri

    POSITION SIZING METHODS (sadece leverage ayarla, gerisi otomatik!):

    1. FIXED_PERCENT (ÖNERİLEN - En basit):
       - size_value: Portfolio'nun kaç %'i (leverage ile çarpılır)
       - max_risk_per_trade: KULLANILMAZ (ignore edilir)
       - Örnek: size_value=10, leverage=5 → Her trade %50 pozisyon (10% × 5x)

    2. FIXED_USD:
       - size_value: Her trade'de kaç $ (leverage UYGULANMAZ!)
       - max_risk_per_trade: KULLANILMAZ
       - Örnek: size_value=1000 → Her trade $1000 pozisyon

    3. RISK_BASED (KARMAŞIK - Dikkatli kullan!):
       - size_value: KULLANILMAZ
       - max_risk_per_trade: Kaybetmeye hazır olduğun % (stop_loss ile bölünür)
       - Formül: Position = (Portfolio × max_risk_per_trade) / stop_loss_distance
       - Örnek: max_risk=2%, stop_loss=2%, leverage=1 → %100 pozisyon
       - ⚠️ UYARI: max_risk/stop_loss > leverage ise LIMIT AŞIMI olur!

    Attributes:
        sizing_method: Pozisyon boyutu hesaplama metodu (yukarıdaki 3'ten biri)
        size_value: Pozisyon boyutu değeri (method'a göre değişir - yukarıya bak)
        max_risk_per_trade: Her trade'de maksimum risk (sadece RISK_BASED'de kullanılır)
        max_correlation: Maksimum korelasyon limiti
        position_correlation_limit: Pozisyon korelasyon limiti
        max_drawdown: Maksimum drawdown (%)
        max_daily_trades: Günlük maksimum trade sayısı
        emergency_stop_enabled: Acil durum durdurucu aktif mi?
        ai_risk_enabled: AI risk yönetimi aktif mi? (henüz implement edilmedi)
        dynamic_position_sizing: Dinamik pozisyon boyutu aktif mi?

    Note:
        max_portfolio_risk otomatik hesaplanır: strategy.leverage × 100
        Örnek: leverage=5 → max_portfolio_risk=500% (5x kaldıraç ile max %500 pozisyon)
    """
    sizing_method: PositionSizeMethod

    # Position sizing parameters (her method kendi parametresini kullanır)
    position_percent_size: float = 10.0      # FIXED_PERCENT için: Portfolio'nun kaç %'i
    position_usd_size: float = 1000.0        # FIXED_USD için: Kaç dolar
    position_quantity_size: float = 0.01     # FIXED_QUANTITY için: Kaç adet (örn: 0.01 BTC)
    max_risk_per_trade: float = 2.0          # RISK_BASED için: Kaybetmeye hazır %

    # Backward compatibility (deprecated - kullanma!)
    size_value: float = 0.0                  # DEPRECATED: Yeni parametreleri kullan (position_*_size)

    # max_portfolio_risk: REMOVED - Now auto-calculated from strategy.leverage
    max_correlation: float = 0.7
    position_correlation_limit: float = 0.7
    max_drawdown: float = 20.0
    max_daily_trades: int = 100
    emergency_stop_enabled: bool = False
    ai_risk_enabled: bool = False
    dynamic_position_sizing: bool = False


@dataclass
class PositionManagement:
    """
    Pozisyon yönetimi parametreleri
    
    Attributes:
        max_positions_per_symbol: Her symbol için maksimum pozisyon
        max_total_positions: Toplam maksimum pozisyon sayısı
        allow_hedging: Hedging izni
        position_timeout_enabled: Timeout kontrolü aktif mi?
        position_timeout: Pozisyon timeout süresi (dakika)
        pyramiding_enabled: Pyramiding aktif mi?
        pyramiding_max_entries: Maksimum pyramiding entry sayısı
        pyramiding_scale_factor: Her entry'de boyut çarpanı
    """
    max_positions_per_symbol: int
    max_total_positions: int
    allow_hedging: bool = False
    position_timeout_enabled: bool = False
    position_timeout: int = 1800  # dakika (default: 30 saat)
    pyramiding_enabled: bool = False
    pyramiding_max_entries: int = 3
    pyramiding_scale_factor: float = 0.5


@dataclass
class ExitStrategy:
    """
    Exit stratejisi parametreleri

    Attributes:
        # Take Profit Methods
        take_profit_method: TP hesaplama metodu
        take_profit_percent: FIXED_PERCENT için %TP
        take_profit_price: FIXED_PRICE için fiyat
        take_profit_risk_reward_ratio: RISK_REWARD için R/R oranı
        take_profit_atr_multiplier: ATR_BASED için ATR çarpanı
        take_profit_fib_level: FIBONACCI için extension seviyesi
        take_profit_ai_level: DYNAMIC_AI için seviye

        # Stop Loss Methods
        stop_loss_method: SL hesaplama metodu
        stop_loss_percent: FIXED_PERCENT için %SL
        stop_loss_price: FIXED_PRICE için fiyat
        stop_loss_atr_multiplier: ATR_BASED için ATR çarpanı
        stop_loss_swing_lookback: SWING_POINTS için lookback period
        stop_loss_fib_level: FIBONACCI için retracement seviyesi
        stop_loss_ai_level: DYNAMIC_AI için seviye

        # Trailing Stop
        trailing_stop_enabled: Trailing stop aktif mi?
        trailing_activation_profit_percent: Trailing aktif olacağı kar yüzdesi
        trailing_callback_percent: Geri çekilme yüzdesi
        trailing_take_profit: TP'ye ulaşınca trail başlat
        trailing_distance: TP'den uzaklık

        # Break Even
        break_even_enabled: Break-even aktif mi?
        break_even_trigger_profit_percent: Tetikleme kar yüzdesi
        break_even_offset: Entry'den offset

        # Partial Exit
        partial_exit_enabled: Kısmi çıkış aktif mi?
        partial_exit_levels: Kar seviyeleri (%)
        partial_exit_sizes: Her seviyede kapatılacak miktar
    """
    # Take Profit (method is required, all parameters optional)
    take_profit_method: ExitMethod = ExitMethod.FIXED_PERCENT
    take_profit_percent: float = 0.0          # FIXED_PERCENT
    take_profit_price: float = 0.0            # FIXED_PRICE
    take_profit_risk_reward_ratio: float = 0.0  # RISK_REWARD
    take_profit_atr_multiplier: float = 0.0   # ATR_BASED
    take_profit_fib_level: float = 0.0        # FIBONACCI
    take_profit_ai_level: int = 0             # DYNAMIC_AI

    # Stop Loss (method is required, all parameters optional)
    stop_loss_method: StopLossMethod = StopLossMethod.FIXED_PERCENT
    stop_loss_percent: float = 0.0            # FIXED_PERCENT
    stop_loss_price: float = 0.0              # FIXED_PRICE
    stop_loss_atr_multiplier: float = 0.0     # ATR_BASED
    stop_loss_swing_lookback: int = 0         # SWING_POINTS
    stop_loss_fib_level: float = 0.0          # FIBONACCI
    stop_loss_ai_level: int = 0               # DYNAMIC_AI

    # Trailing Stop
    trailing_stop_enabled: bool = False
    trailing_activation_profit_percent: float = 1.0
    trailing_callback_percent: float = 0.5
    trailing_take_profit: bool = False
    trailing_distance: float = 0.5
    
    # Break Even
    break_even_enabled: bool = False
    break_even_trigger_profit_percent: float = 1.0
    break_even_offset: float = 0.1
    
    # Partial Exit
    partial_exit_enabled: bool = False
    partial_exit_levels: List[float] = field(default_factory=list)
    partial_exit_sizes: List[float] = field(default_factory=list)


@dataclass
class AIConfig:
    """
    AI Model Konfigürasyonu

    AI modeli strateji kararlarına yardımcı olur:
    - Entry Decision: Sinyalin kalitesini değerlendir
    - TP/SL Optimization: Optimal TP/SL öner
    - Position Sizing: Risk bazlı pozisyon boyutu
    - Exit Timing: Çıkış zamanlaması

    Attributes:
        # General
        ai_enabled: AI aktif mi?
        model_path: Model checkpoint dosyası
        model_type: Model tipi (rl_model, signal_model, lstm, transformer)

        # Entry Decision
        entry_decision: AI giriş kararında kullanılsın mı?
        confidence_threshold: Minimum güven eşiği (0.5-0.9)

        # TP/SL Optimization
        tp_optimization: AI optimal TP önersin mi?
        sl_optimization: AI optimal SL önersin mi?
        use_ai_tp: AI TP'yi direkt kullan (override strategy TP)
        use_ai_sl: AI SL'yi direkt kullan (override strategy SL)

        # Position Sizing
        position_sizing: AI pozisyon boyutu önersin mi?
        risk_assessment: AI risk değerlendirmesi yapsın mı?
        max_ai_position_mult: AI max pozisyon çarpanı (1.0 = normal)

        # Exit Timing
        exit_timing: AI çıkış zamanlaması önersin mi?
        early_exit_enabled: Erken çıkış aktif mi?
        early_exit_threshold: Erken çıkış için kayıp eşiği

        # Break Even & Trailing
        ai_break_even: AI break-even tetiklemesi
        ai_trailing: AI trailing stop yönetimi

        # Lookback/Forward Config (per timeframe)
        lookback_bars: Kaç bar geriye bak (feature extraction)
        forward_bars: Kaç bar ileriye bak (default)
        forward_bars_1m: 1m için forward bars
        forward_bars_5m: 5m için forward bars
        forward_bars_15m: 15m için forward bars
        forward_bars_30m: 30m için forward bars
        forward_bars_1h: 1h için forward bars
        forward_bars_4h: 4h için forward bars
        forward_bars_1d: 1d için forward bars
        forward_bars_1w: 1w için forward bars

    Example:
        ai_config = AIConfig(
            ai_enabled=True,
            model_path="data/checkpoints/ai/global_best.pt",
            entry_decision=True,
            confidence_threshold=0.6,
            tp_optimization=True,
            sl_optimization=True,
        )
    """
    # ═══════════════════════════════════════════════════════════════
    # GENERAL
    # ═══════════════════════════════════════════════════════════════
    ai_enabled: bool = False
    model_path: str = "data/checkpoints/ai/global_best.pt"
    model_type: str = "rl_model"  # rl_model (PPO), signal_model (legacy), lstm, transformer

    # ═══════════════════════════════════════════════════════════════
    # ENTRY DECISION
    # ═══════════════════════════════════════════════════════════════
    entry_decision: bool = True           # AI giriş kararında kullanılsın mı?
    confidence_threshold: float = 0.6     # Min güven (LONG > threshold, SHORT < 1-threshold)

    # ═══════════════════════════════════════════════════════════════
    # TP/SL OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════
    tp_optimization: bool = False         # AI optimal TP önersin mi?
    sl_optimization: bool = False         # AI optimal SL önersin mi?
    use_ai_tp: bool = False               # AI TP'yi direkt kullan (strategy TP'yi override et)
    use_ai_sl: bool = False               # AI SL'yi direkt kullan (strategy SL'yi override et)
    tp_blend_ratio: float = 0.5           # AI/Strategy TP blend oranı (0=strategy, 1=AI)
    sl_blend_ratio: float = 0.5           # AI/Strategy SL blend oranı (0=strategy, 1=AI)

    # ═══════════════════════════════════════════════════════════════
    # POSITION SIZING
    # ═══════════════════════════════════════════════════════════════
    position_sizing: bool = False         # AI pozisyon boyutu önersin mi?
    risk_assessment: bool = False         # AI risk değerlendirmesi yapsın mı?
    max_ai_position_mult: float = 1.5     # AI max pozisyon çarpanı (güven yüksekse büyüt)
    min_ai_position_mult: float = 0.5     # AI min pozisyon çarpanı (güven düşükse küçült)

    # ═══════════════════════════════════════════════════════════════
    # EXIT TIMING
    # ═══════════════════════════════════════════════════════════════
    exit_timing: bool = False             # AI çıkış zamanlaması önersin mi?
    early_exit_enabled: bool = False      # Erken çıkış aktif mi?
    early_exit_profit_threshold: float = 0.5   # Min kar % için erken çıkış
    early_exit_loss_threshold: float = -1.0    # Max kayıp % için erken çıkış

    # ═══════════════════════════════════════════════════════════════
    # BREAK EVEN & TRAILING
    # ═══════════════════════════════════════════════════════════════
    ai_break_even: bool = False           # AI break-even tetiklemesi
    ai_trailing: bool = False             # AI trailing stop yönetimi
    ai_partial_exit: bool = False         # AI partial exit kararı

    # ═══════════════════════════════════════════════════════════════
    # EXIT MODEL (Dynamic Exit Optimization)
    # ═══════════════════════════════════════════════════════════════
    exit_model_enabled: bool = False      # Exit Model aktif mi?
    exit_model_path: str = "data/ai/checkpoints/simple_train/simple_rsi/exit_model.pkl"
    use_exit_model_tp: bool = False       # Exit Model TP'yi override etsin mi?
    use_exit_model_sl: bool = False       # Exit Model SL'yi override etsin mi?
    use_exit_model_trailing: bool = False # Exit Model trailing kararını alsın mı?
    use_exit_model_break_even: bool = False # Exit Model BE kararını alsın mı?
    exit_model_blend_ratio: float = 1.0  # Exit Model blend oranı (0=strategy, 1=AI)

    # ═══════════════════════════════════════════════════════════════
    # LOOKBACK/FORWARD CONFIG
    # ═══════════════════════════════════════════════════════════════
    lookback_bars: int = 200              # Feature extraction için geriye bakış

    # Forward bars (prediction horizon) - timeframe bazlı
    forward_bars: int = 24                # Default forward bars
    forward_bars_1m: int = 24             # 1m: 24 bar = 24 dakika
    forward_bars_5m: int = 24             # 5m: 24 bar = 2 saat
    forward_bars_15m: int = 24            # 15m: 24 bar = 6 saat
    forward_bars_30m: int = 24            # 30m: 24 bar = 12 saat
    forward_bars_1h: int = 24             # 1h: 24 bar = 24 saat
    forward_bars_4h: int = 24             # 4h: 24 bar = 4 gün
    forward_bars_1d: int = 12             # 1d: 12 bar = 12 gün
    forward_bars_1w: int = 5              # 1w: 5 bar = 5 hafta

    def get_forward_bars(self, timeframe: str) -> int:
        """Get forward bars for specific timeframe."""
        tf_map = {
            '1m': self.forward_bars_1m,
            '5m': self.forward_bars_5m,
            '15m': self.forward_bars_15m,
            '30m': self.forward_bars_30m,
            '1h': self.forward_bars_1h,
            '4h': self.forward_bars_4h,
            '1d': self.forward_bars_1d,
            '1w': self.forward_bars_1w,
        }
        return tf_map.get(timeframe, self.forward_bars)


# ============================================================================
# BASE STRATEGY CLASS
# ============================================================================

class BaseStrategy(ABC):
    """
    Base strategy class
    
    Tüm strategy template'leri bu class'ı inherit eder.
    
    Örnek:
        class MyStrategy(BaseStrategy):
            def __init__(self):
                super().__init__()
                
                # Metadata
                self.strategy_name = "My Strategy"
                self.strategy_version = "1.0.0"
                
                # Config
                self.symbols = [SymbolConfig(...)]
                self.risk_management = RiskManagement(...)
                self.entry_conditions = {...}
    """
    
    def __init__(self):
        """Initialize base strategy"""
        
        # ====================================================================
        # METADATA
        # ====================================================================
        self.strategy_name: str = "Unnamed Strategy"
        self.strategy_version: str = "1.0.0"
        self.description: str = ""
        self.author: str = "Unknown"
        self.created_date: str = ""
        
        # ====================================================================
        # DATA MANAGEMENT
        # ====================================================================
        self.backtesting_enabled: bool = False
        self.backtest_start_date: Optional[str] = None
        self.backtest_end_date: Optional[str] = None
        self.initial_balance: float = 10000.0
        self.download_klines: bool = False
        self.update_klines: bool = False
        self.warmup_period: int = 200
        
        # Backtest parametreleri
        self.backtest_parameters: Dict[str, Any] = {
            "min_spread": 0.0,
            "commission": 0.0,
            "max_slippage": 0.0
        }
        
        # ====================================================================
        # SYMBOL MANAGEMENT
        # ====================================================================
        self.symbol_source: str = "strategy"
        self.symbols: List[SymbolConfig] = []
        
        # ====================================================================
        # ACCOUNT MANAGEMENT
        # ====================================================================
        self.side_method: TradingSide = TradingSide.BOTH
        self.leverage: int = 1
        self.set_default_leverage: bool = False
        self.hedge_mode: bool = False
        self.set_margin_type: bool = False
        self.margin_type: str = "isolated"
        
        # ====================================================================
        # INDICATOR MANAGEMENT
        # ====================================================================
        self.mtf_timeframes: List[str] = ["5m"]
        self.primary_timeframe: str = "5m"
        self.technical_parameters: TechnicalParameters = TechnicalParameters()
        
        # ====================================================================
        # SIGNAL MANAGEMENT
        # ====================================================================
        self.entry_conditions: Dict[str, List] = {
            "long": [],
            "short": []
        }
        
        # ====================================================================
        # EXIT MANAGEMENT
        # ====================================================================
        self.exit_strategy: Optional[ExitStrategy] = None
        self.exit_conditions: Dict[str, List] = {
            "long": [],
            "short": [],
            "stop_loss": [],
            "take_profit": []
        }
        
        # ====================================================================
        # RISK MANAGEMENT
        # ====================================================================
        self.risk_management: Optional[RiskManagement] = None
        
        # ====================================================================
        # POSITION MANAGEMENT
        # ====================================================================
        self.position_management: Optional[PositionManagement] = None
        
        # ====================================================================
        # MARKET MANAGEMENT
        # ====================================================================
        self.custom_parameters: Dict[str, Any] = {
            "news_filter": False,
            "session_filter": {
                "enabled": False,
                "sydney": False,
                "tokyo": False,
                "london": False,
                "new_york": False,
                "london_ny_overlap": False,
            },
            "time_filter": {
                "enabled": False,
                "start_hour": 0,
                "end_hour": 24,
                "exclude_hours": [],
            },
            "day_filter": {
                "enabled": False,
                "monday": True,
                "tuesday": True,
                "wednesday": True,
                "thursday": True,
                "friday": True,
                "saturday": True,
                "sunday": True,
            },
        }
        
        # ====================================================================
        # OPTIMIZER MANAGEMENT (backtest için, trade'de kullanılmaz)
        # ====================================================================
        self.optimizer_parameters: Dict[str, Any] = {}

    def __init_subclass__(cls, **kwargs):
        """
        Alt sınıf oluşturulduğunda çağrılır.
        Primary TF'nin MTF içinde olmasını garanti eder.
        """
        super().__init_subclass__(**kwargs)

        # Original __init__'i wrap et
        original_init = cls.__init__

        def wrapped_init(self, *args, **kw):
            original_init(self, *args, **kw)
            self._ensure_primary_tf_in_mtf()

        cls.__init__ = wrapped_init

    def _ensure_primary_tf_in_mtf(self) -> None:
        """
        Primary timeframe'in MTF listesinde olmasını garanti et.

        Bazı stratejilerde primary_timeframe mtf_timeframes içinde olmayabilir.
        Bu güvenlik kontrolü her zaman primary TF'nin MTF'de olmasını sağlar.
        """
        if not hasattr(self, 'primary_timeframe') or not self.primary_timeframe:
            return

        if not hasattr(self, 'mtf_timeframes'):
            self.mtf_timeframes = []

        if self.primary_timeframe not in self.mtf_timeframes:
            self.mtf_timeframes.append(self.primary_timeframe)

    # ========================================================================
    # OPTIONAL OVERRIDE METHODS
    # ========================================================================
    
    def on_init(self) -> None:
        """
        Strategy initialization hook (optional override)
        
        Args:
            symbol: Trading symbol
        """
        pass
    
    def on_bar_close(self, symbol: str, timeframe: str, data: Any) -> None:
        """
        Called on every bar close (optional override)
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            data: Market data (DataFrame veya dict)
        """
        pass
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    def validate_config(self) -> bool:
        """
        Config validation (will be implemented by validation.py)
        
        Returns:
            True if valid, raises exception otherwise
        """
        # Bu method helpers/validation.py tarafından implement edilecek
        # Şimdilik placeholder
        return True
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def get_all_timeframes(self) -> List[str]:
        """Tüm kullanılan timeframe'leri dön"""
        return self.mtf_timeframes
    
    def get_indicator_names(self) -> List[str]:
        """Tüm indikatör isimlerini dön"""
        return list(self.technical_parameters.indicators.keys())
    
    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name='{self.strategy_name}' "
            f"version='{self.strategy_version}'>"
        )
    
    @property
    def max_portfolio_risk(self) -> float:
        """
        Auto-calculated maximum portfolio risk based on leverage

        Formula: max_portfolio_risk = leverage × 100

        Examples:
            leverage=1  → max_portfolio_risk=100  (no leverage, max 100% exposure)
            leverage=5  → max_portfolio_risk=500  (5x leverage, max 500% notional)
            leverage=10 → max_portfolio_risk=1000 (10x leverage, max 1000% notional)
            leverage=20 → max_portfolio_risk=2000 (20x leverage, max 2000% notional)

        Returns:
            Maximum portfolio risk percentage (auto-calculated)
        """
        return self.leverage * 100.0

    @property
    def symbol(self) -> str:
        """
        Get first symbol for backtesting (helper property)

        Returns:
            First symbol from symbols list (e.g., 'BTCUSDT')
            Defaults to 'BTCUSDT' if no symbols configured
        """
        if self.symbols and len(self.symbols) > 0:
            first_symbol_config = self.symbols[0]
            if first_symbol_config.symbol and len(first_symbol_config.symbol) > 0:
                # symbols = ['BTC', 'ETH'], quote = 'USDT' -> 'BTCUSDT'
                return f"{first_symbol_config.symbol[0]}{first_symbol_config.quote}"
        return "BTCUSDT"


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Enums
    'TradingSide',
    'PositionSizeMethod',
    'ExitMethod',
    'StopLossMethod',
    
    # Config dataclasses
    'SymbolConfig',
    'TechnicalParameters',
    'RiskManagement',
    'PositionManagement',
    'ExitStrategy',
    
    # Base class
    'BaseStrategy',
]
