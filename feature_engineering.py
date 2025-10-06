"""
Feature Engineering Module
Generates technical indicators and features for ML models
"""

import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice


class FeatureEngineer:
    def __init__(self, lookback_periods=[5, 10, 20, 50]):
        """
        Initialize feature engineer
        
        Args:
            lookback_periods: List of periods for rolling calculations
        """
        self.lookback_periods = lookback_periods
        
    def add_price_features(self, df):
        """Add price-based features"""
        df = df.copy()
        
        # Price changes
        df['price_change'] = df['close'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # High-Low range
        df['hl_range'] = (df['high'] - df['low']) / df['close']
        df['oc_range'] = (df['open'] - df['close']).abs() / df['close']
        
        # Price position within candle
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
        
        # Returns over different periods
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period)
            df[f'high_{period}'] = df['high'].rolling(period).max()
            df[f'low_{period}'] = df['low'].rolling(period).min()
            df[f'volatility_{period}'] = df['close'].pct_change().rolling(period).std()
        
        return df
    
    def add_trend_indicators(self, df):
        """Add trend-following indicators"""
        df = df.copy()
        
        # Simple Moving Averages
        for period in [7, 14, 21, 50]:
            sma = SMAIndicator(close=df['close'], window=period)
            df[f'sma_{period}'] = sma.sma_indicator()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}']
        
        # Exponential Moving Averages
        for period in [12, 26]:
            ema = EMAIndicator(close=df['close'], window=period)
            df[f'ema_{period}'] = ema.ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()
        
        # ADX (trend strength)
        adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        df['adx'] = adx.adx()
        
        return df
    
    def add_momentum_indicators(self, df):
        """Add momentum indicators"""
        df = df.copy()
        
        # RSI
        for period in [14, 21]:
            rsi = RSIIndicator(close=df['close'], window=period)
            df[f'rsi_{period}'] = rsi.rsi()
        
        # Stochastic
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change
        for period in [9, 14]:
            roc = ROCIndicator(close=df['close'], window=period)
            df[f'roc_{period}'] = roc.roc()
        
        return df
    
    def add_volatility_indicators(self, df):
        """Add volatility indicators"""
        df = df.copy()
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'])
        df['bb_high'] = bb.bollinger_hband()
        df['bb_mid'] = bb.bollinger_mavg()
        df['bb_low'] = bb.bollinger_lband()
        df['bb_width'] = bb.bollinger_wband()
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-10)
        
        # Average True Range
        atr = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        df['atr'] = atr.average_true_range()
        df['atr_percent'] = df['atr'] / df['close']
        
        return df
    
    def add_volume_indicators(self, df):
        """Add volume-based indicators"""
        df = df.copy()
        
        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        
        # Volume moving averages
        for period in [7, 14, 21]:
            df[f'volume_ma_{period}'] = df['volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_ma_{period}']
        
        # On Balance Volume
        obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
        df['obv'] = obv.on_balance_volume()
        df['obv_change'] = df['obv'].pct_change()
        
        # VWAP (if we have enough data)
        if len(df) >= 14:
            vwap = VolumeWeightedAveragePrice(
                high=df['high'], 
                low=df['low'], 
                close=df['close'], 
                volume=df['volume']
            )
            df['vwap'] = vwap.volume_weighted_average_price()
            df['price_to_vwap'] = df['close'] / df['vwap']
        
        return df
    
    def add_time_features(self, df):
        """Add time-based features"""
        df = df.copy()
        
        # Extract time components
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_month'] = df.index.day
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_market_context_features(self, df, order_book_context=None, ticker_context=None):
        """Add market context features from order book and ticker data"""
        df = df.copy()
        
        if order_book_context:
            df['bid_ask_imbalance'] = order_book_context['bid_volume'] / (
                order_book_context['ask_volume'] + 1e-10
            )
            df['spread_percent'] = order_book_context['spread'] / df['close'].iloc[-1]
        
        if ticker_context:
            df['price_change_24h'] = ticker_context['price_change_percent']
            df['high_low_range_24h'] = (
                ticker_context['high_price'] - ticker_context['low_price']
            ) / ticker_context['low_price']
        
        return df
    
    def create_target_labels(self, df, threshold_percent=0.5, lookahead_periods=6):
        """
        Create target labels for large price movements
        
        Args:
            df: DataFrame with OHLCV data
            threshold_percent: Threshold for "large" movement (0.5 = 0.5%)
            lookahead_periods: How many periods ahead to look for the movement
            
        Returns:
            DataFrame with target labels:
            - 0: No significant movement
            - 1: Large upward movement
            - 2: Large downward movement
        """
        df = df.copy()
        
        # Calculate future returns
        future_highs = df['high'].rolling(window=lookahead_periods, min_periods=1).max().shift(-lookahead_periods)
        future_lows = df['low'].rolling(window=lookahead_periods, min_periods=1).min().shift(-lookahead_periods)
        
        # Calculate percentage moves from current close
        upward_move = ((future_highs - df['close']) / df['close']) * 100
        downward_move = ((df['close'] - future_lows) / df['close']) * 100
        
        # Create labels
        df['target'] = 0  # No significant movement
        df.loc[upward_move >= threshold_percent, 'target'] = 1  # Large up
        df.loc[downward_move >= threshold_percent, 'target'] = 2  # Large down
        
        # Additional features: magnitude of expected movement
        df['expected_up_move'] = upward_move
        df['expected_down_move'] = downward_move
        df['expected_max_move'] = np.maximum(upward_move, downward_move)
        
        return df
    
    def generate_all_features(self, df, order_book_context=None, ticker_context=None,
                            create_targets=True, threshold_percent=0.5, lookahead_periods=6):
        """
        Generate all features at once
        
        Args:
            df: DataFrame with OHLCV data
            order_book_context: Optional order book context dict
            ticker_context: Optional ticker context dict
            create_targets: Whether to create target labels
            threshold_percent: Threshold for large movements
            lookahead_periods: Lookahead for target creation
            
        Returns:
            DataFrame with all features
        """
        print("Generating features...")
        
        # Apply all feature engineering steps
        df = self.add_price_features(df)
        print("  ✓ Price features")
        
        df = self.add_trend_indicators(df)
        print("  ✓ Trend indicators")
        
        df = self.add_momentum_indicators(df)
        print("  ✓ Momentum indicators")
        
        df = self.add_volatility_indicators(df)
        print("  ✓ Volatility indicators")
        
        df = self.add_volume_indicators(df)
        print("  ✓ Volume indicators")
        
        df = self.add_time_features(df)
        print("  ✓ Time features")
        
        if order_book_context or ticker_context:
            df = self.add_market_context_features(df, order_book_context, ticker_context)
            print("  ✓ Market context features")
        
        if create_targets:
            df = self.create_target_labels(df, threshold_percent, lookahead_periods)
            print("  ✓ Target labels created")
        
        # Drop rows with NaN values (from rolling calculations)
        initial_rows = len(df)
        df = df.dropna()
        print(f"  ✓ Removed {initial_rows - len(df)} rows with NaN values")
        
        print(f"Final feature set: {len(df)} rows × {len(df.columns)} features")
        
        return df
    
    def get_feature_columns(self, df):
        """Get list of feature columns (excluding OHLCV and target columns)"""
        exclude_cols = ['open', 'high', 'low', 'close', 'volume', 
                       'target', 'expected_up_move', 'expected_down_move', 'expected_max_move']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols
