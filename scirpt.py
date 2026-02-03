"""
Advanced Cryptocurrency Trading Bot with ML Predictions
Implements real-time market analysis, portfolio optimization, and risk management
"""

import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import hashlib
import hmac
import json
import logging
from dataclasses import dataclass
from enum import Enum
from decimal import Decimal
import signal
import sys

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import talib
from scipy import stats

# Performance monitoring
from memory_profiler import profile
import psutil
import gc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"

class TradeSignal(Enum):
    symbol: str
    close: float
    volume: float
@dataclass
class Order:
    order_id: str
    symbol: str
    order_type: OrderType
    side: str
    quantity: float
    price: Optional[float] = None
    timestamp: datetime = None
    status: str = "pending"
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        self.order_id = self._generate_order_id()
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID using hash"""
        data = f"{self.symbol}{self.side}{self.quantity}{datetime.utcnow().timestamp()}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

class PortfolioManager:
    """Advanced portfolio management with risk optimization"""
    
    def __init__(self, initial_capital: float = 10000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions: Dict[str, Dict] = {}
                               stop_loss: float) -> Tuple[float, float]:
        """Calculate optimal position size using Kelly Criterion"""
        if price_risk == 0:
            return 0, 0
            
        position_size = risk_amount / price_risk
        
        # Apply Kelly Criterion
        win_rate = 0.55  # Estimated from historical data
        avg_win_loss_ratio = 1.5
        kelly_f = win_rate - ((1 - win_rate) / avg_win_loss_ratio)
        kelly_position = position_size * kelly_f
        
        # Limit to 25% of portfolio
        max_position_value = self.current_capital * 0.25
        position_value = kelly_position * entry_price
        
        if position_value > max_position_value:
            kelly_position = max_position_value / entry_price
            
        return kelly_position, position_value
    
    def update_portfolio(self, symbol: str, quantity: float, 
                        price: float, side: str) -> None:
        """Update portfolio after trade execution"""
        trade_value = quantity * price
        
        if side == 'buy':
            self.current_capital -= trade_value
            if symbol in self.positions:
                self.positions[symbol]['quantity'] += quantity
                self.positions[symbol]['avg_price'] = (
                    self.positions[symbol]['avg_price'] + price) / 2
            else:
                self.positions[symbol] = {
                    'quantity': quantity,
                    'avg_price': price,
                    'entry_time': datetime.utcnow()
                }
        else:  # sell
            self.current_capital += trade_value
            if symbol in self.positions:
                self.positions[symbol]['quantity'] -= quantity
                if self.positions[symbol]['quantity'] <= 0:
                    del self.positions[symbol]
        
        self.trade_history.append({
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'value': trade_value
        })
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio for performance evaluation"""
        if len(returns) < 2:
            return 0
            
        excess_returns = returns - risk_free_rate / 252
        if excess_returns.std() == 0:
            return 0
            
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

class MLPredictor:
    """Machine learning model for price prediction"""
    
    def __init__(self):
        self.models = {
            'xgb': xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'rf': RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ),
            'gb': GradientBoostingRegressor(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators as features"""
        df = df.copy()
        
        # Price-based features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['price_change'] = df['close'] - df['open']
        
        # Volatility features
        df['volatility'] = df['returns'].rolling(window=20).std()
        df['high_low_ratio'] = df['high'] / df['low']
        
        # Volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_price_trend'] = df['volume'] * df['close'].pct_change()
        
        # Technical indicators using TA-Lib
        df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
            df['close'], timeperiod=20
        )
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # Moving averages
        df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
        df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
        df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
        
        # Price position features
        df['price_vs_sma'] = df['close'] / df['sma_20']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Lag features
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
        
        # Target: future return (next 5 minutes)
        df['target'] = df['close'].shift(-5) / df['close'] - 1
        
        return df.dropna()
    
    def train(self, historical_data: pd.DataFrame) -> None:
        """Train ensemble of models"""
        logger.info("Training ML models...")
        
        df_features = self.create_features(historical_data)
        
        # Prepare features and target
        feature_cols = [col for col in df_features.columns if col not in 
                       ['timestamp', 'symbol', 'target', 'close']]
        X = df_features[feature_cols].values
        y = df_features['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for name, model in self.models.items():
            scores = []
            for train_idx, val_idx in tscv.split(X_scaled):
                X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
                scores.append(score)
            
            avg_score = np.mean(scores)
            logger.info(f"{name.upper()} model trained. Avg R²: {avg_score:.4f}")
        
        self.is_trained = True
        logger.info("All models trained successfully")
    
    def predict(self, current_data: pd.DataFrame) -> Dict:
        """Make ensemble prediction"""
        if not self.is_trained:
            return {'signal': TradeSignal.HOLD, 'confidence': 0}
        
        df_features = self.create_features(current_data)
        feature_cols = [col for col in df_features.columns if col not in 
                       ['timestamp', 'symbol', 'target', 'close']]
        X = df_features[feature_cols].iloc[-1:].values
        X_scaled = self.scaler.transform(X)
        
        # Ensemble predictions
        predictions = []
        weights = {'xgb': 0.5, 'rf': 0.3, 'gb': 0.2}
        
        for name, model in self.models.items():
            pred = model.predict(X_scaled)[0]
            predictions.append(pred * weights[name])
        
        ensemble_pred = np.sum(predictions)
        
        # Convert prediction to trading signal
        if ensemble_pred > 0.005:
            signal = TradeSignal.STRONG_BUY
        elif ensemble_pred > 0.001:
            signal = TradeSignal.BUY
        elif ensemble_pred < -0.005:
            signal = TradeSignal.STRONG_SELL
        elif ensemble_pred < -0.001:
            signal = TradeSignal.SELL
        else:
            signal = TradeSignal.HOLD
        
        confidence = min(abs(ensemble_pred) * 100, 100)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'predicted_return': ensemble_pred,
            'individual_predictions': {name: pred for name, pred in 
                                      zip(self.models.keys(), predictions)}
        }

class TradingBot:
    """Main trading bot class"""
    
    def __init__(self, api_key: str, api_secret: str, symbols: List[str]):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols
        self.portfolio = PortfolioManager()
        self.predictor = MLPredictor()
        self.session = None
        self.is_running = False
        self.orders: Dict[str, Order] = {}
        
        # Performance metrics
        self.metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }
        
        # Cache for market data
        self.market_data_cache: Dict[str, List[MarketData]] = {
            symbol: [] for symbol in symbols
        }
    
    async def initialize(self):
        """Initialize bot components"""
        logger.info("Initializing Trading Bot...")
        self.session = aiohttp.ClientSession()
        
        # Load historical data for training
        historical_data = await self.fetch_historical_data(days=30)
        self.predictor.train(historical_data)
        
        logger.info("Trading Bot initialized successfully")
    
    async def fetch_historical_data(self, days: int = 30) -> pd.DataFrame:
        """Fetch historical market data"""
        # In production, this would fetch from exchange API
        # Here we simulate with random data
        dates = pd.date_range(end=datetime.utcnow(), periods=days*24*60, freq='1min')
        
        data = []
        for symbol in self.symbols:
            # Generate realistic price patterns
            price = 100 + np.random.randn(len(dates)).cumsum() * 0.1
            volume = np.random.lognormal(mean=10, sigma=1, size=len(dates))
            
            for i, date in enumerate(dates):
                open_price = price[i]
                close_price = price[i] + np.random.randn() * 0.1
                high = max(open_price, close_price) + abs(np.random.randn() * 0.05)
                low = min(open_price, close_price) - abs(np.random.randn() * 0.05)
                
                data.append({
                    'timestamp': date,
                    'symbol': symbol,
                    'open': open_price,
                    'high': high,
                    'low': low,
                    'close': close_price,
                    'volume': volume[i],
                    'vwap': (high + low + close_price) / 3,
                    'trades': int(volume[i] / 100)
                })
        
        return pd.DataFrame(data)
    
    async def stream_market_data(self, symbol: str):
        """Stream real-time market data"""
        # In production, this would connect to WebSocket
        # Simulated for demonstration
        while self.is_running:
            try:
                # Simulate receiving market data
                await asyncio.sleep(1)  # 1 second updates
                
                # Generate simulated tick data
                tick = MarketData(
                    timestamp=datetime.utcnow(),
                    symbol=symbol,
                    open=100 + np.random.randn() * 0.1,
                    high=100.2 + np.random.randn() * 0.1,
                    low=99.8 + np.random.randn() * 0.1,
                    close=100 + np.random.randn() * 0.1,
                    volume=np.random.lognormal(10, 1),
                    vwap=100 + np.random.randn() * 0.05,
                    trades=np.random.randint(100, 1000)
                )
                
                # Cache latest 1000 data points
                self.market_data_cache[symbol].append(tick)
                if len(self.market_data_cache[symbol]) > 1000:
                    self.market_data_cache[symbol].pop(0)
                
                # Process tick
                await self.process_tick(tick)
                
            except Exception as e:
                logger.error(f"Error in market data stream for {symbol}: {e}")
                await asyncio.sleep(5)
    
    async def process_tick(self, tick: MarketData):
        """Process incoming market data tick"""
        try:
            # Convert to DataFrame for ML prediction
            df = pd.DataFrame([{
                'timestamp': tick.timestamp,
                'symbol': tick.symbol,
                'open': tick.open,
                'high': tick.high,
                'low': tick.low,
                'close': tick.close,
                'volume': tick.volume
            }])
            
            # Get ML prediction
            prediction = self.predictor.predict(df)
            
            # Execute trade based on signal
            if prediction['signal'] != TradeSignal.HOLD:
                await self.execute_trade(
                    symbol=tick.symbol,
                    signal=prediction['signal'],
                    current_price=tick.close,
                    confidence=prediction['confidence']
                )
            
            # Update metrics
            self.update_metrics()
            
            # Log periodically
            if np.random.random() < 0.01:  # Log 1% of ticks
                logger.info(
                    f"Tick: {tick.symbol} | "
                    f"Price: ${tick.close:.2f} | "
                    f"Signal: {prediction['signal'].name} | "
                    f"Confidence: {prediction['confidence']:.1f}%"
                )
                
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
    
    async def execute_trade(self, symbol: str, signal: TradeSignal, 
                          current_price: float, confidence: float):
        """Execute trade based on ML signal"""
        try:
            # Calculate position size with risk management
            stop_loss = current_price * (0.98 if signal.value > 0 else 1.02)
            position_size, position_value = self.portfolio.calculate_position_size(
                symbol, current_price, stop_loss
            )
            
            if position_size <= 0:
                return
            
            # Determine order type
            order_type = OrderType.LIMIT if confidence > 70 else OrderType.MARKET
            
            # Create order
            side = 'buy' if signal.value > 0 else 'sell'
            order = Order(
                symbol=symbol,
                order_type=order_type,
                side=side,
                quantity=position_size,
                price=current_price if order_type == OrderType.LIMIT else None
            )
            
            # Simulate order execution (in production, would call exchange API)
            logger.info(
                f"Executing {side.upper()} order: {symbol} | "
                f"Size: {position_size:.4f} | "
                f"Price: ${current_price:.2f} | "
                f"Value: ${position_value:.2f} | "
                f"Confidence: {confidence:.1f}%"
            )
            
            # Update portfolio
            self.portfolio.update_portfolio(symbol, position_size, current_price, side)
            
            # Track order
            self.orders[order.order_id] = order
            self.metrics['total_trades'] += 1
            
            # Simulate P&L update after some time
            asyncio.create_task(self.update_trade_pnl(order, current_price))
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
    
    async def update_trade_pnl(self, order: Order, entry_price: float):
        """Simulate P&L update for a trade"""
        await asyncio.sleep(60)  # Wait 1 minute
        
        # Simulate price movement
        exit_price = entry_price * (1 + np.random.randn() * 0.02)
        pnl = (exit_price - entry_price) * order.quantity * (
            1 if order.side == 'buy' else -1
        )
        
        self.metrics['total_pnl'] += pnl
        
        if pnl > 0:
            self.metrics['winning_trades'] += 1
        else:
            self.metrics['losing_trades'] += 1
        
        # Update max drawdown
        current_value = self.portfolio.current_capital
        peak_value = max(self.metrics.get('peak_capital', 0), current_value)
        drawdown = (peak_value - current_value) / peak_value if peak_value > 0 else 0
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], drawdown)
        self.metrics['peak_capital'] = peak_value
    
    def update_metrics(self):
        """Update performance metrics"""
        # Calculate Sharpe ratio from recent returns
        if len(self.portfolio.trade_history) >= 10:
            returns = pd.Series([
                trade['value'] for trade in self.portfolio.trade_history[-100:]
            ]).pct_change().dropna()
            
            if len(returns) >= 2:
                self.metrics['sharpe_ratio'] = self.portfolio.calculate_sharpe_ratio(returns)
    
    async def monitor_performance(self):
        """Monitor bot performance and health"""
        while self.is_running:
            try:
                # Log performance metrics
                win_rate = (self.metrics['winning_trades'] / 
                          max(self.metrics['total_trades'], 1)) * 100
                
                logger.info(
                    "Performance Metrics:\n"
                    f"  Total Trades: {self.metrics['total_trades']}\n"
                    f"  Win Rate: {win_rate:.1f}%\n"
                    f"  Total P&L: ${self.metrics['total_pnl']:.2f}\n"
                    f"  Current Capital: ${self.portfolio.current_capital:.2f}\n"
                    f"  Max Drawdown: {self.metrics['max_drawdown']*100:.1f}%\n"
                    f"  Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}\n"
                    f"  Active Positions: {len(self.portfolio.positions)}"
                )
                
                # Memory usage
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"Memory Usage: {memory_mb:.1f} MB")
                
                # Run garbage collection if memory high
                if memory_mb > 500:
                    gc.collect()
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error in performance monitor: {e}")
                await asyncio.sleep(10)
    
    async def run(self):
        """Main bot execution loop"""
        logger.info("Starting Trading Bot...")
        self.is_running = True
        
        try:
            await self.initialize()
            
            # Start market data streams
            stream_tasks = [
                self.stream_market_data(symbol)
                for symbol in self.symbols
            ]
            
            # Start performance monitor
            monitor_task = asyncio.create_task(self.monitor_performance())
            
            # Run all tasks
            await asyncio.gather(*stream_tasks, monitor_task)
            
        except KeyboardInterrupt:
            logger.info("Shutdown signal received")
        except Exception as e:
            logger.error(f"Bot crashed: {e}")
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Trading Bot...")
        self.is_running = False
        
        if self.session:
            await self.session.close()
        
        # Save final metrics
        self.save_metrics()
        logger.info("Trading Bot shutdown complete")
    
    def save_metrics(self):
        """Save metrics to file"""
        metrics_file = 'trading_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"Metrics saved to {metrics_file}")
    
    @profile
    def memory_intensive_operation(self, data: np.ndarray):
        """Example of memory-intensive operation with profiling"""
        # Large matrix operations
        result = np.linalg.eigvals(data @ data.T)
        return result

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    sys.exit(0)

async def main():
    """Main entry point"""
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Configuration
    API_KEY = "your_api_key_here"  # In production, use environment variables
    API_SECRET = "your_api_secret_here"
    SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
    
    # Create and run bot
    bot = TradingBot(API_KEY, API_SECRET, SYMBOLS)
    
    try:
        await bot.run()
    except Exception as e:
        logger.critical(f"Fatal error in main: {e}")
        raise

if __name__ == "__main__":
    # Run with asyncio
    asyncio.run(main())        risk_amount = self.current_capital * self.risk_per_trade
        price_risk = abs(entry_price - stop_loss)
        
        self.sharp_target = 1.5
        
    def calculate_position_size(self, symbol: str, entry_price: float, 
        self.trade_history: List[Dict] = []
        self.risk_per_trade = 0.02  # 2% risk per trade
        self.max_portfolio_risk = 0.15  # 15% max portfolio risk
    vwap: float
    trades: int

    open: float
    high: float
    low: float
    STRONG_BUY = 2
    BUY = 1
    timestamp: datetime
    HOLD = 0
@dataclass
class MarketData:

