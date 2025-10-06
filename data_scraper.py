import os
import time
import pandas as pd
from datetime import datetime
from binance.client import Client
from binance import ThreadedWebsocketManager
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

## Load API keys from environment variables
API_KEY = os.getenv('BINANCE_API_KEY')
API_SECRET = os.getenv('BINANCE_SECRET_KEY')

# Global client variable
client = None

def get_client():
    """Initialize Binance client lazily"""
    global client
    if client is None:
        # Validate that API keys are provided
        if not API_KEY or not API_SECRET:
            raise ValueError("Please set BINANCE_API_KEY and BINANCE_SECRET_KEY in your .env file")
        client = Client(API_KEY, API_SECRET)
    return client

class DataScraper:
    def __init__(self, symbol="BTCUSDT", interval="5m"):
        self.client = None  # Initialize lazily
        self.symbol = symbol
        self.interval = interval
    
    def _get_client(self):
        """Get client, initializing if needed"""
        if self.client is None:
            self.client = get_client()
        return self.client
        self.ws_manager = None
        self.data = []
        self.twm = None
    
#================= Historical Data Fetching =================#
    def fetch_historical_(self, lookback_days="1 day ago UTC"):
       """Fetch historical data for model training."""
       klines = self._get_client().get_historical_klines(self.symbol, self.interval, lookback_days)
       df = pd.DataFrame(klines, columns =[
           "timestamp", "open", "high", "low", "close", "volume", "close_time",
           "quote_asset_volume", "number_of_trades", "taker_buy_base", "taker_buy_quote", "ignore"
       ])
       df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
       df.set_index("timestamp", inplace=True)
       return df[["open", "high", "low", "close", "volume"]].astype(float)
    def save_training_data(self, filename=None, lookback_days="30 days ago UTC"):
        if filename is None:
            filename = f"data/training_{self.symbol}_{self.interval}_{datetime.now().strftime('%Y%m%d')}.csv"
        df = self.fetch_historical_(lookback_days)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        ## SAVE TO CSV FOR NOW DATABASE KO TO LATER.
        df.to_csv(filename)
        print(f"Saved historical data to {filename}")

#================= Real-time Data Fetching =================#
    def start_stream(self, callback):
        """Start websocket to fetch real-time data."""
        self.twm = ThreadedWebsocketManager(API_KEY, API_SECRET)
        self.twm.start()
        self.twm.start_symbol_ticker_socket(callback=callback, symbol=self.symbol)

        print("Websocket started for real-time data.")
    def stop_stream(self):
        """STOP STREAM."""
        if self.twm:
            self.twm.stop()
            print("Websocket stopped.")

    ## Order book depth to get liquidity context
    def fetch_order_book_context(self, depth=20):
        order_book = self._get_client().get_order_book(symbol=self.symbol, limit=depth)
        bids = pd.DataFrame(order_book['bids'], columns=['price', 'quantity']).astype(float)
        asks = pd.DataFrame(order_book['asks'], columns=['price', 'quantity']).astype(float)

        context = {
            "bid_volume": bids['quantity'].sum(),
            "ask_volume": asks['quantity'].sum(),
            "spread": asks['price'].min() - bids['price'].max()

        }
        return context

    ## 24h ticker for volatility context
    def fetch_24h_ticker_context(self):
        ticker = self._get_client().get_ticker(symbol=self.symbol)
        context = {
            "price_change_percent" : float(ticker['priceChangePercent']),
            "high_price": float(ticker['highPrice']),
            "low_price": float(ticker['lowPrice']),
            "volume": float(ticker['volume'])
        }
        return context
    

    
   
    def fetch_context_data(self):
        """Fetch recent data for context in feature generation."""
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = end_time - (1000 * 60 * 60 * 24)  # Last 24 hours
        klines = self._get_client().get_klines(symbol=self.symbol, interval=self.interval, startTime=start_time, endTime=end_time)
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time',
            'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'])
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df.set_index('open_time', inplace=True)
        df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        ##|| MERGE WITH OTHER CONTEXT DATA ||##
        order_book_context = self.fetch_order_book_context()
        ticker_context = self.fetch_24h_ticker_context()

        context = {
            "order_book": order_book_context,
            "ticker": ticker_context
        }
        return df, context
    ### |||| LATER IN FEATURE ENGINEERING: Merge context into training features 
                ### - spread/close -> liquidity
                ### - bid_volume/ask_volume -> imbalance feature
                ### - price_change_percent -> momentum