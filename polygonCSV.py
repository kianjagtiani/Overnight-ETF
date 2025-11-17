import requests
import pandas as pd
import time
from datetime import datetime, timedelta

class PolygonFetcher:
    """
    Fetch historical data from Polygon.io
    Free tier: 5 calls/minute
    Get free API key at: https://polygon.io/
    """
    
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def fetch_aggregates(self, ticker, multiplier, timespan, from_date, to_date):
        """
        Fetch aggregate bars (OHLCV) for a date range.
        
        Args:
            ticker: Stock ticker (e.g., 'SPY')
            multiplier: Size of timespan (e.g., 5)
            timespan: 'minute', 'hour', 'day'
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
        
        Returns:
            DataFrame with OHLCV data
        """
        url = f"{self.base_url}/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        
        params = {
            'adjusted': 'true',
            'sort': 'asc',
            'limit': 50000,  # Max per request
            'apiKey': self.api_key
        }
        
        print(f"Fetching {ticker} {multiplier}{timespan} from {from_date} to {to_date}...")
        
        all_results = []
        
        while True:
            response = requests.get(url, params=params)
            
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return None
            
            data = response.json()
            
            if 'results' not in data or len(data['results']) == 0:
                break
            
            all_results.extend(data['results'])
            print(f"  Fetched {len(data['results'])} bars (total: {len(all_results)})")
            
            # Check if there's more data
            if 'next_url' not in data:
                break
            
            # Update URL for next page
            url = data['next_url']
            params = {'apiKey': self.api_key}  # next_url already has other params
            
            # Rate limiting for free tier
            time.sleep(12)  # 5 calls/min = one call every 12s
        
        if not all_results:
            print("No data returned")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(all_results)
        
        # Convert timestamp (ms) to datetime
        df['timestamp'] = pd.to_datetime(df['t'], unit='ms')
        df = df.set_index('timestamp')
        
        # Rename columns to standard format
        df = df.rename(columns={
            'o': 'Open',
            'h': 'High',
            'l': 'Low',
            'c': 'Close',
            'v': 'Volume'
        })
        
        # Select relevant columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        return df
    
    def fetch_date_range_chunked(self, ticker, multiplier, timespan, 
                                  start_date, end_date, chunk_days=30):
        """
        Fetch data in chunks to avoid timeout and respect rate limits.
        
        Args:
            chunk_days: Number of days per API call (smaller = more calls but more reliable)
        """
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        all_data = []
        current = start
        
        while current < end:
            chunk_end = min(current + timedelta(days=chunk_days), end)
            
            df = self.fetch_aggregates(
                ticker, multiplier, timespan,
                current.strftime('%Y-%m-%d'),
                chunk_end.strftime('%Y-%m-%d')
            )
            
            if df is not None and len(df) > 0:
                all_data.append(df)
            
            current = chunk_end + timedelta(days=1)
            
            # Rate limit between chunks (5 calls/min)
            if current < end:
                time.sleep(12)
        
        if all_data:
            combined = pd.concat(all_data).sort_index()
            # Remove duplicates
            combined = combined[~combined.index.duplicated(keep='first')]
            print(f"\n✓ Total bars: {len(combined)}")
            print(f"✓ Date range: {combined.index[0]} to {combined.index[-1]}")
            return combined
        
        return None
    
    def save_to_csv(self, df, filename):
        """Save data to CSV"""
        df.to_csv(filename)
        print(f"Saved to {filename}")
    
    def load_from_csv(self, filename):
        """Load saved data"""
        return pd.read_csv(filename, index_col=0, parse_dates=True)


if __name__ == "__main__":
    API_KEY = "Q3bZ5_nC772YLlrcIpSURxm4gDhD8aOr"
    
    fetcher = PolygonFetcher(API_KEY)
    
    # List of symbols to download
    symbols_to_fetch = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'GLD']  # Add whatever you want
    
    # Calculate date range (e.g., last 6 months)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    for symbol in symbols_to_fetch:
        print(f"\n{'='*60}")
        print(f"Downloading {symbol}...")
        print(f"{'='*60}")
        
        df = fetcher.fetch_date_range_chunked(
            ticker=symbol,
            multiplier=5,
            timespan='minute',
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d'),
            chunk_days=30
        )
        
        if df is not None:
            # Filter to market hours
            df_market_hours = df.between_time('09:30', '16:00')
            
            # Save with symbol name in filename
            filename = f'{symbol.lower()}_5min_polygon_6months.csv'
            fetcher.save_to_csv(df_market_hours, filename)
            print(f"✓ Saved {symbol} data to {filename}")
        else:
            print(f"✗ Failed to fetch {symbol}")
        
        # Wait between symbols to respect rate limits
        print("\nWaiting 60 seconds before next symbol...")
        time.sleep(60)
    
    print("\n" + "="*60)
    print("ALL DOWNLOADS COMPLETE!")
    print("="*60)