"""
Stock Pattern Analyzer - Professional Edition
Author: [Your Name]
Version: 2.2 - Fixed Index Bounds Issues
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pc  # Correct import for colors
import requests
import io
from scipy.signal import find_peaks, savgol_filter
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
class Config:
    """Configuration parameters for the application"""
    CACHE_TTL_STOCK_LIST = 3600  # 1 hour
    CACHE_TTL_STOCK_DATA = 600   # 10 minutes
    MIN_DATA_POINTS = 100        # Minimum data points for analysis
    MAX_WORKERS = 10              # Max parallel workers
    
    # Pattern detection parameters (adaptive to volatility)
    PEAK_PROMINENCE_FACTOR = 0.5
    SHOULDER_SYMMETRY_THRESHOLD = 0.15
    RECTANGLE_VARIANCE_THRESHOLD = 0.03
    FLAG_MIN_POLE_MOVE = 0.10
    
    # Risk parameters
    MIN_PATTERN_CONFIDENCE = 0.6
    
    # NSE URLs
    NIFTY50_URL = "https://archives.nseindia.com/content/indices/ind_nifty50list.csv"
    NIFTY500_URL = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"

# --- Page Configuration ---
st.set_page_config(
    page_title="Professional Stock Pattern Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Utility Functions ---
def color_to_rgba(color_name: str, alpha: float = 1.0) -> str:
    """Convert color name to rgba string"""
    color_map = {
        'red': (255, 0, 0),
        'green': (0, 255, 0),
        'blue': (0, 0, 255),
        'orange': (255, 165, 0),
        'yellow': (255, 255, 0),
        'purple': (128, 0, 128),
        'lightgreen': (144, 238, 144),
        'lightcoral': (240, 128, 128),
        'crimson': (220, 20, 60)
    }
    
    if color_name.lower() in color_map:
        r, g, b = color_map[color_name.lower()]
    else:
        # Default to gray if color not found
        r, g, b = 128, 128, 128
    
    return f"rgba({r}, {g}, {b}, {alpha})"

def safe_index_access(df: pd.DataFrame, idx: int) -> Any:
    """Safely access dataframe index with bounds checking"""
    if idx < 0:
        return df.index[0]
    elif idx >= len(df):
        return df.index[-1]
    else:
        return df.index[idx]

def safe_value_access(arr: np.ndarray, idx: int) -> Any:
    """Safely access array value with bounds checking"""
    if idx < 0:
        return arr[0]
    elif idx >= len(arr):
        return arr[-1]
    else:
        return arr[idx]

# --- Data Management ---
class DataManager:
    """Handles all data fetching and caching operations"""
    
    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL_STOCK_LIST)
    def get_stock_list(universe: str) -> List[str]:
        """Fetches the list of stock tickers for the selected universe."""
        url_map = {
            "Nifty 50": Config.NIFTY50_URL,
            "Nifty 500": Config.NIFTY500_URL
        }
        
        url = url_map.get(universe)
        if not url:
            logger.error(f"Unknown universe: {universe}")
            return []
        
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            csv_file = io.StringIO(response.text)
            df = pd.read_csv(csv_file)
            
            if 'Symbol' not in df.columns:
                logger.error("Symbol column not found in CSV")
                return []
            
            # Validate and clean symbols
            symbols = [f"{symbol.strip()}.NS" for symbol in df['Symbol'] if symbol and isinstance(symbol, str)]
            logger.info(f"Successfully fetched {len(symbols)} symbols for {universe}")
            return symbols
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch stock list for {universe}: {e}")
            st.error(f"Failed to fetch stock list for {universe}: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching stock list: {e}")
            return []

    @staticmethod
    @st.cache_data(ttl=Config.CACHE_TTL_STOCK_DATA)
    def get_stock_data(ticker: str, period: str, interval: str) -> Optional[pd.DataFrame]:
        """Retrieves and preprocesses stock data from Yahoo Finance."""
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, interval=interval, auto_adjust=True)
            
            if data.empty or len(data) < Config.MIN_DATA_POINTS:
                return None
            
            # Data preprocessing
            data = DataManager._preprocess_data(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Could not retrieve data for {ticker}: {e}")
            return None
    
    @staticmethod
    def _preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """Preprocesses the data: handles NaN, outliers, and adds technical indicators"""
        # Handle NaN values
        df = df.dropna()
        
        # Remove outliers using IQR method
        for col in ['Open', 'High', 'Low', 'Close']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        # Add technical indicators
        df['SMA_20'] = df['Close'].rolling(window=20, min_periods=1).mean()
        df['SMA_50'] = df['Close'].rolling(window=50, min_periods=1).mean()
        df['Volatility'] = df['Close'].pct_change().rolling(window=20).std()
        df['Volume_SMA'] = df['Volume'].rolling(window=20, min_periods=1).mean()
        
        # Add price change metrics
        df['Returns'] = df['Close'].pct_change()
        
        return df

# --- Pattern Detection Engine ---
class PatternDetector:
    """Professional pattern detection algorithms with confidence scoring"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.price = df['Close'].values
        self.volatility = df['Volatility'].mean() if 'Volatility' in df.columns else self.price.std() / self.price.mean()
        self.adaptive_prominence = self._calculate_adaptive_prominence()
    
    def _calculate_adaptive_prominence(self) -> float:
        """Calculate prominence based on recent volatility"""
        return self.price.std() * Config.PEAK_PROMINENCE_FACTOR * (1 + self.volatility)
    
    def detect_all_patterns(self) -> Dict[str, Any]:
        """Detect all patterns and return results with confidence scores"""
        patterns = {}
        
        # Detect each pattern type
        pattern_detectors = [
            ("H&S", self.detect_head_and_shoulders, {'is_inverse': False}),
            ("Inverse H&S", self.detect_head_and_shoulders, {'is_inverse': True}),
            ("Rectangle", self.detect_rectangle, {}),
            ("Bull Flag", self.detect_flag, {'is_bullish': True}),
            ("Bear Flag", self.detect_flag, {'is_bullish': False}),
            ("Double Top", self.detect_double_top, {}),
            ("Double Bottom", self.detect_double_bottom, {}),
        ]
        
        for pattern_name, detector, kwargs in pattern_detectors:
            try:
                result = detector(**kwargs)
                if result and result.get('confidence', 0) >= Config.MIN_PATTERN_CONFIDENCE:
                    patterns[pattern_name] = result
            except Exception as e:
                logger.error(f"Error detecting {pattern_name}: {e}")
        
        return patterns
    
    def detect_head_and_shoulders(self, is_inverse: bool = False) -> Optional[Dict]:
        """
        Enhanced Head and Shoulders detection with confidence scoring
        """
        try:
            # Smooth the price data to reduce noise
            window_length = min(11, len(self.price))
            if window_length % 2 == 0:
                window_length += 1
            
            if window_length < 5:
                return None
                
            smoothed_price = savgol_filter(self.price, window_length=window_length, polyorder=min(3, window_length-2))
            
            # Find peaks and troughs with adaptive parameters
            distance = max(5, int(len(self.price) * 0.02))
            peaks, peak_props = find_peaks(smoothed_price, distance=distance, prominence=self.adaptive_prominence)
            troughs, trough_props = find_peaks(-smoothed_price, distance=distance, prominence=self.adaptive_prominence)
            
            if is_inverse:
                points, shoulders = troughs, peaks
            else:
                points, shoulders = peaks, troughs
            
            if len(points) < 3:
                return None
            
            best_pattern = None
            best_confidence = 0
            
            # Sliding window to find the best pattern
            for i in range(len(points) - 2):
                for j in range(i + 2, min(i + 5, len(points))):
                    if j >= len(points):
                        continue
                        
                    p1_idx, p2_idx, p3_idx = points[i], points[(i+j)//2], points[j]
                    
                    # Ensure indices are within bounds
                    if p3_idx >= len(self.price):
                        continue
                    
                    # Check if middle point is the head
                    is_head = (self.price[p2_idx] > max(self.price[p1_idx], self.price[p3_idx])) if not is_inverse else \
                              (self.price[p2_idx] < min(self.price[p1_idx], self.price[p3_idx]))
                    
                    if not is_head:
                        continue
                    
                    # Calculate pattern quality metrics
                    shoulder_symmetry = abs(self.price[p1_idx] - self.price[p3_idx]) / abs(self.price[p2_idx])
                    
                    if shoulder_symmetry > Config.SHOULDER_SYMMETRY_THRESHOLD:
                        continue
                    
                    # Find neckline points
                    neckline_points = [p for p in shoulders if p1_idx < p < p3_idx]
                    
                    if len(neckline_points) < 2:
                        continue
                    
                    # Calculate confidence score
                    confidence = self._calculate_pattern_confidence(
                        shoulder_symmetry=shoulder_symmetry,
                        time_symmetry=abs((p2_idx - p1_idx) - (p3_idx - p2_idx)) / (p3_idx - p1_idx),
                        volume_confirmation=self._check_volume_pattern(p1_idx, p2_idx, p3_idx)
                    )
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_pattern = {
                            "L-Shoulder": p1_idx,
                            "Head": p2_idx,
                            "R-Shoulder": p3_idx,
                            "Neckline1": neckline_points[0],
                            "Neckline2": neckline_points[-1],
                            "confidence": confidence,
                            "pattern_type": "Inverse H&S" if is_inverse else "H&S"
                        }
            
            return best_pattern
            
        except Exception as e:
            logger.error(f"Error in H&S detection: {e}")
            return None
    
    def detect_rectangle(self) -> Optional[Dict]:
        """Enhanced rectangle pattern detection with proper bounds checking"""
        try:
            window_size = min(60, len(self.price) // 3)
            
            for start_idx in range(len(self.price) - window_size, max(0, len(self.price) - window_size * 3), -10):
                # Ensure end_idx doesn't exceed bounds
                end_idx = min(start_idx + window_size, len(self.price) - 1)
                
                if end_idx <= start_idx:
                    continue
                
                window_price = self.price[start_idx:end_idx]
                
                if len(window_price) < 10:  # Need minimum points for pattern
                    continue
                
                # Calculate support and resistance levels using percentiles
                resistance_level = np.percentile(window_price, 90)
                support_level = np.percentile(window_price, 10)
                
                # Check if price mostly stays within the range
                within_range = np.sum((window_price >= support_level * 0.98) & 
                                     (window_price <= resistance_level * 1.02)) / len(window_price)
                
                if within_range < 0.8:
                    continue
                
                # Check for minimum touches
                resistance_touches = np.sum(window_price > resistance_level * 0.98)
                support_touches = np.sum(window_price < support_level * 1.02)
                
                if resistance_touches < 2 or support_touches < 2:
                    continue
                
                # Calculate pattern quality
                channel_width = (resistance_level - support_level) / support_level
                
                if 0.03 < channel_width < 0.20:
                    confidence = min(0.9, within_range * 0.8 + min(resistance_touches, support_touches) * 0.05)
                    
                    return {
                        "start_idx": start_idx,
                        "end_idx": end_idx,
                        "resistance": resistance_level,
                        "support": support_level,
                        "confidence": confidence,
                        "touches": {"resistance": resistance_touches, "support": support_touches}
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in rectangle detection: {e}")
            return None
    
    def detect_flag(self, is_bullish: bool = True) -> Optional[Dict]:
        """Enhanced flag pattern detection with proper bounds checking"""
        try:
            # Look for pole in recent data
            lookback = min(40, len(self.price) // 3)
            
            for pole_start_idx in range(len(self.price) - lookback, max(0, len(self.price) - lookback * 2), -5):
                pole_end_idx = min(pole_start_idx + 20, len(self.price) - 1)
                
                if pole_end_idx <= pole_start_idx:
                    continue
                
                pole_price = self.price[pole_start_idx:pole_end_idx]
                
                if len(pole_price) < 5:
                    continue
                
                # Find the pole extremum
                if is_bullish:
                    pole_peak_idx_rel = np.argmax(pole_price)
                else:
                    pole_peak_idx_rel = np.argmin(pole_price)
                
                pole_peak_idx = pole_start_idx + pole_peak_idx_rel
                
                # Check pole strength
                pole_move = abs(self.price[pole_peak_idx] - self.price[pole_start_idx]) / self.price[pole_start_idx]
                
                if pole_move < Config.FLAG_MIN_POLE_MOVE:
                    continue
                
                # Analyze flag formation
                flag_start = pole_peak_idx
                flag_end = min(flag_start + 20, len(self.price) - 1)
                
                if flag_end - flag_start < 5:
                    continue
                
                flag_price = self.price[flag_start:flag_end]
                
                # Check flag characteristics
                flag_slope = np.polyfit(range(len(flag_price)), flag_price, 1)[0]
                flag_direction_correct = (is_bullish and flag_slope < 0) or (not is_bullish and flag_slope > 0)
                
                if not flag_direction_correct:
                    continue
                
                # Check retracement depth
                if is_bullish:
                    retracement = (self.price[pole_peak_idx] - np.min(flag_price)) / (self.price[pole_peak_idx] - self.price[pole_start_idx])
                else:
                    retracement = (np.max(flag_price) - self.price[pole_peak_idx]) / (self.price[pole_start_idx] - self.price[pole_peak_idx])
                
                if 0.2 < retracement < 0.6:  # Ideal retracement between 20% and 60%
                    confidence = 0.7 + (0.3 * (1 - abs(retracement - 0.38)))  # Best at 38.2% Fibonacci
                    
                    return {
                        "pole_start": pole_start_idx,
                        "pole_end": pole_peak_idx,
                        "flag_start": flag_start,
                        "flag_end": flag_end,
                        "confidence": confidence,
                        "retracement": retracement
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in flag detection: {e}")
            return None
    
    def detect_double_top(self) -> Optional[Dict]:
        """Detect double top pattern with bounds checking"""
        try:
            window_length = min(11, len(self.price))
            if window_length % 2 == 0:
                window_length += 1
            
            if window_length < 5:
                return None
                
            smoothed_price = savgol_filter(self.price, window_length=window_length, polyorder=min(3, window_length-2))
            peaks, _ = find_peaks(smoothed_price, distance=10, prominence=self.adaptive_prominence)
            
            if len(peaks) < 2:
                return None
            
            # Look for two peaks of similar height
            for i in range(len(peaks) - 1):
                for j in range(i + 1, min(i + 5, len(peaks))):
                    peak1_idx, peak2_idx = peaks[i], peaks[j]
                    
                    # Ensure indices are within bounds
                    if peak2_idx >= len(self.price):
                        continue
                    
                    peak1_price, peak2_price = self.price[peak1_idx], self.price[peak2_idx]
                    
                    # Check if peaks are similar in height (within 3%)
                    if abs(peak1_price - peak2_price) / max(peak1_price, peak2_price) > 0.03:
                        continue
                    
                    # Find the valley between peaks
                    valley_idx = np.argmin(self.price[peak1_idx:peak2_idx]) + peak1_idx
                    valley_price = self.price[valley_idx]
                    
                    # Valley should be at least 5% below peaks
                    if (min(peak1_price, peak2_price) - valley_price) / valley_price < 0.05:
                        continue
                    
                    confidence = 0.7 + 0.3 * (1 - abs(peak1_price - peak2_price) / max(peak1_price, peak2_price))
                    
                    return {
                        "peak1": peak1_idx,
                        "peak2": peak2_idx,
                        "valley": valley_idx,
                        "confidence": confidence
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in double top detection: {e}")
            return None
    
    def detect_double_bottom(self) -> Optional[Dict]:
        """Detect double bottom pattern with bounds checking"""
        try:
            window_length = min(11, len(self.price))
            if window_length % 2 == 0:
                window_length += 1
            
            if window_length < 5:
                return None
                
            smoothed_price = savgol_filter(self.price, window_length=window_length, polyorder=min(3, window_length-2))
            troughs, _ = find_peaks(-smoothed_price, distance=10, prominence=self.adaptive_prominence)
            
            if len(troughs) < 2:
                return None
            
            # Look for two troughs of similar depth
            for i in range(len(troughs) - 1):
                for j in range(i + 1, min(i + 5, len(troughs))):
                    trough1_idx, trough2_idx = troughs[i], troughs[j]
                    
                    # Ensure indices are within bounds
                    if trough2_idx >= len(self.price):
                        continue
                    
                    trough1_price, trough2_price = self.price[trough1_idx], self.price[trough2_idx]
                    
                    # Check if troughs are similar in depth (within 3%)
                    if abs(trough1_price - trough2_price) / min(trough1_price, trough2_price) > 0.03:
                        continue
                    
                    # Find the peak between troughs
                    peak_idx = np.argmax(self.price[trough1_idx:trough2_idx]) + trough1_idx
                    peak_price = self.price[peak_idx]
                    
                    # Peak should be at least 5% above troughs
                    if (peak_price - max(trough1_price, trough2_price)) / max(trough1_price, trough2_price) < 0.05:
                        continue
                    
                    confidence = 0.7 + 0.3 * (1 - abs(trough1_price - trough2_price) / min(trough1_price, trough2_price))
                    
                    return {
                        "trough1": trough1_idx,
                        "trough2": trough2_idx,
                        "peak": peak_idx,
                        "confidence": confidence
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in double bottom detection: {e}")
            return None
    
    def _calculate_pattern_confidence(self, **metrics) -> float:
        """Calculate overall confidence score for a pattern"""
        weights = {
            'shoulder_symmetry': 0.3,
            'time_symmetry': 0.2,
            'volume_confirmation': 0.5
        }
        
        confidence = 0
        for metric, value in metrics.items():
            if metric in weights:
                # Normalize value to 0-1 range
                normalized_value = 1 - min(value, 1) if metric != 'volume_confirmation' else value
                confidence += weights[metric] * normalized_value
        
        return min(confidence, 1.0)
    
    def _check_volume_pattern(self, idx1: int, idx2: int, idx3: int) -> float:
        """Check if volume confirms the pattern"""
        try:
            if 'Volume' not in self.df.columns:
                return 0.5  # Neutral if no volume data
            
            vol1 = self.df['Volume'].iloc[max(0, idx1-2):min(len(self.df), idx1+3)].mean()
            vol2 = self.df['Volume'].iloc[max(0, idx2-2):min(len(self.df), idx2+3)].mean()
            vol3 = self.df['Volume'].iloc[max(0, idx3-2):min(len(self.df), idx3+3)].mean()
            
            avg_vol = self.df['Volume'].mean()
            
            # Higher volume at pattern points is good
            vol_score = (vol1 + vol2 + vol3) / (3 * avg_vol)
            
            return min(vol_score / 2, 1.0)  # Normalize to 0-1
            
        except Exception:
            return 0.5

# --- Visualization ---
class ChartVisualizer:
    """Professional chart visualization with pattern annotations"""
    
    @staticmethod
    def create_annotated_chart(df: pd.DataFrame, ticker: str, patterns: Dict) -> go.Figure:
        """Creates an annotated chart with detected patterns"""
        
        # Create subplots with different row heights
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{ticker} - Price Action', 'Volume', 'Pattern Confidence'),
            row_heights=[0.6, 0.2, 0.2],
            specs=[[{"secondary_y": True}], [{"secondary_y": False}], [{"secondary_y": False}]]
        )
        
        # Main candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color='green',
                decreasing_line_color='red'
            ),
            row=1, col=1, secondary_y=False
        )
        
        # Add moving averages if available
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                          line=dict(color='blue', width=1)),
                row=1, col=1, secondary_y=False
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                          line=dict(color='orange', width=1)),
                row=1, col=1, secondary_y=False
            )
        
        # Volume bars
        colors = ['red' if df['Close'].iloc[i] < df['Open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color=colors, opacity=0.5),
            row=2, col=1
        )
        
        # Pattern confidence timeline
        confidence_data = ChartVisualizer._create_confidence_timeline(df, patterns)
        if confidence_data:
            fig.add_trace(
                go.Bar(x=list(confidence_data.keys()), y=list(confidence_data.values()),
                      name='Pattern Confidence', marker_color='purple'),
                row=3, col=1
            )
        
        # Annotate patterns
        ChartVisualizer._annotate_patterns(fig, df, patterns)
        
        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{ticker} - Technical Analysis",
                font=dict(size=20)
            ),
            xaxis_rangeslider_visible=False,
            hovermode='x unified',
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=100, b=50),
            height=800,
            template="plotly_dark",
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(showspikes=True, spikemode='across', spikethickness=1)
        fig.update_yaxes(showspikes=True, spikemode='across', spikethickness=1)
        
        return fig
    
    @staticmethod
    def _annotate_patterns(fig: go.Figure, df: pd.DataFrame, patterns: Dict):
        """Add pattern-specific annotations to the chart"""
        
        annotation_colors = {
            "H&S": "red",
            "Inverse H&S": "green",
            "Rectangle": "orange",
            "Bull Flag": "lightgreen",
            "Bear Flag": "lightcoral",
            "Double Top": "red",
            "Double Bottom": "green"
        }
        
        for pattern_name, pattern_data in patterns.items():
            color = annotation_colors.get(pattern_name, "yellow")
            
            if pattern_name in ["H&S", "Inverse H&S"]:
                ChartVisualizer._annotate_head_shoulders(fig, df, pattern_data, color, pattern_name)
            elif pattern_name == "Rectangle":
                ChartVisualizer._annotate_rectangle(fig, df, pattern_data, color)
            elif "Flag" in pattern_name:
                ChartVisualizer._annotate_flag(fig, df, pattern_data, color, pattern_name)
            elif "Double" in pattern_name:
                ChartVisualizer._annotate_double_pattern(fig, df, pattern_data, color, pattern_name)
    
    @staticmethod
    def _annotate_head_shoulders(fig, df, pattern, color, name):
        """Annotate Head and Shoulders pattern with bounds checking"""
        points = [pattern["L-Shoulder"], pattern["Head"], pattern["R-Shoulder"]]
        labels = ["LS", "H", "RS"]
        
        # Ensure indices are within bounds
        valid_points = []
        valid_labels = []
        for idx, (point, label) in enumerate(zip(points, labels)):
            if 0 <= point < len(df):
                valid_points.append(point)
                valid_labels.append(label)
        
        if not valid_points:
            return
        
        # Add pattern points
        fig.add_trace(
            go.Scatter(
                x=df.index[valid_points],
                y=df['Close'].iloc[valid_points],
                mode='markers+text',
                name=f'{name} Points',
                marker=dict(color=color, size=12, symbol='diamond'),
                text=valid_labels,
                textposition="top center",
                textfont=dict(size=10, color=color),
                showlegend=True
            ),
            row=1, col=1
        )
        
        # Add neckline
        neck_points = [pattern["Neckline1"], pattern["Neckline2"]]
        if all(0 <= p < len(df) for p in neck_points):
            fig.add_shape(
                type="line",
                x0=df.index[neck_points[0]],
                y0=df['Close'].iloc[neck_points[0]],
                x1=df.index[neck_points[1]],
                y1=df['Close'].iloc[neck_points[1]],
                line=dict(color=color, width=2, dash="dash"),
                row=1, col=1
            )
        
        # Add confidence annotation
        head_idx = pattern["Head"]
        if 0 <= head_idx < len(df):
            fig.add_annotation(
                x=df.index[head_idx],
                y=df['High'].iloc[head_idx] * 1.02,
                text=f"Confidence: {pattern.get('confidence', 0):.2%}",
                showarrow=False,
                font=dict(size=10, color=color),
                row=1, col=1
            )
    
    @staticmethod
    def _annotate_rectangle(fig, df, pattern, color):
        """Annotate Rectangle pattern with bounds checking"""
        # Ensure indices are within bounds
        start_idx = min(pattern["start_idx"], len(df) - 1)
        end_idx = min(pattern["end_idx"], len(df) - 1)
        
        # Use the color_to_rgba function for proper color handling
        fillcolor = color_to_rgba(color, 0.2)
        
        fig.add_shape(
            type="rect",
            x0=safe_index_access(df, start_idx),
            y0=pattern["support"],
            x1=safe_index_access(df, end_idx),
            y1=pattern["resistance"],
            line=dict(color=color, width=2),
            fillcolor=fillcolor,
            layer="below",
            row=1, col=1
        )
        
        # Add labels
        mid_idx = (start_idx + end_idx) // 2
        if 0 <= mid_idx < len(df):
            fig.add_annotation(
                x=df.index[mid_idx],
                y=pattern["resistance"] * 1.01,
                text=f"Rectangle (Conf: {pattern.get('confidence', 0):.2%})",
                showarrow=False,
                font=dict(size=10, color=color),
                row=1, col=1
            )
    
    @staticmethod
    def _annotate_flag(fig, df, pattern, color, name):
        """Annotate Flag pattern with bounds checking"""
        # Ensure indices are within bounds
        pole_start = min(pattern["pole_start"], len(df) - 1)
        pole_end = min(pattern["pole_end"], len(df) - 1)
        flag_start = min(pattern["flag_start"], len(df) - 1)
        flag_end = min(pattern["flag_end"], len(df) - 1)
        
        # Draw pole
        if 0 <= pole_start < len(df) and 0 <= pole_end < len(df):
            fig.add_shape(
                type="line",
                x0=df.index[pole_start],
                y0=df['Close'].iloc[pole_start],
                x1=df.index[pole_end],
                y1=df['Close'].iloc[pole_end],
                line=dict(color=color, width=3),
                row=1, col=1
            )
        
        # Draw flag
        flag_indices = [i for i in range(flag_start, min(flag_end + 1, len(df)))]
        if flag_indices:
            fig.add_trace(
                go.Scatter(
                    x=df.index[flag_indices],
                    y=df['Close'].iloc[flag_indices],
                    mode='lines',
                    name=name,
                    line=dict(color=color, width=2, dash='dash'),
                    showlegend=True
                ),
                row=1, col=1
            )
        
        # Add annotation
        if 0 <= flag_start < len(df):
            fig.add_annotation(
                x=df.index[flag_start],
                y=df['Close'].iloc[flag_start] * 1.02,
                text=f"{name} (Conf: {pattern.get('confidence', 0):.2%})",
                showarrow=True,
                arrowhead=2,
                font=dict(size=10, color=color),
                row=1, col=1
            )
    
    @staticmethod
    def _annotate_double_pattern(fig, df, pattern, color, name):
        """Annotate Double Top/Bottom pattern with bounds checking"""
        if "Top" in name:
            points = [pattern["peak1"], pattern["valley"], pattern["peak2"]]
            labels = ["P1", "V", "P2"]
        else:
            points = [pattern["trough1"], pattern["peak"], pattern["trough2"]]
            labels = ["T1", "P", "T2"]
        
        # Ensure indices are within bounds
        valid_points = []
        valid_labels = []
        for point, label in zip(points, labels):
            if 0 <= point < len(df):
                valid_points.append(point)
                valid_labels.append(label)
        
        if not valid_points:
            return
        
        fig.add_trace(
            go.Scatter(
                x=df.index[valid_points],
                y=df['Close'].iloc[valid_points],
                mode='markers+lines+text',
                name=name,
                marker=dict(color=color, size=10),
                line=dict(color=color, width=1, dash='dot'),
                text=valid_labels,
                textposition="top center",
                showlegend=True
            ),
            row=1, col=1
        )
    
    @staticmethod
    def _create_confidence_timeline(df: pd.DataFrame, patterns: Dict) -> Dict:
        """Create confidence timeline for patterns"""
        confidence_data = {}
        for pattern_name, pattern_data in patterns.items():
            if 'confidence' in pattern_data:
                confidence_data[pattern_name] = pattern_data['confidence']
        return confidence_data

# --- Main Analysis Engine ---
class StockAnalyzer:
    """Main analysis engine with parallel processing"""
    
    def __init__(self, universe: str, timeframe: str):
        self.universe = universe
        self.timeframe = timeframe
        self.period, self.interval = self._map_timeframe(timeframe)
        self.results = []
        
    def _map_timeframe(self, timeframe: str) -> Tuple[str, str]:
        """Map user selection to yfinance parameters"""
        mappings = {
            "1 Day": ("6mo", "1d"),
            "1 Hour": ("60d", "1h"),
            "4 Hour": ("60d", "1h"),  # Will resample
            "1 Week": ("2y", "1wk")
        }
        return mappings.get(timeframe, ("6mo", "1d"))
    
    def analyze_single_stock(self, ticker: str) -> Optional[Dict]:
        """Analyze a single stock for patterns"""
        try:
            df = DataManager.get_stock_data(ticker, self.period, self.interval)
            
            if df is None or len(df) < Config.MIN_DATA_POINTS:
                return None
            
            detector = PatternDetector(df)
            patterns = detector.detect_all_patterns()
            
            if patterns:
                return {
                    'ticker': ticker,
                    'patterns': patterns,
                    'data': df,
                    'summary': self._create_summary(patterns)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {e}")
            return None
    
    def analyze_universe(self, progress_callback=None) -> List[Dict]:
        """Analyze all stocks in the universe with parallel processing"""
        stock_list = DataManager.get_stock_list(self.universe)
        
        if not stock_list:
            return []
        
        results = []
        total_stocks = len(stock_list)
        
        with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
            futures = {executor.submit(self.analyze_single_stock, ticker): ticker 
                      for ticker in stock_list}
            
            completed = 0
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    logger.error(f"Error processing future: {e}")
                
                completed += 1
                if progress_callback:
                    progress_callback(completed / total_stocks)
        
        # Sort by pattern confidence
        results.sort(key=lambda x: max([p.get('confidence', 0) 
                                       for p in x['patterns'].values()]), reverse=True)
        
        return results
    
    def _create_summary(self, patterns: Dict) -> Dict:
        """Create a summary of detected patterns"""
        bullish_patterns = ["Inverse H&S", "Bull Flag", "Double Bottom"]
        bearish_patterns = ["H&S", "Bear Flag", "Double Top"]
        
        bullish_count = sum(1 for p in patterns if p in bullish_patterns)
        bearish_count = sum(1 for p in patterns if p in bearish_patterns)
        
        avg_confidence = np.mean([p.get('confidence', 0) for p in patterns.values()])
        
        return {
            'bullish_count': bullish_count,
            'bearish_count': bearish_count,
            'neutral_count': len(patterns) - bullish_count - bearish_count,
            'avg_confidence': avg_confidence,
            'bias': 'Bullish' if bullish_count > bearish_count else 'Bearish' if bearish_count > bullish_count else 'Neutral'
        }

# --- Streamlit UI ---
def main():
    st.title("📊 Professional Stock Pattern Analyzer")
    st.markdown("*Advanced pattern detection with confidence scoring and risk metrics*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            selected_universe = st.selectbox(
                "Universe",
                ["Nifty 50", "Nifty 500"],
                help="Select the stock universe to analyze"
            )
        
        with col2:
            selected_timeframe = st.selectbox(
                "Timeframe",
                ["1 Day", "1 Hour", "4 Hour", "1 Week"],
                help="Select the timeframe for analysis"
            )
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            min_confidence = st.slider(
                "Minimum Pattern Confidence",
                min_value=0.5,
                max_value=0.95,
                value=Config.MIN_PATTERN_CONFIDENCE,
                step=0.05,
                help="Minimum confidence threshold for pattern detection"
            )
            Config.MIN_PATTERN_CONFIDENCE = min_confidence
            
            max_workers = st.slider(
                "Parallel Workers",
                min_value=1,
                max_value=20,
                value=Config.MAX_WORKERS,
                help="Number of parallel workers for faster processing"
            )
            Config.MAX_WORKERS = max_workers
        
        st.markdown("---")
        
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        # Add information
        st.info("""
        **Pattern Types Detected:**
        - Head & Shoulders (Bearish)
        - Inverse Head & Shoulders (Bullish)
        - Rectangle (Neutral)
        - Bull/Bear Flags
        - Double Top/Bottom
        """)
    
    # Main content area
    if run_button:
        analyzer = StockAnalyzer(selected_universe, selected_timeframe)
        
        with st.spinner(f"Analyzing {selected_universe} stocks..."):
            progress_bar = st.progress(0)
            
            def update_progress(value):
                progress_bar.progress(value)
            
            results = analyzer.analyze_universe(progress_callback=update_progress)
            progress_bar.empty()
        
        if results:
            st.success(f"✅ Analysis complete! Found patterns in {len(results)} stocks.")
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            total_patterns = sum(len(r['patterns']) for r in results)
            avg_confidence = np.mean([max(p.get('confidence', 0) for p in r['patterns'].values()) 
                                     for r in results])
            
            with col1:
                st.metric("Stocks with Patterns", len(results))
            with col2:
                st.metric("Total Patterns", total_patterns)
            with col3:
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            with col4:
                bullish_count = sum(r['summary']['bullish_count'] for r in results)
                bearish_count = sum(r['summary']['bearish_count'] for r in results)
                st.metric("Market Bias", 
                         "Bullish" if bullish_count > bearish_count else "Bearish" if bearish_count > bullish_count else "Neutral")
            
            st.markdown("---")
            
            # Display results
            tabs = st.tabs(["📈 Charts", "📊 Summary Table", "🎯 Top Opportunities"])
            
            with tabs[0]:
                # Charts tab
                cols_per_row = 2
                for idx, result in enumerate(results[:20]):  # Limit to top 20
                    if idx % cols_per_row == 0:
                        cols = st.columns(cols_per_row)
                    
                    with cols[idx % cols_per_row]:
                        st.subheader(result['ticker'])
                        
                        # Pattern badges
                        pattern_html = ""
                        for pattern_name in result['patterns']:
                            confidence = result['patterns'][pattern_name].get('confidence', 0)
                            color = "green" if "Bull" in pattern_name or "Inverse" in pattern_name else "red"
                            pattern_html += f'<span style="background-color:{color};color:white;padding:2px 6px;margin:2px;border-radius:3px;font-size:12px;">{pattern_name} ({confidence:.0%})</span>'
                        
                        st.markdown(pattern_html, unsafe_allow_html=True)
                        
                        # Create and display chart
                        try:
                            fig = ChartVisualizer.create_annotated_chart(
                                result['data'],
                                result['ticker'],
                                result['patterns']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating chart for {result['ticker']}: {e}")
            
            with tabs[1]:
                # Summary table
                table_data = []
                for result in results:
                    for pattern_name, pattern_data in result['patterns'].items():
                        table_data.append({
                            'Ticker': result['ticker'],
                            'Pattern': pattern_name,
                            'Confidence': f"{pattern_data.get('confidence', 0):.1%}",
                            'Bias': result['summary']['bias']
                        })
                
                df_summary = pd.DataFrame(table_data)
                st.dataframe(df_summary, use_container_width=True, hide_index=True)
                
                # Download button
                csv = df_summary.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name=f"pattern_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            
            with tabs[2]:
                # Top opportunities
                st.subheader("🎯 Top Trading Opportunities")
                
                top_bullish = [r for r in results if r['summary']['bias'] == 'Bullish'][:5]
                top_bearish = [r for r in results if r['summary']['bias'] == 'Bearish'][:5]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Top Bullish Setups")
                    for result in top_bullish:
                        patterns_str = ", ".join(result['patterns'].keys())
                        confidence = result['summary']['avg_confidence']
                        st.info(f"**{result['ticker']}**\nPatterns: {patterns_str}\nConfidence: {confidence:.1%}")
                
                with col2:
                    st.markdown("### 📉 Top Bearish Setups")
                    for result in top_bearish:
                        patterns_str = ", ".join(result['patterns'].keys())
                        confidence = result['summary']['avg_confidence']
                        st.warning(f"**{result['ticker']}**\nPatterns: {patterns_str}\nConfidence: {confidence:.1%}")
        
        else:
            st.warning("No patterns found in the selected universe and timeframe.")
    
    else:
        # Welcome screen
        st.markdown("""
        ### Welcome to the Professional Stock Pattern Analyzer
        
        This advanced tool uses sophisticated algorithms to detect technical chart patterns across multiple stocks simultaneously.
        
        **Features:**
        - 🔍 Multi-pattern detection with confidence scoring
        - ⚡ Parallel processing for faster analysis  
        - 📊 Professional-grade visualizations
        - 🎯 Risk-adjusted pattern identification
        - 📈 Support for multiple timeframes
        
        **Get Started:**
        1. Select your preferred stock universe (Nifty 50 or Nifty 500)
        2. Choose a timeframe for analysis
        3. Adjust advanced settings if needed
        4. Click "Run Analysis" to begin
        
        *Note: Analysis may take 1-2 minutes depending on the universe size and network speed.*
        """)

if __name__ == "__main__":
    main()
