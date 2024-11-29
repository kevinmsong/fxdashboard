import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.subplots as sp
from textblob import TextBlob
from newsapi import NewsApiClient
import ta
from typing import Dict, List, Tuple
import plotly.express as px

# Page config
st.set_page_config(
    page_title="fxdashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Updated CSS with JetBrains Mono and refined styling
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
        
        /* Base theme */
        .main {
            background-color: #000000;
        }
        .stApp {
            background-color: #0D1117;
        }
        
        /* Main container spacing */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
            max-width: 95rem !important;
        }
        
        /* Typography */
        .main * {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.9rem;
        }
        h1 { font-size: 1.4rem !important; font-weight: 600 !important; }
        h2 { font-size: 1.3rem !important; font-weight: 600 !important; }
        h3 { font-size: 1.2rem !important; font-weight: 600 !important; }
        small { font-size: 0.8rem !important; }
        
        /* Header styling */
        .header-container {
            background-color: #1A1A1A;
            padding: 0.6rem 1.2rem;
            border-bottom: 1px solid #30363D;
            margin: 0 -1rem 1rem -1rem;
        }
        .header-text {
            font-size: 1.2rem !important;
            margin: 0;
            padding: 0;
            font-weight: 600;
        }
        
        /* Cards and containers */
        .ticker-cell {
            background-color: #1A1A1A;
            padding: 0.6rem;
            border-radius: 4px;
            border: 1px solid #30363D;
            margin: 0.4rem 0;
        }
        .news-card {
            background-color: #1A1A1A;
            padding: 0.8rem;
            border-radius: 4px;
            margin: 0.5rem 0;
            border: 1px solid #30363D;
        }
        .stats-container {
            background-color: #1A1A1A;
            padding: 0.6rem;
            border-radius: 4px;
            margin: 0.3rem 0;
            border: 1px solid #30363D;
        }
        
        /* Data displays */
        .data-label {
            color: #666;
            font-size: 0.85rem !important;
            font-weight: 400;
        }
        .data-value {
            color: #FFF;
            font-size: 0.95rem !important;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace !important;
        }
        
        /* Price displays */
        .price-value {
            font-size: 1.1rem !important;
            font-weight: 600;
            font-family: 'JetBrains Mono', monospace !important;
        }
        .price-change {
            font-size: 0.9rem !important;
            font-weight: 600;
        }
        
        /* Charts */
        .stPlotlyChart {
            background-color: #1A1A1A;
            padding: 0.6rem;
            border-radius: 4px;
            border: 1px solid #30363D;
        }
        
        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
            background-color: #1A1A1A;
            padding: 0.2rem 0.5rem;
            border-radius: 4px 4px 0 0;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #2D3748;
            padding: 0.5rem 1rem;
            font-size: 0.85rem !important;
            font-weight: 400;
        }
        
        /* Sentiment indicators */
        .sentiment-positive { color: #00D700 !important; }
        .sentiment-negative { color: #FF4B4B !important; }
        .sentiment-neutral { color: #FFB000 !important; }
        
        /* Links */
        .news-link {
            color: #61dafb !important;
            text-decoration: none;
        }
        .news-link:hover {
            text-decoration: underline;
            opacity: 0.8;
        }
        
        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.95rem !important;
            font-weight: 600;
        }
        
        /* Tables */
        .dataframe {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 0.85rem !important;
        }
        
        /* Scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #1A1A1A;
        }
        ::-webkit-scrollbar-thumb {
            background: #30363D;
            border-radius: 3px;
        }
    </style>
""", unsafe_allow_html=True)

# Constants
MAJOR_PAIRS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/CHF": "USDCHF=X",
    "USD/CAD": "USDCAD=X",
    "AUD/USD": "AUDUSD=X",
    "NZD/USD": "NZDUSD=X"
}

TIMEFRAMES = {
    "1M": "1m",
    "5M": "5m",
    "15M": "15m",
    "1H": "1h",
    "4H": "4h",
    "1D": "1d",
}

class NewsAggregator:
    def __init__(self, api_key: str):
        self.newsapi = NewsApiClient(api_key=api_key)
        
    def get_forex_news(self) -> Dict:
        """Get general forex market news and analysis"""
        try:
            # Get market news
            market_news = self.newsapi.get_everything(
                q='forex OR "foreign exchange" OR currency trading',
                language='en',
                sort_by='publishedAt',
                page_size=30
            )
            
            # Get specific analysis content
            analysis_news = self.newsapi.get_everything(
                q=('forex analysis OR "technical analysis" OR "forex forecast" OR '
                   '"currency analysis" OR "forex outlook" OR "forex strategy" OR '
                   '"forex trading strategy" OR "market analysis" OR "forex prediction"'),
                language='en',
                sort_by='publishedAt',
                page_size=20
            )
            
            # Process and combine results
            all_articles = market_news['articles'] + analysis_news['articles']
            return self._process_news(all_articles)
            
        except Exception as e:
            st.error(f"Error fetching forex news: {str(e)}")
            return {'breaking': [], 'market': [], 'analysis': []}

    def _process_news(self, articles: List[Dict]) -> Dict:
        """Process and categorize news articles"""
        breaking = []
        market = []
        analysis = []
        
        # Track seen URLs to avoid duplicates
        seen_urls = set()
        
        # Analysis keywords with weights
        analysis_indicators = {
            'technical analysis': 2.0,
            'price analysis': 2.0,
            'market analysis': 2.0,
            'forex analysis': 2.0,
            'trading strategy': 1.5,
            'forecast': 1.5,
            'outlook': 1.5,
            'prediction': 1.5,
            'technical': 1.0,
            'resistance': 1.0,
            'support': 1.0,
            'trend': 1.0,
            'pattern': 1.0,
            'indicator': 1.0,
            'signal': 1.0
        }
        
        for article in articles:
            if not article['title'] or not article['description'] or article['url'] in seen_urls:
                continue
                
            seen_urls.add(article['url'])
            processed = self._analyze_article(article)
            
            # Convert to lowercase for matching
            title_lower = article['title'].lower()
            desc_lower = article['description'].lower()
            
            # Calculate analysis score
            analysis_score = 0
            for term, weight in analysis_indicators.items():
                if term in title_lower:
                    analysis_score += weight * 2  # Higher weight for title matches
                if term in desc_lower:
                    analysis_score += weight
            
            # Check for breaking news
            if any(keyword in title_lower for keyword in ['breaking', 'alert', 'just in', 'urgent']):
                breaking.append(processed)
            # High analysis score articles go to analysis section
            elif analysis_score >= 2.0:
                analysis.append(processed)
            else:
                market.append(processed)
        
        # Sort analysis articles by sentiment strength (more opinionated articles first)
        analysis.sort(key=lambda x: abs(x['sentiment']), reverse=True)
        
        return {
            'breaking': self._unique_articles(breaking[:5]),
            'market': self._unique_articles(market[:10]),
            'analysis': self._unique_articles(analysis[:5])
        }

    def _analyze_article(self, article: Dict) -> Dict:
        """Analyze single article for sentiment and relevance"""
        text = f"{article['title']} {article['description']}"
        analysis = TextBlob(text)
        
        return {
            'title': article['title'],
            'description': article['description'],
            'source': article['source']['name'],
            'url': article['url'],
            'published': datetime.strptime(article['publishedAt'], '%Y-%m-%dT%H:%M:%SZ'),
            'sentiment': analysis.sentiment.polarity,
            'subjectivity': analysis.sentiment.subjectivity
        }
    
    @staticmethod
    def _unique_articles(articles: List[Dict]) -> List[Dict]:
        """Remove duplicate articles based on title similarity"""
        unique = []
        seen_titles = set()
        
        for article in articles:
            title_key = article['title'].lower()[:50]  # Use first 50 chars as key
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique.append(article)
        
        return unique

    def get_pair_news(self, currency_pair: str) -> List[Dict]:
        """Get news for specific currency pair"""
        try:
            base, quote = currency_pair.split('/')
            news = self.newsapi.get_everything(
                q=f'{base} {quote} exchange rate OR {currency_pair} forex',
                language='en',
                sort_by='publishedAt',
                page_size=10
            )
            return [self._analyze_article(article) for article in news['articles']
                    if article['title'] and article['description']]
        except Exception as e:
            st.error(f"Error fetching pair news: {str(e)}")
            return []

class MarketData:
    @staticmethod
    def get_technical_signals(df: pd.DataFrame) -> Dict[str, str]:
        """Generate technical analysis signals"""
        signals = {}
        
        try:
            # Trend Signals
            current_close = float(df['Close'].iloc[-1])
            ema_200 = float(df['EMA_200'].iloc[-1])
            signals['Trend'] = "Bullish" if current_close > ema_200 else "Bearish"
            
            # RSI Signals
            current_rsi = float(df['RSI'].iloc[-1])
            if current_rsi > 70:
                signals['RSI'] = "Overbought"
            elif current_rsi < 30:
                signals['RSI'] = "Oversold"
            else:
                signals['RSI'] = "Neutral"
            
            # MACD Signal
            current_macd = float(df['MACD'].iloc[-1])
            signals['MACD'] = "Bullish" if current_macd > 0 else "Bearish"
            
            # Volatility Signal
            current_vol = float(df['Volatility'].iloc[-1])
            vol_series = pd.Series(df['Volatility'].dropna())
            vol_percentile = vol_series.rank(pct=True).iloc[-1]
            
            if vol_percentile > 0.8:
                signals['Volatility'] = "High"
            elif vol_percentile < 0.2:
                signals['Volatility'] = "Low"
            else:
                signals['Volatility'] = "Normal"
            
        except Exception as e:
            st.error(f"Error calculating signals: {str(e)}")
            signals = {
                'Trend': "N/A",
                'RSI': "N/A",
                'MACD': "N/A",
                'Volatility': "N/A"
            }
        
        return signals

    @staticmethod
    def get_forex_data(symbol: str, timeframe: str, period: str = '1d') -> pd.DataFrame:
        """Fetch and process forex data"""
        try:
            df = yf.download(symbol, period=period, interval=timeframe)
            if df.empty:
                return pd.DataFrame()
            
            return MarketData._calculate_indicators(df)
        except Exception as e:
            st.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def _calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators"""
        try:
            df = df.copy()
            
            # Convert to 1D arrays
            close = df['Close'].astype(float).values.flatten()
            high = df['High'].astype(float).values.flatten()
            low = df['Low'].astype(float).values.flatten()
            
            # Create series with proper index
            close_series = pd.Series(close, index=df.index)
            high_series = pd.Series(high, index=df.index)
            low_series = pd.Series(low, index=df.index)
            
            # Calculate indicators
            df['SMA_20'] = ta.trend.sma_indicator(close_series, 20)
            df['EMA_20'] = ta.trend.ema_indicator(close_series, 20)
            df['EMA_50'] = ta.trend.ema_indicator(close_series, 50)
            df['EMA_200'] = ta.trend.ema_indicator(close_series, 200)
            df['RSI'] = ta.momentum.rsi(close_series, 14)
            df['MACD'] = ta.trend.macd_diff(close_series)
            
            # Volatility calculation
            df['Daily_Return'] = df['Close'].pct_change()
            df['Volatility'] = df['Daily_Return'].rolling(window=20).std()
            
            return df
            
        except Exception as e:
            st.error(f"Error calculating indicators: {str(e)}")
            return df

    @staticmethod
    def calculate_correlation_matrix(pairs: List[str], timeframe: str) -> pd.DataFrame:
        """Calculate correlation matrix for currency pairs"""
        returns_dict = {}
        for pair in pairs:
            df = MarketData.get_forex_data(MAJOR_PAIRS[pair], timeframe)
            if not df.empty:
                returns_dict[pair] = df['Daily_Return']
        
        if returns_dict:
            return pd.DataFrame(returns_dict).corr()
        return pd.DataFrame()

class TerminalUI:
    def __init__(self, news_aggregator: NewsAggregator):
        self.news = news_aggregator
        self.current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def render_header(self):
        """Render terminal header with proper spacing"""
        # Add empty space to account for streamlit's top bar
        st.markdown("<div style='height: 3rem;'></div>", unsafe_allow_html=True)
        
        # Header container
        st.markdown("""
            <div class="header-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h3 class="header-text">📊 FX ANALYTICS</h3>
                        <div class="timestamp">Last updated: {}</div>
                    </div>
                    <div>🔄 Live Feed</div>
                </div>
            </div>
        """.format(self.current_time), unsafe_allow_html=True)
        
        # Add small space after header
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    def render_market_overview(self, timeframe: str):
        """Render enhanced market overview section"""
        st.markdown("### Market Overview")
        
        # Create main metrics
        cols = st.columns(len(MAJOR_PAIRS))
        for idx, (pair, symbol) in enumerate(MAJOR_PAIRS.items()):
            with cols[idx]:
                data = MarketData.get_forex_data(symbol, timeframe)
                if not data.empty:
                    current = float(data['Close'].iloc[-1])
                    prev = float(data['Close'].iloc[-2])
                    change = ((current - prev) / prev) * 100
                    
                    signals = MarketData.get_technical_signals(data)
                    
                    # Enhanced ticker cell with signals
                    color = "green" if change >= 0 else "red"
                    st.markdown(f"""
                        <div class="ticker-cell">
                            <div>{pair}</div>
                            <div style="color: {color}">
                                {current:.4f} ({change:+.2f}%)
                            </div>
                            <div class="data-label">
                                {signals['Trend']} | RSI: {signals['RSI']}
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

    def render_detailed_analysis(self, timeframe: str):
        """Render detailed analysis section"""
        # Create tabs for different analysis views
        tab1, tab2, tab3 = st.tabs(["Charts", "Correlations", "Heat Map"])
        
        with tab1:
            # Currency pair selector
            selected_pair = st.selectbox("Select Currency Pair", list(MAJOR_PAIRS.keys()))
            data = MarketData.get_forex_data(MAJOR_PAIRS[selected_pair], timeframe)
            
            if not data.empty:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Main price chart with indicators
                    fig = self._create_advanced_chart(data, selected_pair)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    self._render_technical_summary(data)
        
        with tab2:
            self.render_correlation_matrix(timeframe)
        
        with tab3:
            self.render_currency_heat_map(timeframe)

    def _create_advanced_chart(self, data: pd.DataFrame, pair: str) -> go.Figure:
        """Create advanced chart with multiple indicators"""
        fig = sp.make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{pair} Price', 'RSI', 'MACD')
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ),
            row=1, col=1
        )

        # Add EMAs
        for period, color in zip([20, 50, 200], ['orange', 'blue', 'red']):
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[f'EMA_{period}'],
                    name=f'EMA {period}',
                    line=dict(color=color)
                ),
                row=1, col=1
            )

        # RSI
        fig.add_trace(
            go.Scatter(x=data.index, y=data['RSI'], name='RSI'),
            row=2, col=1
        )
        
        # Add RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

        # MACD
        fig.add_trace(
            go.Scatter(x=data.index, y=data['MACD'], name='MACD'),
            row=3, col=1
        )

        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False
        )

        return fig

    def _render_technical_summary(self, data: pd.DataFrame):
        """Render technical analysis summary"""
        st.markdown("### Technical Analysis")
        signals = MarketData.get_technical_signals(data)
        
        # Signal indicators
        for signal, value in signals.items():
            color = (
                "green" if value in ["Bullish", "Low"] 
                else "red" if value in ["Bearish", "High", "Overbought"] 
                else "yellow"
            )
            st.markdown(f"""
                <div class="stats-container">
                    <div class="data-label">{signal}</div>
                    <div class="data-value" style="color: {color}">{value}</div>
                </div>
            """, unsafe_allow_html=True)

        # Volatility gauge
        vol = data['Volatility'].iloc[-1] * np.sqrt(252) * 100
        st.markdown("### Volatility (Annualized)")
        st.progress(min(vol/50, 1.0))
        st.markdown(f"<div class='data-value'>{vol:.1f}%</div>", unsafe_allow_html=True)

    def render_correlation_matrix(self, timeframe: str):
        """Render correlation matrix heatmap"""
        st.markdown("### Currency Correlations")
        pairs = list(MAJOR_PAIRS.keys())
        corr_matrix = MarketData.calculate_correlation_matrix(pairs, timeframe)
        
        fig = px.imshow(
            corr_matrix,
            labels=dict(color="Correlation"),
            color_continuous_scale="RdBu",
            aspect="auto"
        )
        fig.update_layout(
            template='plotly_dark',
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_currency_heat_map(self, timeframe: str):
        """Render currency strength heat map"""
        st.markdown("### Currency Strength Heat Map")
        
        # Calculate strength for each currency
        strength_data = {}
        for pair, symbol in MAJOR_PAIRS.items():
            data = MarketData.get_forex_data(symbol, timeframe)
            if not data.empty:
                returns = data['Daily_Return'].iloc[-1] * 100
                base, quote = pair.split('/')
                
                if base not in strength_data:
                    strength_data[base] = []
                if quote not in strength_data:
                    strength_data[quote] = []
                
                strength_data[base].append(returns)
                strength_data[quote].append(-returns)
        
        # Calculate average strength
        strength = {curr: np.mean(vals) for curr, vals in strength_data.items()}
        
        # Create heat map
        currencies = list(strength.keys())
        values = list(strength.values())
        
        fig = go.Figure(data=[go.Bar(
            x=currencies,
            y=values,
            marker_color=['red' if x < 0 else 'green' for x in values]
        )])
        
        fig.update_layout(
            template='plotly_dark',
            title="Currency Strength (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    def render_news_section(self):
        """Render news section"""
        news_data = self.news.get_forex_news()
        
        # Breaking News
        if news_data['breaking']:
            st.markdown("### 🚨 Breaking News")
            for article in news_data['breaking']:
                self._render_news_card(article, breaking=True)
        
        # Market News and Analysis in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📰 Market News")
            for article in news_data['market']:
                self._render_news_card(article)
                
        with col2:
            st.markdown("### 📊 Analysis")
            for article in news_data['analysis']:
                self._render_news_card(article)

    def _render_news_card(self, article: Dict, breaking: bool = False):
        """Render more compact news card with clickable link"""
        sentiment_class = (
            "sentiment-positive" if article['sentiment'] > 0.1
            else "sentiment-negative" if article['sentiment'] < -0.1
            else "sentiment-neutral"
        )
        
        time_ago = self._get_time_ago(article['published'])
        
        st.markdown(f"""
            <div class="news-card" style="border-left: 2px solid {'#FF0000' if breaking else '#30363D'}">
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem;">
                    <small>{article['source']} • {time_ago}</small>
                    <small class="{sentiment_class}">
                        {'▲' if article['sentiment'] > 0.1 else '▼' if article['sentiment'] < -0.1 else '■'}
                    </small>
                </div>
                <h4>
                    <a href="{article['url']}" target="_blank" class="news-link">
                        {article['title']}
                    </a>
                </h4>
                <p>{article['description'][:150]}...</p>
            </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def _get_time_ago(published_date: datetime) -> str:
        """Convert datetime to relative time string"""
        now = datetime.now()
        diff = now - published_date
        
        if diff.days > 0:
            return f"{diff.days}d ago"
        hours = diff.seconds // 3600
        if hours > 0:
            return f"{hours}h ago"
        minutes = (diff.seconds % 3600) // 60
        return f"{minutes}m ago"

def main():
    # Initialize components
    news_aggregator = NewsAggregator(api_key='82471e592e8c4d3d95ec33e3add393be')
    terminal = TerminalUI(news_aggregator)
    
    # Render layout
    terminal.render_header()
    
    # Fixed timeframe (1M)
    timeframe = "1M"
    
    # Market Overview
    terminal.render_market_overview(timeframe)
    
    # Detailed Analysis
    terminal.render_detailed_analysis(timeframe)
    
    # News Section
    terminal.render_news_section()

if __name__ == "__main__":
    main()