"""
Enhanced Professional Fintech Platform with Multi-Agent System
Modern UI with Gemini AI Integration
Dual Navigation: Features + AI Agents
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import google.generativeai as genai
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import json
import time
from typing import Dict, List, Optional, Tuple
import ta
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# ==================== Configuration ====================
st.set_page_config(
    page_title="WVB Side Project- AI Platform",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Professional AI-Powered Financial Analysis Platform"
    }
)

# ==================== Custom CSS for Professional UI ====================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* Metric cards */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #3a5998;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00d4ff, #0099ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(90deg, #0099ff, #00d4ff);
        color: white;
        border: none;
        padding: 10px 24px;
        border-radius: 5px;
        font-weight: bold;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 153, 255, 0.4);
    }
    
    /* Input fields */
    .stTextInput > div > div > input {
        background-color: #1a1a2e;
        border: 1px solid #3a5998;
        border-radius: 5px;
        color: white;
    }
    
    /* Selectbox */
    .stSelectbox > div > div {
        background-color: #1a1a2e;
        border: 1px solid #3a5998;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #1a1a2e, #16213e);
        border-radius: 10px;
        padding: 5px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a0a0a0;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #0099ff, #00d4ff);
        color: white !important;
        border-radius: 5px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #1e3c72, #2a5298);
        border-radius: 5px;
        color: white;
    }
    
    /* Charts background */
    .js-plotly-plot {
        background: rgba(26, 26, 46, 0.5) !important;
        border-radius: 10px;
        padding: 10px;
    }
    
    /* Success/Error/Warning/Info boxes */
    .stAlert {
        background: rgba(26, 26, 46, 0.8);
        border: 1px solid;
        border-radius: 5px;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: rgba(26, 26, 46, 0.7);
        border-radius: 10px;
        border: 1px solid #3a5998;
    }
    
    /* Dataframe */
    .dataframe {
        background: #1a1a2e !important;
        color: white !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #00d4ff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==================== Initialize Session State ====================
# Load API key from environment only
if 'api_key' not in st.session_state:
    api_key_from_env = os.getenv('GEMINI_API_KEY', '')
    if api_key_from_env:
        st.session_state.api_key = api_key_from_env
        st.session_state.api_configured = True
    else:
        st.session_state.api_key = ''
        st.session_state.api_configured = False

if 'selected_agent' not in st.session_state:
    st.session_state.selected_agent = None
if 'agent_mode' not in st.session_state:
    st.session_state.agent_mode = False
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = {}
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'NVDA']
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

# ==================== Professional Data Classes ====================
class MarketData:
    """Professional market data handler"""
    
    @staticmethod
    @st.cache_data(ttl=300)
    def get_stock_data(ticker: str, period: str = "1mo") -> Tuple[pd.DataFrame, Dict]:
        """Fetch and cache stock data"""
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period=period)
            info = ticker_obj.info
            
            # Add technical indicators
            if not df.empty:
                df = MarketData.add_technical_indicators(df)
            
            return df, info
        except Exception as e:
            st.error(f"Error fetching {ticker}: {str(e)}")
            return pd.DataFrame(), {}
    
    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Add comprehensive technical indicators"""
        if df.empty:
            return df
        
        # Moving Averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # RSI
        try:
            df['RSI'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
        except:
            df['RSI'] = 50
        
        # MACD
        try:
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Diff'] = macd.macd_diff()
        except:
            df['MACD'] = 0
            df['MACD_Signal'] = 0
            df['MACD_Diff'] = 0
        
        # Bollinger Bands
        try:
            bb = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = bb.bollinger_hband()
            df['BB_Middle'] = bb.bollinger_mavg()
            df['BB_Lower'] = bb.bollinger_lband()
        except:
            df['BB_Upper'] = df['Close']
            df['BB_Middle'] = df['Close']
            df['BB_Lower'] = df['Close']
        
        # ATR
        try:
            df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close']).average_true_range()
        except:
            df['ATR'] = 0
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
    
    @staticmethod
    @st.cache_data(ttl=600)
    def get_market_overview() -> Dict:
        """Get market indices overview"""
        indices = {
            '^GSPC': 'S&P 500',
            '^DJI': 'Dow Jones',
            '^IXIC': 'NASDAQ',
            '^VIX': 'VIX',
            '^TNX': '10Y Treasury',
            'GC=F': 'Gold',
            'CL=F': 'Crude Oil',
            'BTC-USD': 'Bitcoin'
        }
        
        overview = {}
        for symbol, name in indices.items():
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="1d")
                
                if not hist.empty:
                    current = hist['Close'].iloc[-1]
                    prev_close = hist['Open'].iloc[0]
                    change = ((current - prev_close) / prev_close) * 100
                else:
                    current = info.get('regularMarketPrice', 0)
                    change = info.get('regularMarketChangePercent', 0)
                
                overview[name] = {
                    'price': current,
                    'change': change,
                    'symbol': symbol
                }
            except:
                overview[name] = {'price': 0, 'change': 0, 'symbol': symbol}
        
        return overview

# ==================== AI Agent Classes ====================
class AIFinancialAdvisor:
    """General AI Financial Advisor"""
    
    def __init__(self, api_key: str):
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.configured = True
        else:
            self.configured = False
    
    def analyze_stock(self, ticker: str, df: pd.DataFrame, info: Dict) -> str:
        """Comprehensive stock analysis"""
        if not self.configured:
            return "Please configure your Gemini API key first."
        
        try:
            # Prepare analysis data
            current_price = info.get('currentPrice', df['Close'].iloc[-1] if not df.empty else 0)
            market_cap = info.get('marketCap', 0)
            pe_ratio = info.get('trailingPE', 0)
            dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
            
            # Technical indicators from latest data
            if not df.empty:
                latest = df.iloc[-1]
                rsi = latest.get('RSI', 50)
                macd = latest.get('MACD', 0)
                macd_signal = latest.get('MACD_Signal', 0)
                
                # Price changes
                price_1d = ((df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100) if len(df) > 1 else 0
                price_1w = ((df['Close'].iloc[-1] / df['Close'].iloc[-5] - 1) * 100) if len(df) > 5 else 0
                price_1m = ((df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100) if len(df) > 0 else 0
                
                # Volume analysis
                avg_volume = df['Volume'].mean()
                current_volume = df['Volume'].iloc[-1]
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            else:
                rsi = 50
                macd = 0
                macd_signal = 0
                price_1d = price_1w = price_1m = 0
                volume_ratio = 1
            
            prompt = f"""
            As a professional financial analyst, provide a comprehensive analysis of {ticker}:
            
            CURRENT DATA:
            - Price: ${current_price:.2f}
            - Market Cap: ${market_cap/1e9:.2f}B
            - P/E Ratio: {pe_ratio:.2f}
            - Dividend Yield: {dividend_yield:.2f}%
            
            PERFORMANCE:
            - 1 Day: {price_1d:+.2f}%
            - 1 Week: {price_1w:+.2f}%
            - 1 Month: {price_1m:+.2f}%
            
            TECHNICAL INDICATORS:
            - RSI: {rsi:.2f}
            - MACD: {macd:.4f} (Signal: {macd_signal:.4f})
            - Volume Ratio: {volume_ratio:.2f}x average
            
            Provide:
            1. INVESTMENT RATING (Buy/Hold/Sell) with confidence level
            2. KEY STRENGTHS (2-3 points)
            3. KEY RISKS (2-3 points)
            4. TECHNICAL OUTLOOK
            5. PRICE TARGET (next 3-6 months)
            6. ACTIONABLE RECOMMENDATION
            
            Keep the analysis professional, data-driven, and concise.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def portfolio_optimization(self, portfolio: Dict) -> str:
        """AI-powered portfolio optimization suggestions"""
        if not self.configured:
            return "Please configure your Gemini API key first."
        
        try:
            portfolio_str = json.dumps(portfolio, indent=2)
            
            prompt = f"""
            As a portfolio manager, analyze and optimize this portfolio:
            
            {portfolio_str}
            
            Provide:
            1. PORTFOLIO HEALTH SCORE (0-100)
            2. DIVERSIFICATION ANALYSIS
            3. RISK ASSESSMENT
            4. REBALANCING SUGGESTIONS
            5. SPECIFIC ACTIONS TO IMPROVE RETURNS
            
            Consider modern portfolio theory and current market conditions.
            Be specific and actionable.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Portfolio analysis error: {str(e)}"
    
    def market_insights(self, query: str) -> str:
        """Get market insights and recommendations"""
        if not self.configured:
            return "Please configure your Gemini API key first."
        
        try:
            prompt = f"""
            As a senior market strategist, answer this question:
            
            {query}
            
            Provide:
            - Clear, actionable insights
            - Data-driven reasoning
            - Risk considerations
            - Specific recommendations
            
            Keep the response professional and concise.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
            
        except Exception as e:
            return f"Error generating insights: {str(e)}"

class SpecializedAgent:
    """Base class for specialized AI agents"""
    
    def __init__(self, name: str, role: str, expertise: List[str], api_key: str):
        self.name = name
        self.role = role
        self.expertise = expertise
        self.configured = False
        
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            self.configured = True
    
    def chat(self, query: str) -> str:
        """Chat with the agent"""
        if not self.configured:
            return "API key not configured."
        
        try:
            prompt = f"""
            You are {self.name}, a {self.role}.
            Your expertise includes: {', '.join(self.expertise)}
            
            User query: {query}
            
            Provide a professional, detailed response based on your expertise.
            """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

class ChartGenerator:
    """Professional financial charts"""
    
    @staticmethod
    def create_advanced_candlestick(df: pd.DataFrame, ticker: str) -> go.Figure:
        """Create professional candlestick chart with indicators"""
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
            row_heights=[0.5, 0.15, 0.15, 0.2],
            subplot_titles=(f'{ticker} Price', 'Volume', 'RSI', 'MACD')
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='OHLC',
                increasing_line_color='#00d4ff',
                decreasing_line_color='#ff3366'
            ),
            row=1, col=1
        )
        
        # Add moving averages
        if 'SMA_20' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20',
                          line=dict(color='#ffa500', width=1)),
                row=1, col=1
            )
        
        if 'SMA_50' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50',
                          line=dict(color='#00ff00', width=1)),
                row=1, col=1
            )
        
        # Bollinger Bands
        if 'BB_Upper' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Upper',
                          line=dict(color='rgba(250,250,250,0.3)', width=1, dash='dash')),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Lower',
                          line=dict(color='rgba(250,250,250,0.3)', width=1, dash='dash'),
                          fill='tonexty', fillcolor='rgba(100,100,100,0.1)'),
                row=1, col=1
            )
        
        # Volume bars
        colors = ['#00d4ff' if df['Close'].iloc[i] >= df['Open'].iloc[i] else '#ff3366'
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(x=df.index, y=df['Volume'], name='Volume',
                  marker_color=colors, opacity=0.7),
            row=2, col=1
        )
        
        # RSI
        if 'RSI' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['RSI'], name='RSI',
                          line=dict(color='#9966ff', width=2)),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red",
                         line_width=1, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green",
                         line_width=1, row=3, col=1)
        
        # MACD
        if 'MACD' in df.columns:
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD'], name='MACD',
                          line=dict(color='#00d4ff', width=2)),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal',
                          line=dict(color='#ffa500', width=2)),
                row=4, col=1
            )
            fig.add_trace(
                go.Bar(x=df.index, y=df['MACD_Diff'], name='Histogram',
                      marker_color='#808080', opacity=0.5),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            template='plotly_dark',
            height=800,
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False,
            paper_bgcolor='rgba(26, 26, 46, 0.8)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            font=dict(color='white'),
            title=dict(
                text=f"{ticker} Technical Analysis",
                font=dict(size=20, color='#00d4ff')
            )
        )
        
        # Update y-axes labels
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="MACD", row=4, col=1)
        
        return fig
    
    @staticmethod
    def create_portfolio_charts(portfolio: Dict) -> Tuple[go.Figure, go.Figure]:
        """Create portfolio allocation and performance charts"""
        
        if not portfolio:
            return go.Figure(), go.Figure()
        
        # Prepare data
        tickers = list(portfolio.keys())
        values = [pos['value'] for pos in portfolio.values()]
        returns = [pos['return_pct'] for pos in portfolio.values()]
        
        # Allocation pie chart
        fig_pie = go.Figure(data=[
            go.Pie(
                labels=tickers,
                values=values,
                hole=0.4,
                marker=dict(
                    colors=px.colors.sequential.Viridis,
                    line=dict(color='white', width=2)
                ),
                textfont=dict(size=14, color='white'),
                textposition='auto',
                textinfo='label+percent'
            )
        ])
        
        fig_pie.update_layout(
            title="Portfolio Allocation",
            template='plotly_dark',
            height=400,
            paper_bgcolor='rgba(26, 26, 46, 0.8)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            font=dict(color='white')
        )
        
        # Performance bar chart
        colors = ['#00d4ff' if r >= 0 else '#ff3366' for r in returns]
        
        fig_bar = go.Figure(data=[
            go.Bar(
                x=tickers,
                y=returns,
                marker_color=colors,
                text=[f"{r:+.1f}%" for r in returns],
                textposition='outside',
                textfont=dict(size=12, color='white')
            )
        ])
        
        fig_bar.update_layout(
            title="Portfolio Performance",
            template='plotly_dark',
            height=400,
            yaxis_title="Return (%)",
            paper_bgcolor='rgba(26, 26, 46, 0.8)',
            plot_bgcolor='rgba(26, 26, 46, 0.5)',
            font=dict(color='white'),
            showlegend=False
        )
        
        return fig_pie, fig_bar

# ==================== Main Application ====================
def main():
    # Check API configuration
    if not st.session_state.api_configured:
        st.error("⚠️ GEMINI_API_KEY not found in .env file. Please create a .env file with your API key.")
        st.info("Create a .env file in your project directory with: GEMINI_API_KEY=your_api_key_here")
        return
    
    # Header with gradient effect
    st.markdown("""
    <div style='text-align: center; padding: 20px; background: linear-gradient(90deg, #1e3c72, #2a5298); border-radius: 10px; margin-bottom: 30px;'>
        <h1 style='color: white; font-size: 3em; margin: 0;'>💎WVB Sample AI Platform</h1>
        <p style='color: #a0a0ff; font-size: 1.2em; margin-top: 10px;'>Professional Investment Intelligence Powered by AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize AI components
    ai_advisor = AIFinancialAdvisor(st.session_state.api_key)
    
    # Create specialized agents
    agents = {
        "Market Analyst": SpecializedAgent(
            "Market Analyst AI",
            "Senior Market Analyst",
            ["Market trends", "Sector analysis", "Economic indicators"],
            st.session_state.api_key
        ),
        "Technical Analyst": SpecializedAgent(
            "Technical Analyst AI",
            "Senior Technical Analyst",
            ["Chart patterns", "Technical indicators", "Trading signals"],
            st.session_state.api_key
        ),
        "Risk Manager": SpecializedAgent(
            "Risk Manager AI",
            "Chief Risk Officer",
            ["Risk assessment", "Portfolio risk", "Hedging strategies"],
            st.session_state.api_key
        ),
        "Portfolio Manager": SpecializedAgent(
            "Portfolio Manager AI",
            "Senior Portfolio Manager",
            ["Asset allocation", "Portfolio optimization", "Rebalancing"],
            st.session_state.api_key
        ),
        "Quant Analyst": SpecializedAgent(
            "Quant Analyst AI",
            "Quantitative Analyst",
            ["Statistical analysis", "Algorithmic trading", "Backtesting"],
            st.session_state.api_key
        ),
        "News Analyst": SpecializedAgent(
            "News Analyst AI",
            "News & Sentiment Analyst",
            ["News analysis", "Sentiment analysis", "Market events"],
            st.session_state.api_key
        )
    }
    
    # Dual Navigation System
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Main Feature Tabs
        main_tab = st.selectbox(
            "Select Feature",
            ["🤖 AI Analysis Hub", "📈 Technical Analysis", "💼 Portfolio Manager", 
             "🔍 Smart Screener", "📊 Market Insights", "⚡ Live Trading Signals"],
            key="main_navigation"
        )
    
    with col2:
        # Agent Navigation
        if st.button("🤖 AI Agents Panel", use_container_width=True):
            st.session_state.agent_mode = not st.session_state.agent_mode
    
    # Show agent panel if activated
    if st.session_state.agent_mode:
        st.markdown("""
        <div style='background: linear-gradient(90deg, #1a1a2e, #16213e); padding: 20px; border-radius: 15px; margin: 20px 0;'>
            <h2 style='text-align: center; color: white;'>🤖 Select Your AI Agent</h2>
        </div>
        """, unsafe_allow_html=True)
        
        agent_cols = st.columns(6)
        agent_icons = {
            "Market Analyst": "📊",
            "Technical Analyst": "📈",
            "Risk Manager": "🛡️",
            "Portfolio Manager": "💼",
            "Quant Analyst": "🔢",
            "News Analyst": "📰"
        }
        
        for idx, (agent_name, agent) in enumerate(agents.items()):
            with agent_cols[idx]:
                if st.button(
                    f"{agent_icons[agent_name]}\n{agent_name}",
                    key=f"agent_{agent_name}",
                    use_container_width=True
                ):
                    st.session_state.selected_agent = agent_name
                    if agent_name not in st.session_state.chat_messages:
                        st.session_state.chat_messages[agent_name] = []
        
        # Agent Chat Interface
        if st.session_state.selected_agent:
            current_agent = agents[st.session_state.selected_agent]
            
            st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 15px; border-radius: 10px; margin: 20px 0;'>
                <h3 style='color: white;'>Chatting with: {current_agent.name}</h3>
                <p style='color: #e0e0ff;'>Expertise: {', '.join(current_agent.expertise)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Chat history
            if st.session_state.selected_agent not in st.session_state.chat_messages:
                st.session_state.chat_messages[st.session_state.selected_agent] = []
            
            for msg in st.session_state.chat_messages[st.session_state.selected_agent]:
                with st.chat_message(msg["role"]):
                    st.write(msg["content"])
            
            # Chat input
            if prompt := st.chat_input(f"Ask {st.session_state.selected_agent}..."):
                st.session_state.chat_messages[st.session_state.selected_agent].append(
                    {"role": "user", "content": prompt}
                )
                with st.chat_message("user"):
                    st.write(prompt)
                
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = current_agent.chat(prompt)
                        st.write(response)
                        st.session_state.chat_messages[st.session_state.selected_agent].append(
                            {"role": "assistant", "content": response}
                        )
        
        st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
            <h2 style='color: white; margin: 0;'>📊 Market Data</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # API Status
        st.success("✅ AI Engine Connected")
        
        st.divider()
        
        # Market Overview
        st.markdown("### 📊 Global Markets")
        
        with st.spinner("Loading market data..."):
            market_overview = MarketData.get_market_overview()
            
            for name, data in market_overview.items():
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{name}**")
                with col2:
                    color = "🟢" if data['change'] >= 0 else "🔴"
                    st.markdown(f"{color} {data['change']:+.2f}%")
                st.markdown(f"${data['price']:,.2f}")
                st.markdown("---")
        
        # Watchlist
        st.markdown("### 👁️ Watchlist")
        new_ticker = st.text_input("Add ticker", key="watchlist_add")
        if st.button("➕ Add", key="add_to_watchlist"):
            if new_ticker and new_ticker.upper() not in st.session_state.watchlist:
                st.session_state.watchlist.append(new_ticker.upper())
                st.success(f"Added {new_ticker.upper()}")
                st.rerun()
        
        # Display watchlist
        for ticker in st.session_state.watchlist:
            col1, col2 = st.columns([3, 1])
            with col1:
                if st.button(f"📈 {ticker}", key=f"watch_{ticker}"):
                    st.session_state.selected_ticker = ticker
            with col2:
                if st.button("❌", key=f"remove_{ticker}"):
                    st.session_state.watchlist.remove(ticker)
                    st.rerun()
    
    # Main Content based on selected tab
    if main_tab == "🤖 AI Analysis Hub":
        render_ai_analysis_hub(ai_advisor)
    elif main_tab == "📈 Technical Analysis":
        render_technical_analysis()
    elif main_tab == "💼 Portfolio Manager":
        render_portfolio_manager(ai_advisor)
    elif main_tab == "🔍 Smart Screener":
        render_smart_screener()
    elif main_tab == "📊 Market Insights":
        render_market_insights(ai_advisor)
    elif main_tab == "⚡ Live Trading Signals":
        render_trading_signals(ai_advisor)
    
    # Footer
    st.markdown("""
    <div style='margin-top: 50px; padding: 20px; background: linear-gradient(90deg, #1e3c72, #2a5298); border-radius: 10px; text-align: center;'>
        <p style='color: white; margin: 0;'>Built with ❤️ by Vaibhav </p>
        <p style='color: #a0a0ff; margin: 5px 0;'>Professional Multi-Agent Financial Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)

# ==================== Feature Rendering Functions ====================
def render_ai_analysis_hub(ai_advisor):
    """Render AI Analysis Hub"""
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); border-radius: 10px;'>
        <h2 style='color: white; text-align: center;'>🧠 AI-Powered Financial Intelligence</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        ticker_input = st.text_input("Enter Ticker Symbol", value="AAPL", key="ai_ticker")
        analysis_type = st.selectbox(
            "Analysis Type",
            ["Comprehensive", "Technical", "Fundamental", "Risk Assessment"]
        )
        
        if st.button("🚀 Generate AI Analysis", key="generate_analysis"):
            if ticker_input:
                with st.spinner(f"Analyzing {ticker_input}..."):
                    df, info = MarketData.get_stock_data(ticker_input.upper(), "3mo")
                    if not df.empty:
                        analysis = ai_advisor.analyze_stock(ticker_input.upper(), df, info)
                        st.session_state.analysis_history.append({
                            'ticker': ticker_input.upper(),
                            'analysis': analysis,
                            'timestamp': datetime.now()
                        })
    
    with col2:
        # Display latest analysis
        if st.session_state.analysis_history:
            latest = st.session_state.analysis_history[-1]
            st.markdown(f"### Analysis for {latest['ticker']}")
            st.info(latest['analysis'])
        else:
            st.info("Enter a ticker and click 'Generate AI Analysis' to start")
    
    # General Chat Interface
    st.markdown("### 💬 Ask Your AI Financial Advisor")
    
    if "general_chat" not in st.session_state.chat_messages:
        st.session_state.chat_messages["general_chat"] = []
    
    # Display chat history
    for msg in st.session_state.chat_messages["general_chat"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Chat input
    if question := st.chat_input("Ask any financial question..."):
        st.session_state.chat_messages["general_chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ai_advisor.market_insights(question)
                st.write(response)
                st.session_state.chat_messages["general_chat"].append({"role": "assistant", "content": response})

def render_technical_analysis():
    """Render Technical Analysis"""
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); border-radius: 10px;'>
        <h2 style='color: white; text-align: center;'>📊 Professional Technical Analysis</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        ta_ticker = st.text_input("Ticker Symbol", value="AAPL", key="ta_ticker")
    with col2:
        ta_period = st.selectbox("Period", ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y"])
    with col3:
        ta_interval = st.selectbox("Interval", ["1m", "5m", "15m", "1h", "1d"])
    
    if st.button("📈 Load Chart", key="load_ta_chart"):
        with st.spinner(f"Loading {ta_ticker} data..."):
            df, info = MarketData.get_stock_data(ta_ticker.upper(), ta_period)
            
            if not df.empty:
                # Display metrics
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                
                with col1:
                    current_price = info.get('currentPrice', df['Close'].iloc[-1])
                    st.metric("Price", f"${current_price:.2f}")
                
                with col2:
                    day_change = info.get('regularMarketChangePercent', 0)
                    st.metric("Day Change", f"{day_change:+.2f}%")
                
                with col3:
                    volume = df['Volume'].iloc[-1]
                    st.metric("Volume", f"{volume/1e6:.1f}M")
                
                with col4:
                    rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else 50
                    st.metric("RSI", f"{rsi:.1f}")
                
                with col5:
                    if 'MACD' in df.columns:
                        macd_signal = "Buy" if df['MACD'].iloc[-1] > df['MACD_Signal'].iloc[-1] else "Sell"
                    else:
                        macd_signal = "N/A"
                    st.metric("MACD Signal", macd_signal)
                
                with col6:
                    volatility = df['Close'].pct_change().std() * np.sqrt(252) * 100
                    st.metric("Volatility", f"{volatility:.1f}%")
                
                # Display advanced chart
                st.plotly_chart(
                    ChartGenerator.create_advanced_candlestick(df, ta_ticker.upper()),
                    use_container_width=True
                )

def render_portfolio_manager(ai_advisor):
    """Render Portfolio Manager"""
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); border-radius: 10px;'>
        <h2 style='color: white; text-align: center;'>💼 Smart Portfolio Management</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Portfolio Input
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Add Position")
        p_ticker = st.text_input("Ticker", key="p_ticker")
        p_shares = st.number_input("Shares", min_value=1, value=10)
        p_price = st.number_input("Purchase Price", min_value=0.01, value=100.0)
        
        if st.button("➕ Add to Portfolio"):
            if p_ticker:
                try:
                    current = yf.Ticker(p_ticker.upper()).info.get('currentPrice', p_price)
                    value = p_shares * current
                    return_amt = (current - p_price) * p_shares
                    return_pct = ((current - p_price) / p_price) * 100
                    
                    st.session_state.portfolio[p_ticker.upper()] = {
                        'shares': p_shares,
                        'purchase_price': p_price,
                        'current_price': current,
                        'value': value,
                        'return_amt': return_amt,
                        'return_pct': return_pct
                    }
                    st.success(f"Added {p_ticker.upper()} to portfolio")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error adding position: {str(e)}")
    
    with col2:
        if st.session_state.portfolio:
            # Portfolio Summary
            total_value = sum(pos['value'] for pos in st.session_state.portfolio.values())
            total_return = sum(pos['return_amt'] for pos in st.session_state.portfolio.values())
            total_return_pct = (total_return / (total_value - total_return)) * 100 if total_value > total_return else 0
            
            st.markdown("### Portfolio Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Value", f"${total_value:,.2f}")
            with col2:
                st.metric("Total Return", f"${total_return:+,.2f}")
            with col3:
                st.metric("Return %", f"{total_return_pct:+.1f}%")
            with col4:
                st.metric("Positions", len(st.session_state.portfolio))
            
            # Portfolio Table
            portfolio_df = pd.DataFrame.from_dict(st.session_state.portfolio, orient='index')
            portfolio_df.index.name = 'Ticker'
            st.dataframe(
                portfolio_df.style.format({
                    'purchase_price': '${:.2f}',
                    'current_price': '${:.2f}',
                    'value': '${:,.2f}',
                    'return_amt': '${:+,.2f}',
                    'return_pct': '{:+.1f}%'
                }).background_gradient(subset=['return_pct'], cmap='RdYlGn'),
                use_container_width=True
            )
            
            # Portfolio Charts
            fig_pie, fig_bar = ChartGenerator.create_portfolio_charts(st.session_state.portfolio)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # AI Portfolio Analysis
            if st.button("🤖 Get AI Portfolio Optimization"):
                with st.spinner("Analyzing portfolio..."):
                    optimization = ai_advisor.portfolio_optimization(st.session_state.portfolio)
                    st.info(optimization)
            
            # Clear Portfolio
            if st.button("🗑️ Clear Portfolio"):
                st.session_state.portfolio = {}
                st.rerun()
        else:
            st.info("No positions in portfolio. Add positions to get started!")

def render_smart_screener():
    """Render Smart Screener"""
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); border-radius: 10px;'>
        <h2 style='color: white; text-align: center;'>🔍 Intelligent Stock Screener</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Screener filters
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        min_market_cap = st.number_input("Min Market Cap (B)", min_value=0, value=10) * 1e9
        min_pe = st.number_input("Min P/E Ratio", min_value=0, value=0)
    
    with col2:
        max_market_cap = st.number_input("Max Market Cap (B)", min_value=0, value=1000) * 1e9
        max_pe = st.number_input("Max P/E Ratio", min_value=0, value=50)
    
    with col3:
        min_dividend = st.number_input("Min Dividend Yield (%)", min_value=0.0, value=0.0)
        min_volume = st.number_input("Min Volume (M)", min_value=0, value=1) * 1e6
    
    with col4:
        sector = st.selectbox("Sector", ["Any", "Technology", "Healthcare", "Finance", "Consumer", "Energy", "Industrial"])
        
        if st.button("🔍 Run Screener"):
            with st.spinner("Screening stocks..."):
                # Sample screening with top stocks
                tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'JNJ', 'V', 'WMT', 'PG']
                results = []
                
                progress = st.progress(0)
                for i, ticker in enumerate(tickers):
                    try:
                        stock = yf.Ticker(ticker)
                        info = stock.info
                        
                        # Check filters
                        market_cap = info.get('marketCap', 0)
                        pe_ratio = info.get('trailingPE', 0)
                        dividend_yield = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
                        volume = info.get('volume', 0)
                        stock_sector = info.get('sector', '')
                        
                        if (min_market_cap <= market_cap <= max_market_cap and
                            min_pe <= pe_ratio <= max_pe and
                            dividend_yield >= min_dividend and
                            volume >= min_volume and
                            (sector == "Any" or stock_sector == sector)):
                            
                            results.append({
                                'Ticker': ticker,
                                'Company': info.get('longName', ticker),
                                'Price': info.get('currentPrice', 0),
                                'Market Cap (B)': market_cap / 1e9,
                                'P/E': pe_ratio,
                                'Dividend Yield': dividend_yield,
                                'Sector': stock_sector
                            })
                    except:
                        pass
                    
                    progress.progress((i + 1) / len(tickers))
                
                if results:
                    st.success(f"Found {len(results)} stocks matching criteria")
                    results_df = pd.DataFrame(results)
                    st.dataframe(
                        results_df.style.format({
                            'Price': '${:.2f}',
                            'Market Cap (B)': '${:.2f}B',
                            'P/E': '{:.2f}',
                            'Dividend Yield': '{:.2f}%'
                        }).background_gradient(subset=['P/E'], cmap='YlOrRd'),
                        use_container_width=True
                    )
                else:
                    st.warning("No stocks found matching criteria")

def render_market_insights(ai_advisor):
    """Render Market Insights"""
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px;'>
        <h2 style='color: white; text-align: center;'>📰 Market Intelligence & News</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Market sentiment indicators
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style='padding: 15px; background: linear-gradient(135deg, #667eea, #764ba2); border-radius: 10px;'>
            <h3 style='color: white; text-align: center;'>Market Sentiment</h3>
            <h1 style='color: #00ff00; text-align: center;'>BULLISH</h1>
            <p style='color: white; text-align: center;'>VIX: 15.3 📉</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='padding: 15px; background: linear-gradient(135deg, #f093fb, #f5576c); border-radius: 10px;'>
            <h3 style='color: white; text-align: center;'>Fear & Greed</h3>
            <h1 style='color: #ffa500; text-align: center;'>65</h1>
            <p style='color: white; text-align: center;'>Greed</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='padding: 15px; background: linear-gradient(135deg, #4facfe, #00f2fe); border-radius: 10px;'>
            <h3 style='color: white; text-align: center;'>Market Trend</h3>
            <h1 style='color: white; text-align: center;'>⬆️</h1>
            <p style='color: white; text-align: center;'>Uptrend</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Market news and insights
    st.markdown("### 📰 Latest Market News")
    
    news_items = [
        {"title": "Fed Signals Potential Rate Cuts in 2025", "sentiment": "Positive", "impact": "High"},
        {"title": "Tech Earnings Beat Expectations", "sentiment": "Positive", "impact": "Medium"},
        {"title": "Oil Prices Surge on Supply Concerns", "sentiment": "Negative", "impact": "Medium"},
        {"title": "AI Stocks Continue Rally", "sentiment": "Positive", "impact": "High"},
        {"title": "Inflation Data Shows Cooling Trend", "sentiment": "Positive", "impact": "High"}
    ]
    
    for item in news_items:
        sentiment_color = "#00ff00" if item['sentiment'] == "Positive" else "#ff3366"
        impact_emoji = "🔥" if item['impact'] == "High" else "⚡"
        
        st.markdown(f"""
        <div style='padding: 10px; margin: 5px 0; background: rgba(26, 26, 46, 0.8); border-left: 4px solid {sentiment_color}; border-radius: 5px;'>
            <strong>{item['title']}</strong> {impact_emoji}<br>
            <span style='color: {sentiment_color};'>{item['sentiment']}</span> | Impact: {item['impact']}
        </div>
        """, unsafe_allow_html=True)

def render_trading_signals(ai_advisor):
    """Render Live Trading Signals"""
    st.markdown("""
    <div style='padding: 20px; background: linear-gradient(135deg, #f43b47 0%, #453a94 100%); border-radius: 10px;'>
        <h2 style='color: white; text-align: center;'>⚡ Real-Time Trading Signals</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate sample trading signals
    signals = [
        {"ticker": "AAPL", "signal": "BUY", "confidence": 85, "target": 195.00, "stop": 175.00},
        {"ticker": "TSLA", "signal": "HOLD", "confidence": 60, "target": 250.00, "stop": 200.00},
        {"ticker": "NVDA", "signal": "BUY", "confidence": 92, "target": 550.00, "stop": 480.00},
        {"ticker": "AMZN", "signal": "SELL", "confidence": 75, "target": 170.00, "stop": 190.00},
        {"ticker": "GOOGL", "signal": "BUY", "confidence": 88, "target": 160.00, "stop": 145.00}
    ]
    
    for signal in signals:
        signal_color = "#00ff00" if signal['signal'] == "BUY" else "#ff3366" if signal['signal'] == "SELL" else "#ffa500"
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.markdown(f"<h3 style='color: {signal_color};'>{signal['ticker']}</h3>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<h4 style='color: {signal_color};'>{signal['signal']}</h4>", unsafe_allow_html=True)
        with col3:
            st.metric("Confidence", f"{signal['confidence']}%")
        with col4:
            st.metric("Target", f"${signal['target']:.2f}")
        with col5:
            st.metric("Stop Loss", f"${signal['stop']:.2f}")
        
        st.markdown("---")
    
    if st.button("🤖 Generate AI Trading Signals"):
        with st.spinner("Analyzing market conditions..."):
            for ticker in ['AAPL', 'GOOGL', 'MSFT']:
                df, info = MarketData.get_stock_data(ticker, "1mo")
                if not df.empty:
                    analysis = ai_advisor.analyze_stock(ticker, df, info)
                    st.info(f"**{ticker} Analysis:**\n{analysis}")

if __name__ == "__main__":
    main()