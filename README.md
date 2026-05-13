# InvestEdge - Stock Recommendation & Trading Platform

## Overview
InvestEdge is an AI-powered stock market analysis and recommendation platform that helps users make data-driven investment decisions. The platform uses Machine Learning and Deep Learning techniques to analyze historical stock data, generate predictions, and provide buy/sell recommendations through an interactive dashboard.

The project is developed using Python, Streamlit, TensorFlow/Keras, and Yahoo Finance API.

---

## Features
- Real-time stock market data fetching
- LSTM-based stock price prediction
- Buy/Sell recommendation system
- Technical indicators:
  - RSI (Relative Strength Index)
  - MACD
  - Bollinger Bands
- Interactive charts and visualizations
- Future stock price prediction
- Portfolio and trade simulation
- Performance evaluation metrics

---

## Technologies Used
- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Yahoo Finance API

---

## Project Structure
```bash
InvestEdge/
│
├── comprehensive_stock_dashboard_updated.py
├── requirements.txt
├── README.md
└── trade_log.csv



Installation & Setup
1. Clone the Repository

git clone <repository-link>
cd InvestEdge


2. Install Dependencies

pip install -r requirements.txt

3. Run the Application

streamlit run app2.py


How It Works-
User enters a stock ticker symbol.
Stock market data is fetched using Yahoo Finance API.
Data is preprocessed and scaled.
LSTM model is trained on historical stock prices.
The model predicts future stock prices.
Technical indicators and charts are generated.
Buy/Sell recommendations are displayed.

Future Enhancements-
Integration with live trading APIs
Sentiment analysis using news and social media
Portfolio optimization
Multi-stock comparison
Cloud deployment support
