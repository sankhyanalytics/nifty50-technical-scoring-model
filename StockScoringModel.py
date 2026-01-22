# ========================
# Importing Libraries
# ========================
import os
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse

from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.linear_model import LinearRegression

# ========================
# FastAPI App
# ========================
app = FastAPI(
    title="Nifty50 Technical Scoring API",
    version="1.0"
)

CSV_PATH = "N50.csv"
LOOKBACK_DAYS = 365


# ========================
# OBV Slope via Linear Regression
# ========================
def compute_obv_slope(obv_series, window=7):
    if len(obv_series) < window:
        return 0.0

    y = obv_series[-window:].values.reshape(-1, 1)
    X = np.arange(window).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)
    return float(model.coef_[0][0])


# ========================
# Core Pipeline
# ========================
def build_summary_df():
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=LOOKBACK_DAYS)

    stocklist = pd.read_csv(CSV_PATH)
    symbols = stocklist["Symbol"].unique()
    symbols_ns = [s + ".NS" for s in symbols]

    stockdata = yf.download(
        tickers=symbols_ns,
        start=start_date,
        end=end_date,
        progress=False,
        threads=True
    )

    close_df = stockdata["Close"].copy()
    close_df.columns = close_df.columns.str.replace(".NS", "", regex=False)
    returns_df = close_df.pct_change() * 100

    rsi, macd_h, adx = {}, {}, {}
    aroon_up, aroon_down, obv_slope = {}, {}, {}

    for stock in close_df.columns:
        df = pd.DataFrame({
            "Close": stockdata["Close"][stock + ".NS"],
            "High": stockdata["High"][stock + ".NS"],
            "Low": stockdata["Low"][stock + ".NS"],
            "Volume": stockdata["Volume"][stock + ".NS"],
        }).dropna()

        if df.empty:
            continue

        rsi[stock] = RSIIndicator(df["Close"], 14).rsi().iloc[-1]
        macd_h[stock] = MACD(df["Close"], 12, 26, 9).macd_diff().iloc[-1]
        adx[stock] = ADXIndicator(df["High"], df["Low"], df["Close"], 14).adx().iloc[-1]

        aroon = AroonIndicator(df["High"], df["Low"], 25)
        aroon_up[stock] = aroon.aroon_up().iloc[-1]
        aroon_down[stock] = aroon.aroon_down().iloc[-1]

        obv_series = OnBalanceVolumeIndicator(
            df["Close"], df["Volume"]
        ).on_balance_volume()

        obv_slope[stock] = compute_obv_slope(obv_series)

    # ========================
    # Build Summary DF (aligned)
    # ========================
    summary_df = pd.DataFrame(index=close_df.columns)

    summary_df["Stock"] = summary_df.index
    summary_df["Max_Value"] = close_df.max()
    summary_df["Min_Value"] = close_df.min()
    summary_df["Current_Value"] = close_df.iloc[-1]
    summary_df["Avg_Daily_Return_%"] = returns_df.mean()
    summary_df["Volatility_%"] = returns_df.std()

    summary_df["RSI"] = pd.Series(rsi)
    summary_df["MACD_Hist"] = pd.Series(macd_h)
    summary_df["ADX"] = pd.Series(adx)
    summary_df["Aroon_Up"] = pd.Series(aroon_up)
    summary_df["Aroon_Down"] = pd.Series(aroon_down)
    summary_df["OBV_Slope"] = pd.Series(obv_slope)

    # Remove dummy / invalid stocks
    summary_df = summary_df.dropna(
        subset=["Current_Value", "RSI", "MACD_Hist", "ADX", "OBV_Slope"]
    )

    summary_df = summary_df.reset_index(drop=True)

    # ========================
    # Scoring Logic
    # ========================
    summary_df["RSI_Score"] = (
        (summary_df["RSI"] >= 40) & (summary_df["RSI"] <= 60)
    ).astype(int)

    summary_df["MACD_Score"] = (summary_df["MACD_Hist"] > 0).astype(int)
    summary_df["ADX_Score"] = (summary_df["ADX"] >= 25).astype(int)

    summary_df["Aroon_Score"] = (
        (summary_df["Aroon_Up"] > 70) &
        (summary_df["Aroon_Down"] < 30)
    ).astype(int)

    summary_df["OBV_Score"] = (summary_df["OBV_Slope"] > 0).astype(int)

    summary_df["Technical_Score"] = summary_df[
        ["RSI_Score", "MACD_Score", "ADX_Score", "Aroon_Score", "OBV_Score"]
    ].sum(axis=1)

    def interpret(score):
        if score == 5:
            return "Strong Buy"
        elif score == 4:
            return "Buy / High-priority Watchlist"
        elif score == 3:
            return "Watchlist"
        return "Ignore"

    summary_df["Score_Interpretation"] = summary_df["Technical_Score"].apply(interpret)

    return summary_df


# ========================
# HOME – Button Page
# ========================
@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
    <head>
        <title>Nifty50 Technical Scoring</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                padding: 40px;
            }
            button {
                padding: 14px 28px;
                font-size: 16px;
                cursor: pointer;
            }
        </style>
    </head>
    <body>
        <h2>Nifty50 Technical Scoring Model</h2>
        <form action="/run" method="get">
            <button type="submit">Load Summary</button>
        </form>
    </body>
    </html>
    """


# ========================
# RUN – Compute & Show Summary
# ========================
@app.get("/run", response_class=HTMLResponse)
def show_summary(response: Response):
    try:
        response.headers["Cache-Control"] = "no-store"

        df = build_summary_df()
        last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        table_html = df.to_html(index=False, border=0, justify="center")

        return f"""
        <html>
        <head>
            <title>Nifty50 Summary</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    padding: 20px;
                }}

                .table-container {{
                    max-height: 600px;
                    overflow-y: auto;
                    border: 1px solid #ccc;
                }}

                table {{
                    border-collapse: collapse;
                    width: 100%;
                    font-size: 13px;
                }}

                th, td {{
                    border: 1px solid #ddd;
                    padding: 6px 8px;
                    text-align: center;
                    white-space: nowrap;
                }}

                th {{
                    background-color: #f4f6f8;
                    position: sticky;
                    top: 0;
                }}

                tr:nth-child(even) {{
                    background-color: #fafafa;
                }}
            </style>
        </head>
        <body>
            <h2>Nifty50 Technical Scoring Summary</h2>
            <p><b>Last Updated:</b> {last_updated}</p>

            <div class="table-container">
                {table_html}
            </div>

            <br>
            <a href="/">⬅ Back</a>
        </body>
        </html>
        """
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ========================
# Health Check
# ========================
@app.get("/health")
def health():
    return {"status": "ok"}


# ========================
# Local / Render Entrypoint
# ========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("StockScoringModel:app", host="0.0.0.0", port=port)
