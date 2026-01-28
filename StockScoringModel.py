# ========================
# Importing Libraries
# ========================
import os
import json
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator, AroonIndicator
from ta.volume import OnBalanceVolumeIndicator
from sklearn.linear_model import LinearRegression

# ========================
# FastAPI App
# ========================
app = FastAPI(
    title="Nifty50 Technical Dashboard",
    version="4.2"
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
    stocklist["Symbol"] = stocklist["Symbol"].str.strip()
    stocklist["Sector"] = stocklist["Industry"].fillna("Unknown")

    sector_map = dict(zip(stocklist["Symbol"], stocklist["Sector"]))

    symbols = stocklist["Symbol"].unique()
    symbols_ns = [s + ".NS" for s in symbols]

    stockdata = yf.download(
        tickers=symbols_ns,
        start=start_date,
        end=end_date,
        progress=False,
        threads=True,
        auto_adjust=False
    )

    close_df = stockdata["Close"].copy()
    close_df.columns = close_df.columns.str.replace(".NS", "", regex=False)
    returns_df = close_df.pct_change(fill_method=None) * 100

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

    summary_df = pd.DataFrame(index=close_df.columns)
    summary_df["Stock"] = summary_df.index
    summary_df["Sector"] = summary_df["Stock"].map(sector_map).fillna("Unknown")

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

    summary_df = summary_df.dropna(
        subset=["Current_Value", "RSI", "MACD_Hist", "ADX", "OBV_Slope"]
    ).reset_index(drop=True)

    summary_df["RSI_Score"] = ((summary_df["RSI"] >= 40) & (summary_df["RSI"] <= 60)).astype(int)
    summary_df["MACD_Score"] = (summary_df["MACD_Hist"] > 0).astype(int)
    summary_df["ADX_Score"] = (summary_df["ADX"] >= 25).astype(int)
    summary_df["Aroon_Score"] = ((summary_df["Aroon_Up"] > 70) & (summary_df["Aroon_Down"] < 30)).astype(int)
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

    # ========================
    # ROUND ALL NUMERIC COLUMNS
    # ========================
    numeric_cols = summary_df.select_dtypes(include=[np.number]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].round(2)

    return summary_df


# ========================
# DASHBOARD ENDPOINT
# ========================
@app.get("/", response_class=HTMLResponse)
def dashboard():
    df = build_summary_df()

    display_df = df.drop(
        columns=["RSI_Score","MACD_Score","ADX_Score","Aroon_Score","OBV_Score"]
    )

    data = display_df.to_dict(orient="records")
    pie_map = df.groupby("Score_Interpretation")["Stock"].count().to_dict()

    unique_values = {
        col: sorted(display_df[col].astype(str).unique())
        for col in display_df.columns
    }

    score_col_index = list(display_df.columns).index("Score_Interpretation")

    return f"""
<!DOCTYPE html>
<html>
<head>
<title>Nifty50 Technical Dashboard</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>

<style>
body {{
    font-family: Arial;
    padding: 20px;
}}

select {{
    width: 100%;
    height: 110px;
}}

button {{
    padding: 6px 12px;
    margin: 8px 6px;
}}

#pie-container {{
    margin-bottom: 60px;
}}

#table-container {{
    margin-top: 60px;
}}

table {{
    border-collapse: collapse;
    width: 100%;
    font-size: 13px;
}}

th, td {{
    border: 1px solid #ddd;
    padding: 6px;
    text-align: center;
}}

th {{
    background: #f4f6f8;
    position: sticky;
    top: 0;
}}

tr:nth-child(even) {{
    background: #fafafa;
}}

.footer {{
    margin-top: 50px;
    padding: 16px 18px;
    background-color: #fff3cd;
    border-left: 6px solid #ff9800;
    font-size: 14px;
    font-weight: bold;
    color: #5a3e00;
    line-height: 1.6;
}}
</style>
</head>

<body>

<h2>Nifty50 Technical Scoring Dashboard</h2>

<p><b>Tip:</b> Hold <b>Ctrl</b> (Windows) or <b>Cmd</b> (Mac) to select multiple values</p>

<button onclick="resetFilters()">🔄 Reset Filters</button>
<button onclick="downloadCSV()">⬇ Download CSV</button>

<div id="pie-container">
    <div id="pie" style="height:420px;"></div>
</div>

<div id="table-container">
<table id="dataTable">
<thead>
<tr>
{''.join(f"<th>{c}</th>" for c in display_df.columns)}
</tr>
<tr>
{''.join(
    "<th><select multiple size='6' onchange='applyFilters()'>" +
    "".join(f"<option value='{v}'>{v}</option>" for v in unique_values[c]) +
    "</select></th>"
    for c in display_df.columns
)}
</tr>
</thead>

<tbody>
{''.join(
    "<tr>" + "".join(f"<td>{row[c]}</td>" for c in display_df.columns) + "</tr>"
    for row in data
)}
</tbody>
</table>
</div>

<div class="footer">
⚠ <b>IMPORTANT NOTE</b><br><br>
Technical Score ranges from 0–5 and is computed by adding the score for 5 technical features:
<br><br>
(1) RSI: 40 ≤ RSI ≤ 60 → Score = 1, else 0<br>
(2) MACD Histogram: &gt; 0 → Score = 1, else 0<br>
(3) ADX: &gt; 25 → Score = 1, else 0<br>
(4) Aroon: Up &gt; 70 AND Down &lt; 30 → Score = 1, else 0<br>
(5) OBV: Slope &gt; 0 → Score = 1, else 0
</div>

<script>
const rows = document.querySelectorAll("#dataTable tbody tr");
const filters = document.querySelectorAll("thead select");
const pieData = {json.dumps(pie_map)};
const tableData = {json.dumps(data)};

function applyFilters() {{
  rows.forEach(row => {{
    let visible = true;
    filters.forEach((sel, i) => {{
      const selected = Array.from(sel.selectedOptions).map(o => o.value);
      if (selected.length && !selected.includes(row.cells[i].innerText)) {{
        visible = false;
      }}
    }});
    row.style.display = visible ? "" : "none";
  }});
}}

function resetFilters() {{
  filters.forEach(sel => Array.from(sel.options).forEach(o => o.selected = false));
  rows.forEach(r => r.style.display = "");
}}

function downloadCSV() {{
  let csv = Object.keys(tableData[0]).join(",") + "\\n";
  tableData.forEach(r => {{
    csv += Object.values(r).join(",") + "\\n";
  }});
  const blob = new Blob([csv], {{ type: "text/csv" }});
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "nifty50_summary.csv";
  a.click();
}}

Plotly.newPlot("pie", [{{
  type: "pie",
  labels: Object.keys(pieData),
  values: Object.values(pieData),
  hovertemplate:
    "<b>%{{label}}</b><br>" +
    "Stocks: %{{value}}<br>" +
    "<extra></extra>"
}}]);

document.getElementById("pie").on("plotly_click", function(d) {{
  const category = d.points[0].label;
  const sel = filters[{score_col_index}];
  Array.from(sel.options).forEach(o => o.selected = false);
  Array.from(sel.options)
    .filter(o => o.value === category)
    .forEach(o => o.selected = true);
  applyFilters();
}});
</script>

</body>
</html>
"""
    

# ========================
# Entrypoint
# ========================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("StockScoringModel:app", host="0.0.0.0", port=port)
