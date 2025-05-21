from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime
from dateutil.parser import parse as parse_date
from sqlalchemy import create_engine

app = Flask(__name__)
app.secret_key = "supersecretkey"

# ========== DATA FUNCTIONS ==========

def fetch_data_from_db():
    engine = create_engine('mysql+pymysql://root:password1234@localhost/goober_schema')
    query = """
        SELECT 
            Date, blocknumber, BuySell,
            `Purchase offer`, `Sell offer`,
            `Quantity`, `Price per unit`, `Invoice`
        FROM electricity_bids2;
    """
    df = pd.read_sql(query, engine)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(how='all', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce').dt.normalize()
    return df

df = fetch_data_from_db()
print("Available dates in database:", df['Date'].dt.date.unique())

# ========== CALCULATION FUNCTIONS ==========

def calculate_weighted_average(filtered_df):
    """Calculate weighted average as Total Invoice / Total Quantity"""
    if 'Invoice' not in filtered_df.columns or 'Quantity' not in filtered_df.columns:
        return None
    total_invoice = filtered_df['Invoice'].sum()
    total_quantity = filtered_df['Quantity'].sum()
    if total_quantity == 0:
        return None
    return total_invoice / total_quantity

def calculate_metrics(filtered_df):
    """Pre-calculate all important metrics"""
    buy_orders = filtered_df[filtered_df['BuySell'] == 'B']
    sell_orders = filtered_df[filtered_df['BuySell'] == 'S']

    metrics = {
        'total_purchase': buy_orders['Purchase offer'].sum(),
        'total_sell': sell_orders['Sell offer'].sum(),
        'total_quantity': filtered_df['Quantity'].sum(),
        'avg_price': filtered_df['Price per unit'].mean(),
        'weighted_avg_price': calculate_weighted_average(filtered_df),
        'min_price': filtered_df['Price per unit'].min(),
        'max_price': filtered_df['Price per unit'].max(),
        'total_invoice': filtered_df['Invoice'].sum(),
        'buy_order_count': len(buy_orders),
        'sell_order_count': len(sell_orders),
        'record_count': len(filtered_df)
    }
    return {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in metrics.items()}

# ========== CLOSEST MATCH HELPER ==========

def find_closest_rows(df, query):
    """Detect 'closest' or 'nearest' queries and precompute matching rows"""
    query_l = query.lower()
    results = []
    try:
        if 'closest' in query_l or 'nearest' in query_l:
            top_n = int(re.search(r'top\s+(\d+)', query_l).group(1)) if re.search(r'top\s+(\d+)', query_l) else 1
            # select column
            if 'invoice' in query_l:
                col = 'Invoice'
            elif 'sell price' in query_l or 'sell offer' in query_l:
                col = 'Sell offer'
            elif 'purchase price' in query_l or 'purchase offer' in query_l:
                col = 'Purchase offer'
            else:
                return results
            # reference value or average
            if 'average' in query_l:
                ref_value = df[col].mean()
            else:
                m = re.search(r'(\d+(?:\.\d+)?)', query_l)
                ref_value = float(m.group(1)) if m else None
            if ref_value is None:
                return results
            diffs = (df[col] - ref_value).abs()
            closest = df.loc[diffs.nsmallest(top_n).index]
            for _, row in closest.iterrows():
                results.append(f"{col}={row[col]} (block {row['blocknumber']}) is closest to {ref_value}")
    except Exception as e:
        print(f"[Closest Match Error]: {e}")
    return results

# ========== HOUR-BASED PARSING ==========

def parse_hour_range_from_query(query):
    """Map natural language hours (e.g., '10am', 'hour 15', 'morning') to block ranges"""
    query_l = query.lower()
    # map named periods
    period_map = {
        'morning': (6, 12), 'afternoon': (12, 18),
        'evening': (18, 24), 'night': (0, 6)
    }
    # explicit hour expressions
    h_match = re.search(r'(\d{1,2})(am|pm)?', query_l)
    if h_match:
        h = int(h_match.group(1))
        if h_match.group(2) == 'pm' and h < 12:
            h += 12
        start_block = h * 4
        return (start_block, start_block + 3)
    for period, (h1, h2) in period_map.items():
        if period in query_l:
            return (h1 * 4, h2 * 4 - 1)
    return None

# ========== DATE PARSING ==========

def extract_date_filters(query, df):
    """Parse explicit dates and date ranges (numeric and word forms)"""
    query_l = query.lower()
    # numeric dates
    date_matches = re.findall(r"(\d{1,2}[\/\-.]\d{1,2}(?:[\/\-.]\d{2,4})?)", query_l)
    # word dates
    date_words = re.findall(
        r"(?:\b(\d{1,2})(?:st|nd|rd|th)?\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b)|"
        r"(?:\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+(\d{1,2})(?:st|nd|rd|th)?\b)",
        query_l
    )
    def find_year(month, day=1):
        month_df = df[df['Date'].dt.month == month]
        if not month_df.empty:
            return int(month_df['Date'].dt.year.mode()[0])
        return datetime.now().year

    start_date = end_date = None
    # range case
    if any(token in query_l for token in ['to', 'until', 'through', 'and']):
        if len(date_matches) >= 2:
            d1 = parse_date(date_matches[0], dayfirst=True)
            d2 = parse_date(date_matches[1], dayfirst=True)
            if d1.year == 1900:
                d1 = d1.replace(year=find_year(d1.month, d1.day))
            if d2.year == 1900:
                d2 = d2.replace(year=find_year(d2.month, d2.day))
            start_date, end_date = d1, d2
        elif len(date_words) >= 2:
            parsed = []
            for m in date_words:
                day = m[0] or m[3]
                month_str = m[1] or m[2]
                if day and month_str:
                    month = parse_date(month_str, fuzzy=True).month
                    year = find_year(month, int(day))
                    parsed.append(parse_date(f"{day} {month} {year}"))
            if len(parsed) >= 2:
                start_date, end_date = parsed[0], parsed[1]
    # single date
    elif len(date_matches) == 1:
        d = parse_date(date_matches[0], dayfirst=True)
        if d.year == 1900:
            d = d.replace(year=find_year(d.month, d.day))
        start_date = end_date = d
    elif len(date_words) == 1:
        m = date_words[0]
        day = m[0] or m[3]
        month_str = m[1] or m[2]
        month = parse_date(month_str, fuzzy=True).month
        year = find_year(month, int(day))
        d = parse_date(f"{day} {month} {year}")
        start_date = end_date = d
    return start_date, end_date

# ========== FILTER FUNCTIONS ==========

def parse_block_range_from_query(query):
    m = re.search(r'blocks? (\d+)\s*(?:to|-|through)\s*(\d+)', query.lower())
    return (int(m.group(1)), int(m.group(2))) if m else None

def extract_filters_from_query(query, df):
    query_l = query.lower()
    # dates
    start_date, end_date = extract_date_filters(query_l, df)
    # blocks or hours
    block_range = parse_block_range_from_query(query_l)
    hour_range = parse_hour_range_from_query(query_l)
    if hour_range:
        block_range = hour_range
    # buy/sell
    buy_sell = None
    if 'buy' in query_l or 'purchase' in query_l:
        buy_sell = 'B'
    elif 'sell' in query_l:
        buy_sell = 'S'
    return {
        'start_date': start_date,
        'end_date': end_date,
        'block_range': block_range,
        'buy_sell': buy_sell
    }

# ========== LLM INTERACTION & ROUTES ==========

LMSTUDIO_API_URL = 'http://127.0.0.1:1234/v1/chat/completions'

def create_llm_prompt(user_query, metrics, sample_data, closest_matches=None):
    prompt = f"""
**Electricity Market Data Analysis**

**Metrics:**
- Total Purchase: {metrics['total_purchase']}
- Total Sell: {metrics['total_sell']}
- Total Qty: {metrics['total_quantity']}
- Avg Price: {metrics['avg_price']}
- Weighted Avg Price: {metrics['weighted_avg_price']}
- Price Range: {metrics['min_price']} - {metrics['max_price']}

**Closest Matches:**
"""
    if closest_matches:
        for m in closest_matches:
            prompt += f"- {m}\n"
    else:
        prompt += "- None\n"
    prompt += f"""
**Sample (3 rows):**
{sample_data.head(3).to_markdown()}

**Question:** {user_query}
**Note:** 1 block = 15 minutes; 4 blocks = 1 hour (0-2879)

**Answer Format:**
Answer:
Calculation:
Note:
"""
    return prompt

def query_llm_service(prompt):
    res = requests.post(
        LMSTUDIO_API_URL,
        json={"model":"local-model","messages":[{"role":"user","content":prompt}],"temperature":0.1,"max_tokens":500},
        timeout=70
    )
    res.raise_for_status()
    return res.json()['choices'][0]['message']['content']

@app.route('/query', methods=['POST'])
def handle_query():
    user_query = request.json.get('query','').strip()
    if not user_query:
        return jsonify({'error':'No query'}),400

    filters = extract_filters_from_query(user_query, df)
    filtered = df.copy()
    if filters['start_date'] and filters['end_date']:
        filtered = filtered[(filtered['Date']>=filters['start_date'])&(filtered['Date']<=filters['end_date'])]
    if filters['block_range']:
        sb, eb = filters['block_range']
        filtered = filtered[(filtered['blocknumber']>=sb)&(filtered['blocknumber']<=eb)]
    if filters['buy_sell']:
        filtered = filtered[filtered['BuySell']==filters['buy_sell']]
    if filtered.empty:
        available = df['Date'].dt.strftime('%Y-%m-%d').unique()[-5:]
        return jsonify({'answer':f"No data. Available: {', '.join(available)}"})

    metrics = calculate_metrics(filtered)
    closest = find_closest_rows(filtered, user_query)
    sample = filtered[['blocknumber','Date','BuySell','Quantity','Price per unit','Invoice']]
    prompt = create_llm_prompt(user_query, metrics, sample, closest)
    answer = query_llm_service(prompt)

    return jsonify({'answer':answer,'metrics':metrics,'filters':filters,'closest_matches':closest})

@app.route('/')
def index(): return render_template('index2.html')

if __name__=='__main__':
    app.run(debug=True)
