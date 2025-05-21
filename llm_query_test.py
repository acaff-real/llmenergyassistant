from flask import Flask, request, jsonify, render_template, session
import pandas as pd
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
            Date, Hour, 
            `Purchase Bid (MW)`, `Sell Bid (MW)`, 
            `Final Scheduled Volume (MW)`, `MCP (Rs/MWh)`, 
            `Purchase Bid Price (Rs/MWh)`, `Sell Bid Price (Rs/MWh)`
        FROM electricity_bids;
    """
    df = pd.read_sql(query, engine)
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(how='all', inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce').dt.normalize()
    df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')
    return df

df = fetch_data_from_db()
print("Available dates in database:", df['Date'].dt.date.unique())

# ========== LLM CONFIGURATION ==========

LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_MODEL_NAME = "mathstral-7b-v0.1"

# ========== IMPROVED DATA PROCESSING ==========

def prepare_llm_data_package(filtered_df):
    """Create a structured data package with pre-calculated metrics"""
    metrics = {
        'total_purchase': round(filtered_df['Purchase Bid (MW)'].sum(), 2),
        'total_sell': round(filtered_df['Sell Bid (MW)'].sum(), 2),
        'avg_mcp': round(filtered_df['MCP (Rs/MWh)'].mean(), 2),
        'min_mcp': round(filtered_df['MCP (Rs/MWh)'].min(), 2),
        'max_mcp': round(filtered_df['MCP (Rs/MWh)'].max(), 2),
        'record_count': len(filtered_df)
    }
    
    # Sample records (first and last 3)
    sample_records = pd.concat([
        filtered_df.head(3),
        filtered_df.tail(3)
    ]).to_dict('records')
    
    return {
        'metrics': metrics,
        'sample_records': sample_records,
        'date_range': {
            'start': filtered_df['Date'].min().strftime('%Y-%m-%d'),
            'end': filtered_df['Date'].max().strftime('%Y-%m-%d')
        }
    }

def create_llm_prompt(user_query, data_package):
    """Generate a structured prompt with clear instructions"""
    return f"""
**Electricity Market Data Analysis**

**Date Range:** {data_package['date_range']['start']} to {data_package['date_range']['end']}
**Total Records:** {data_package['metrics']['record_count']}

**Pre-Calculated Metrics:**
- Total Purchase Bid: {data_package['metrics']['total_purchase']} MW
- Total Sell Bid: {data_package['metrics']['total_sell']} MW
- MCP Range: {data_package['metrics']['min_mcp']} to {data_package['metrics']['max_mcp']} Rs/MWh
- Average MCP: {data_package['metrics']['avg_mcp']} Rs/MWh

**Sample Data (First & Last 3 Records):**
{data_package['sample_records']}

**User Question:** "{user_query}"

**Instructions:**
1. Use ONLY the provided data
2. For calculations, reference these verified metrics:
   - Total Purchase: {data_package['metrics']['total_purchase']} MW
   - Total Sell: {data_package['metrics']['total_sell']} MW
3. Respond with this format:
   **Answer:** [concise response]
   **Data Used:** [specific values referenced]
   **Calculation:** [if applicable]
"""

# ========== CORE FUNCTIONALITY ==========

def query_lmstudio(prompt):
    """Enhanced LLM query with better error handling"""
    try:
        response = requests.post(
            LMSTUDIO_API_URL,
            headers={"Content-Type": "application/json"},
            json={
                "model": LMSTUDIO_MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,  # Lower for more deterministic answers
                "max_tokens": 1500,
                "stop": ["**"]  # Helps maintain structured responses
            },
            timeout=500
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
    except Exception as e:
        print(f"LLM Error: {str(e)}")
        raise Exception("Failed to process query with LLM")

# ========== (Keep all your existing filter/parse functions as-is) ==========
# [find_available_year, parse_hour_range_from_query, extract_filters_from_query, detect_question_type]

# ========== ROUTE HANDLERS ==========

@app.route('/query', methods=['POST'])
def query_llm():
    try:
        user_query = request.json.get("query")
        if not user_query:
            return jsonify({"error": "No query provided."}), 400

        # Extract filters
        start_date, end_date, hour_range = extract_filters_from_query(user_query, df)
        print(f"Filters -> Date: {start_date} to {end_date}, Hours: {hour_range}")

        # Apply filters
        filtered_df = df.copy()
        if start_date and end_date:
            filtered_df = filtered_df[(filtered_df['Date'] >= start_date) & 
                                    (filtered_df['Date'] <= end_date)]
        
        if hour_range:
            start_hr, end_hr = hour_range
            if start_hr <= end_hr:
                filtered_df = filtered_df[(filtered_df['Hour'] >= start_hr) & 
                                         (filtered_df['Hour'] <= end_hr)]
            else:
                filtered_df = filtered_df[(filtered_df['Hour'] >= start_hr) | 
                                         (filtered_df['Hour'] <= end_hr)]

        # Check data availability
        if len(filtered_df) < 2:
            available_dates = df['Date'].dt.strftime('%Y-%m-%d').unique()
            return jsonify({
                "answer": f"Insufficient data. Available dates: {', '.join(sorted(available_dates)[-5:])}",
                "question_type": detect_question_type(user_query)
            })

        # Prepare data and query LLM
        data_package = prepare_llm_data_package(filtered_df)
        prompt = create_llm_prompt(user_query, data_package)
        
        print(f"Sending prompt to LLM:\n{prompt[:500]}...")  # Log partial prompt
        final_answer = query_lmstudio(prompt)

        return jsonify({
            "answer": final_answer,
            "question_type": detect_question_type(user_query),
            "metrics": data_package['metrics'],
            "filters_applied": {
                "start_date": start_date.strftime('%Y-%m-%d') if start_date else None,
                "end_date": end_date.strftime('%Y-%m-%d') if end_date else None,
                "hour_range": hour_range
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "message": "Analysis failed. Please check your query and try again."
        }), 500

@app.route('/')
def index():
    return render_template('index2.html')

if __name__ == "__main__":
    app.run(debug=True)