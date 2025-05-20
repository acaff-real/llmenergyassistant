from flask import Flask, request, jsonify, render_template, session
import pandas as pd
import requests
import re
from datetime import datetime

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Required for session usage

CSV_PATH = "bingbong.csv"
df = pd.read_csv(CSV_PATH, skiprows=4)
df.dropna(axis=1, how='all', inplace=True)
df.dropna(how='all', inplace=True)

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df['Hour'] = pd.to_numeric(df['Hour'], errors='coerce')

LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1/chat/completions"
LMSTUDIO_MODEL_NAME = "deepseek-r1-distill-qwen-7b"

def query_lmstudio(prompt, history=None):
    messages = [{"role": "system", "content": "You are a helpful assistant for analyzing electricity market bid data. Be concise and only use the provided data."}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        LMSTUDIO_API_URL,
        headers={"Content-Type": "application/json"},
        json={
            "model": LMSTUDIO_MODEL_NAME,
            "messages": messages,
            "temperature": 0.4
        }
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def extract_filters_from_query(query):
    query = query.lower()

    # Date range
    range_match = re.search(r'from (\d{1,2}[-/]\d{1,2}[-/]\d{4}) to (\d{1,2}[-/]\d{1,2}[-/]\d{4})', query)
    if range_match:
        start_date = pd.to_datetime(range_match.group(1).replace('/', '-'), dayfirst=True)
        end_date = pd.to_datetime(range_match.group(2).replace('/', '-'), dayfirst=True)
    else:
        single_date = re.search(r'(\d{1,2}[-/]\d{1,2}[-/]\d{4})', query)
        start_date = end_date = pd.to_datetime(single_date.group(1).replace('/', '-'), dayfirst=True) if single_date else None

    # Extended natural hour ranges
    hour_ranges = {
        "morning": (6, 11),
        "afternoon": (12, 17),
        "evening": (18, 21),
        "night": (22, 23),   # Include 0-5 of next day
        "late night": (0, 5)
    }

    for term, (start, end) in hour_ranges.items():
        if term in query:
            return start_date, end_date, (start, end)

    # Numeric hour range
    hour_match = re.search(r'between (\d{1,2}) and (\d{1,2})', query)
    if hour_match:
        return start_date, end_date, (int(hour_match.group(1)), int(hour_match.group(2)))

    single_hour_match = re.search(r'hour\s*(\d{1,2})', query)
    if single_hour_match:
        h = int(single_hour_match.group(1))
        return start_date, end_date, (h, h)

    return start_date, end_date, None

def detect_question_type(query):
    query = query.lower()
    if "average" in query or "avg" in query:
        return "Average"
    elif "sum" in query or "total" in query:
        return "Sum"
    elif "maximum" in query or "highest" in query or "max" in query:
        return "Maximum"
    elif "minimum" in query or "lowest" in query or "min" in query:
        return "Minimum"
    elif "compare" in query or "difference" in query:
        return "Comparison"
    return "General"

@app.route('/query', methods=['POST'])
def query_llm():
    user_query = request.json.get("query")
    if not user_query:
        return jsonify({"error": "No query provided."}), 400

    start_date, end_date, hour_range = extract_filters_from_query(user_query)

    filtered_df = df.copy()

    if start_date and end_date:
        filtered_df = filtered_df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    if hour_range:
        start_hr, end_hr = hour_range
        if start_hr <= end_hr:
            filtered_df = filtered_df[(df['Hour'] >= start_hr) & (df['Hour'] <= end_hr)]
        else:
            # Wrap-around for night like 22-5
            filtered_df = filtered_df[(df['Hour'] >= start_hr) | (df['Hour'] <= end_hr)]

    if filtered_df.empty:
        return jsonify({"answer": "No data available for the specified filters.", "question_type": detect_question_type(user_query)})

    preferred_columns = ['Date', 'Hour', 'Purchase Bid (MW)', 'Sell Bid (MW)', 
                         'Final Scheduled Volume (MW)', 'MCP (Rs/MWh)']
    relevant_columns = [col for col in preferred_columns if col in filtered_df.columns]
    filtered_data = filtered_df[relevant_columns]

    # Convert all filtered data to string once (no chunking needed with the first option)
    data_string = filtered_data.to_string(index=False)

    session_history = session.get("chat_history", [])
    prompt = f"""
Here is electricity market bid data:\n\n{data_string}
User question: "{user_query}"
Answer concisely based on this data.
"""
    try:
        final_answer = query_lmstudio(prompt, history=session_history[-2:])
    except Exception as e:
        return jsonify({"error": f"LLM Error: {str(e)}"}), 500

    # Save to history
    session_history.append({"role": "user", "content": user_query})
    session_history.append({"role": "assistant", "content": final_answer.strip()})
    session["chat_history"] = session_history[-10:]  # Keep last 5 exchanges

    question_type = detect_question_type(user_query)

    return jsonify({"answer": final_answer.strip(), "question_type": question_type})

@app.route("/preview", methods=['GET'])
def preview_data():
    """Returns a small preview for the frontend UI."""
    preview = df.head(15).to_dict(orient="records")
    return jsonify(preview)

@app.route("/")
def home():
    return render_template("index2.html")

if __name__ == '__main__':
    app.run(debug=True)
