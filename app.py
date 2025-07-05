from flask import Flask, render_template, request, jsonify
from ml import (
    analyze_performance, get_meta_info, get_state_summary,
    battleground_faceoff, get_candidate_history, search_candidates, _DF
)
import numpy as np

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/meta")
def meta():
    years, _ = get_meta_info()
    return jsonify({"years": [int(y) for y in years]})

@app.route("/api/search_candidates")
def search_candidates_api():
    query = request.args.get("query", "")
    year = request.args.get("year", type=int)
    if not query:
        return jsonify([])
    matches = search_candidates(query, year)
    return jsonify(matches)

@app.route("/api/analyze")
def analyze_api():
    year = request.args.get("year", type=int)
    candidate = request.args.get("candidate", "").strip()
    if not year or not candidate:
        return jsonify({"status": "error", "message": "Year and Candidate Name are required."}), 400
    return jsonify(analyze_performance(year, candidate))

@app.route("/api/history")
def history_api():
    candidate = request.args.get("candidate", "").strip()
    if not candidate:
        return jsonify({"status": "error", "message": "Candidate Name is required."}), 400
    return jsonify(get_candidate_history(candidate))

@app.route("/api/top_performers")
def top_performers():
    year = request.args.get("year", type=int)
    if not year:
        return jsonify({"status": "error", "message": "Year is required."}), 400
    
    df = _DF[_DF["year"] == year].copy()
    
    df['overperformance_score'] = df['candidate_alpha_score']
    df = df.sort_values(by="overperformance_score", ascending=False).drop_duplicates(subset=['candidate'])

    # FIX: Select only the columns needed by the frontend to reduce data transfer
    # and explicitly handle NaN values which are invalid in JSON.
    display_cols = ['candidate', 'party', 'constituency_name', 'overperformance_score']
    
    top_df = df.head(15)[display_cols].replace({np.nan: None})
    bottom_df = df.tail(15).sort_values(by="overperformance_score", ascending=True)[display_cols].replace({np.nan: None})

    top = top_df.to_dict(orient="records")
    bottom = bottom_df.to_dict(orient="records")
    
    return jsonify({ "status": "success", "top_overperformers": top, "top_underperformers": bottom })


@app.route("/api/state_insights")
def state_insights():
    year = request.args.get("year", type=int)
    state = request.args.get("state", "").strip()
    if not year or not state:
        return jsonify({"status": "error", "message": "Year and State are required."}), 400
    return jsonify(get_state_summary(year, state))

@app.route("/api/battleground")
def battleground():
    year = request.args.get("year", type=int)
    state = request.args.get("state", "").strip()
    constituency = request.args.get("constituency", "").strip()
    cand1 = request.args.get("cand1", "").strip()
    cand2 = request.args.get("cand2", "").strip()
    if not all([year, state, constituency, cand1, cand2]):
        return jsonify({"status": "error", "message": "All fields are required."}), 400
    return jsonify(battleground_faceoff(year, state, constituency, cand1, cand2))

if __name__ == "__main__":
    app.run(debug=True)