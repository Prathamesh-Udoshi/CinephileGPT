import streamlit as st
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
# Add the backend directory to sys.path so app and evaluation imports work
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from datetime import datetime
from sqlalchemy.orm import Session
from app.core.database import SessionLocal
from evaluation.models import EvaluationRun, EvaluationCaseResult

# Page layout setup
st.set_page_config(
    page_title="CinephileGPT Eval Center",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Glassmorphism & Curated Theme Palette)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    /* Main body styles */
    .reportview-container {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sleek card styling */
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        margin-bottom: 15px;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #9ea4b0;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 5px 0;
        background: linear-gradient(45deg, #e50914, #ff6b6b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-green {
        background: linear-gradient(45deg, #00c853, #b9f6ca) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    .metric-blue {
        background: linear-gradient(45deg, #29b6f6, #81d4fa) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }

    .metric-orange {
        background: linear-gradient(45deg, #ffa726, #ffe082) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    /* Custom headers */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to query DB or reports directory fallbacks
def load_historical_runs():
    db = SessionLocal()
    try:
        runs = db.query(EvaluationRun).order_by(EvaluationRun.id.desc()).all()
        # Convert to list of dicts for ease
        run_list = []
        for r in runs:
            run_list.append({
                "id": r.id,
                "created_at": r.created_at,
                "overall_score": r.overall_score,
                "pass_rate": r.pass_rate,
                "total_cases": r.total_cases,
                "passed_cases": r.passed_cases,
                "failed_cases": r.failed_cases,
                "recommendation_score": r.recommendation_score,
                "personality_score": r.personality_score,
                "memory_score": r.memory_score,
                "retrieval_score": r.retrieval_score,
                "refusal_score": r.refusal_score,
                "recommendation_pass_rate": r.recommendation_pass_rate,
                "personality_pass_rate": r.personality_pass_rate,
                "memory_pass_rate": r.memory_pass_rate,
                "retrieval_pass_rate": r.retrieval_pass_rate,
                "refusal_pass_rate": r.refusal_pass_rate,
                "status": r.status
            })
        return run_list
    except Exception as e:
        st.warning(f"Failed to query database run history: {e}")
        return []
    finally:
        db.close()

def load_case_results(run_id: int):
    db = SessionLocal()
    try:
        cases = db.query(EvaluationCaseResult).filter(EvaluationCaseResult.run_id == run_id).all()
        case_list = []
        for c in cases:
            case_list.append({
                "id": c.id,
                "case_id": c.case_id,
                "category": c.category,
                "difficulty": c.difficulty or "N/A",
                "query": c.query,
                "expected": c.expected,
                "actual_response": c.actual_response,
                "passed": c.passed,
                "score": c.score,
                "sub_scores": c.sub_scores or {},
                "strengths": c.strengths or [],
                "weaknesses": c.weaknesses or [],
                "reasoning": c.reasoning or ""
            })
        return case_list
    except Exception as e:
        st.error(f"Failed to query database test cases: {e}")
        return []
    finally:
        db.close()

# Sidebar Setup
st.sidebar.markdown("<h1 style='color: #e50914;'>🎬 CinephileGPT</h1>", unsafe_allow_html=True)
st.sidebar.markdown("### Evaluation Dashboard")

runs = load_historical_runs()

if not runs:
    st.sidebar.info("No runs found in PostgreSQL. Trying local reports directory fallback...")
    
    # Load fallback from json report if database is empty or not updated
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    summary_path = os.path.join(reports_dir, "evaluation_summary.json")
    results_path = os.path.join(reports_dir, "evaluation_results.json")
    
    if os.path.exists(summary_path) and os.path.exists(results_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)
            
        mock_run = {
            "id": 1,
            "created_at": datetime.fromisoformat(summary.get("timestamp", datetime.now().isoformat())),
            "overall_score": summary.get("overall_score"),
            "pass_rate": summary.get("pass_rate"),
            "total_cases": summary.get("total_cases"),
            "passed_cases": summary.get("passed_cases"),
            "failed_cases": summary.get("failed_cases"),
            "recommendation_score": summary.get("category_summary", {}).get("recommendation", {}).get("avg_score", 0.0),
            "personality_score": summary.get("category_summary", {}).get("personality", {}).get("avg_score", 0.0),
            "memory_score": summary.get("category_summary", {}).get("memory", {}).get("avg_score", 0.0),
            "retrieval_score": summary.get("category_summary", {}).get("retrieval", {}).get("avg_score", 0.0),
            "refusal_score": summary.get("category_summary", {}).get("refusal", {}).get("avg_score", 0.0),
            "recommendation_pass_rate": summary.get("category_summary", {}).get("recommendation", {}).get("pass_rate", 0.0),
            "personality_pass_rate": summary.get("category_summary", {}).get("personality", {}).get("pass_rate", 0.0),
            "memory_pass_rate": summary.get("category_summary", {}).get("memory", {}).get("pass_rate", 0.0),
            "retrieval_pass_rate": summary.get("category_summary", {}).get("retrieval", {}).get("pass_rate", 0.0),
            "refusal_pass_rate": summary.get("category_summary", {}).get("refusal", {}).get("pass_rate", 0.0),
            "status": "completed"
        }
        runs = [mock_run]
        selected_run = mock_run
        
        # Override results getter
        def load_case_results_mock(run_id):
            mock_cases = []
            for idx, c in enumerate(results):
                mock_cases.append({
                    "id": idx,
                    "case_id": c.get("case_id"),
                    "category": c.get("category"),
                    "difficulty": c.get("difficulty", "N/A"),
                    "query": c.get("query"),
                    "expected": c.get("expected", {}),
                    "actual_response": c.get("actual_response"),
                    "passed": c.get("passed"),
                    "score": c.get("score"),
                    "sub_scores": c.get("sub_scores", {}),
                    "strengths": c.get("strengths", []),
                    "weaknesses": c.get("weaknesses", []),
                    "reasoning": c.get("reasoning", "")
                })
            return mock_cases
        get_cases_func = load_case_results_mock
    else:
        st.error("No evaluation data found. Run a test first: `python evaluation/runner.py`")
        st.stop()
else:
    # Selected run from db selector
    run_options = {f"Run #{r['id']} ({r['created_at'].strftime('%Y-%m-%d %H:%M')})": r for r in runs}
    selected_label = st.sidebar.selectbox("Select Evaluation Run:", list(run_options.keys()))
    selected_run = run_options[selected_label]
    get_cases_func = load_case_results

# Load case results for selected run
case_results = get_cases_func(selected_run["id"])
df_cases = pd.DataFrame(case_results)

# Sidebar pages
page = st.sidebar.radio(
    "Navigation", 
    ["System Overview", "Historical Runs & Trends", "Category Diagnostics", "Test Case Investigator"]
)

# Extract aggregates for selected run
overall_score = selected_run["overall_score"]
pass_rate = selected_run["pass_rate"]
total_cases = selected_run["total_cases"]
passed_cases = selected_run["passed_cases"]
failed_cases = selected_run["failed_cases"]

category_scores = {
    "Recommendation": selected_run["recommendation_score"],
    "Personality": selected_run["personality_score"],
    "Memory": selected_run["memory_score"],
    "Retrieval": selected_run["retrieval_score"],
    "Refusal": selected_run["refusal_score"]
}

category_pass_rates = {
    "Recommendation": selected_run["recommendation_pass_rate"],
    "Personality": selected_run["personality_pass_rate"],
    "Memory": selected_run["memory_pass_rate"],
    "Retrieval": selected_run["retrieval_pass_rate"],
    "Refusal": selected_run["refusal_pass_rate"]
}

best_cat = max(category_scores.keys(), key=lambda k: category_scores[k] or 0.0)
worst_cat = min(category_scores.keys(), key=lambda k: category_scores[k] or 10.0)

# Page 1: System Overview
if page == "System Overview":
    st.markdown("## 📊 CinephileGPT LLMOps Evaluation Center")
    st.write(f"Showing performance diagnostics for **Run #{selected_run['id']}** (Executed: {selected_run['created_at'].strftime('%Y-%m-%d %H:%M:%S')})")
    
    # Large metrics cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Overall System Score</div>
            <div class="metric-value">{overall_score:.2f}/10.0</div>
            <div style="font-size: 0.8rem; color: #888;">LLM & retrieval blend</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Pass Rate</div>
            <div class="metric-value metric-green">{pass_rate*100:.1f}%</div>
            <div style="font-size: 0.8rem; color: #888;">{passed_cases} / {total_cases} test cases</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Best Category</div>
            <div class="metric-value metric-blue">{best_cat}</div>
            <div style="font-size: 0.8rem; color: #888;">Avg Score: {category_scores[best_cat]:.2f}/10</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Worst Category</div>
            <div class="metric-value metric-orange">{worst_cat}</div>
            <div style="font-size: 0.8rem; color: #888;">Avg Score: {category_scores[worst_cat]:.2f}/10</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Category Performance Table & Chart
    c_left, c_right = st.columns([3, 2])
    with c_left:
        st.markdown("### 📈 Category Breakdown")
        
        # Build category table
        cat_data = []
        for cat_name in category_scores.keys():
            db_cat_name = cat_name.lower()
            cat_df = df_cases[df_cases["category"] == db_cat_name]
            total_cat = len(cat_df)
            passed_cat = len(cat_df[cat_df["passed"] == True])
            failed_cat = total_cat - passed_cat
            
            cat_data.append({
                "Category": cat_name,
                "Avg Score (1-10)": round(category_scores[cat_name], 2),
                "Pass Rate (%)": f"{(category_pass_rates[cat_name] or 0.0)*100:.1f}%",
                "Total Cases": total_cat,
                "Passed": passed_cat,
                "Failed": failed_cat
            })
            
        st.table(pd.DataFrame(cat_data))
        
    with c_right:
        st.markdown("### 🏆 Performance radar")
        # Radar Chart
        categories = list(category_scores.keys())
        scores = [category_scores[c] for c in categories]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor='rgba(229, 9, 20, 0.2)',
            line=dict(color='#e50914', width=2),
            name='Category Score'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            showlegend=False,
            margin=dict(l=40, r=40, t=20, b=20),
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    # Failed cases sample preview
    st.markdown("### ❌ Sample Failures & Judge Feedback Preview")
    failed_df = df_cases[df_cases["passed"] == False]
    if failed_df.empty:
        st.success("🎉 All test cases passed in this run!")
    else:
        sample_fails = failed_df.head(3)
        for idx, row in sample_fails.iterrows():
            with st.expander(f"🔴 [{row['category'].upper()}] - {row['case_id']}: '{row['query'][:60]}...'"):
                st.markdown(f"**Query:** {row['query']}")
                st.markdown(f"**Score:** {row['score']:.1f}/10.0")
                st.markdown(f"**Actual Response:** {row['actual_response']}")
                st.markdown(f"**Judge Reasoning:** {row['reasoning']}")
                if row['weaknesses']:
                    st.markdown("**Weaknesses Detected:**")
                    for w in row['weaknesses']:
                        st.markdown(f"- {w}")

# Page 2: Historical Runs & Trends
elif page == "Historical Runs & Trends":
    st.markdown("## 📜 Historical Runs & Score Trends")
    
    if len(runs) < 2:
        st.info("Only one run exists in the history database. Trend lines will display as points.")
        
    df_runs = pd.DataFrame(runs)
    df_runs = df_runs.sort_values(by="id")
    
    # Trend Chart
    st.markdown("### 📈 Overall Score & Pass Rate Trends")
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_runs["id"],
        y=df_runs["overall_score"],
        mode='lines+markers',
        name='Overall Score',
        line=dict(color='#e50914', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=df_runs["id"],
        y=df_runs["pass_rate"] * 10,
        mode='lines+markers',
        name='Pass Rate (scaled x10)',
        line=dict(color='#00c853', width=3, dash='dash'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Performance Trends Over Runs",
        xaxis=dict(title="Run ID / Number", tickmode='linear'),
        yaxis=dict(title="Score (1-10 scale)", range=[0, 10.5]),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Category trends
    st.markdown("### 📊 Category Score Trends")
    fig_cat = go.Figure()
    
    colors = {
        "recommendation_score": "#29b6f6",
        "personality_score": "#ffa726",
        "memory_score": "#ab47bc",
        "retrieval_score": "#26a69a",
        "refusal_score": "#ec407a"
    }
    
    for score_col, label in [
        ("recommendation_score", "Recommendation"),
        ("personality_score", "Personality"),
        ("memory_score", "Memory"),
        ("retrieval_score", "Retrieval"),
        ("refusal_score", "Refusal")
    ]:
        fig_cat.add_trace(go.Scatter(
            x=df_runs["id"],
            y=df_runs[score_col],
            mode='lines+markers',
            name=label,
            line=dict(color=colors[score_col], width=2),
            marker=dict(size=6)
        ))
        
    fig_cat.update_layout(
        xaxis=dict(title="Run ID", tickmode='linear'),
        yaxis=dict(title="Score (1-10 scale)", range=[0, 10.5]),
        legend=dict(x=0, y=1.1, orientation="h"),
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig_cat, use_container_width=True)

    # Historic runs data grid
    st.markdown("### 📂 Run History Log")
    run_logs = df_runs[[
        "id", "created_at", "overall_score", "pass_rate", 
        "recommendation_score", "personality_score", "memory_score", "retrieval_score", "refusal_score"
    ]].copy()
    run_logs["created_at"] = run_logs["created_at"].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
    run_logs["pass_rate"] = run_logs["pass_rate"].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(run_logs.sort_values(by="id", ascending=False), use_container_width=True)

# Page 3: Category Diagnostics
elif page == "Category Diagnostics":
    st.markdown("## 🔍 Category-Specific Performance Diagnostics")
    
    selected_diag_cat = st.selectbox(
        "Select Category to Analyze:",
        ["Recommendation", "Personality", "Memory", "Retrieval", "Refusal"]
    )
    
    cat_db_name = selected_diag_cat.lower()
    df_cat_cases = df_cases[df_cases["category"] == cat_db_name]
    
    if df_cat_cases.empty:
        st.warning(f"No cases found in this run for category: {selected_diag_cat}")
    else:
        st.markdown(f"### Diagnostics details for {selected_diag_cat}")
        
        # Highlight metrics
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Category Avg Score", f"{category_scores[selected_diag_cat]:.2f}/10.0")
        with c2:
            st.metric("Category Pass Rate", f"{(category_pass_rates[selected_diag_cat] or 0.0)*100:.1f}%")
        with c3:
            st.metric("Total Cases Checked", len(df_cat_cases))
            
        st.markdown("---")
        
        # Sub-metrics breakdowns (average scores of judge scoring rules)
        sub_scores_list = []
        for idx, r in df_cat_cases.iterrows():
            sub_scores_list.append(r["sub_scores"])
            
        df_sub = pd.DataFrame(sub_scores_list)
        if not df_sub.empty:
            avg_sub = df_sub.mean()
            
            st.markdown("#### 🎯 Judge Sub-Scores Breakdown (Rules Average)")
            fig_sub = px.bar(
                x=avg_sub.index,
                y=avg_sub.values,
                labels={"x": "Scoring Dimension", "y": "Average Score (1-10)"},
                color=avg_sub.values,
                color_continuous_scale="Viridis",
                range_y=[0, 10]
            )
            fig_sub.update_layout(
                height=350,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_sub, use_container_width=True)
            
            # Show table of sub-scores
            st.table(pd.DataFrame({
                "Metric Score Rule": avg_sub.index,
                "Average Score": [round(val, 2) for val in avg_sub.values]
            }))
        else:
            st.info("No sub-score breakdown metrics available for this category.")

# Page 4: Test Case Investigator
elif page == "Test Case Investigator":
    st.markdown("## 🔍 Detailed Test Case Investigator")
    
    # Filters
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    with col_f1:
        f_cat = st.selectbox("Category Filter:", ["All", "Recommendation", "Personality", "Memory", "Retrieval", "Refusal"])
    with col_f2:
        f_status = st.selectbox("Status Filter:", ["All", "Passed", "Failed"])
    with col_f3:
        f_diff = st.selectbox("Difficulty Filter:", ["All", "easy", "medium", "hard", "expert"])
    with col_f4:
        f_search = st.text_input("Search Query Text:")
        
    # Apply filtering
    filtered_df = df_cases.copy()
    if f_cat != "All":
        filtered_df = filtered_df[filtered_df["category"] == f_cat.lower()]
    if f_status != "All":
        is_pass_val = True if f_status == "Passed" else False
        filtered_df = filtered_df[filtered_df["passed"] == is_pass_val]
    if f_diff != "All":
        filtered_df = filtered_df[filtered_df["difficulty"] == f_diff]
    if f_search:
        filtered_df = filtered_df[filtered_df["query"].str.contains(f_search, case=False)]
        
    st.markdown(f"Found **{len(filtered_df)}** matching test cases.")
    
    # Export options
    st.markdown("### 📥 Export Run Results")
    exp_col1, exp_col2 = st.columns(2)
    with exp_col1:
        # Download JSON
        json_str = df_cases.to_json(orient="records", indent=2)
        st.download_button(
            label="Download Full Run Results as JSON 📥",
            data=json_str,
            file_name=f"cinephile_eval_run_{selected_run['id']}_results.json",
            mime="application/json"
        )
    with exp_col2:
        # Download CSV
        # Flatten expected data into rows for CSV exports
        csv_df = df_cases.copy()
        csv_df["expected"] = csv_df["expected"].apply(lambda x: json.dumps(x))
        csv_df["sub_scores"] = csv_df["sub_scores"].apply(lambda x: json.dumps(x))
        csv_df["strengths"] = csv_df["strengths"].apply(lambda x: ", ".join(x))
        csv_df["weaknesses"] = csv_df["weaknesses"].apply(lambda x: ", ".join(x))
        csv_str = csv_df.to_csv(index=False)
        st.download_button(
            label="Download Full Run Results as CSV 📥",
            data=csv_str,
            file_name=f"cinephile_eval_run_{selected_run['id']}_results.csv",
            mime="text/csv"
        )
        
    st.markdown("---")
    
    # Case selector
    if filtered_df.empty:
        st.info("No test cases match the active filters.")
    else:
        # Create a selectbox list of cases
        case_options = {f"[{row['category'].upper()}] {row['case_id']} - Query: '{row['query'][:50]}...' ({'✅' if row['passed'] else '❌'})": row for idx, row in filtered_df.iterrows()}
        selected_case_label = st.selectbox("Select a test case to investigate details:", list(case_options.keys()))
        selected_case = case_options[selected_case_label]
        
        # Details layout
        det_col1, det_col2 = st.columns([1, 1])
        with det_col1:
            st.markdown("### 📝 Case Input & Context")
            st.markdown(f"**Test Case ID:** `{selected_case['case_id']}`")
            st.markdown(f"**Category:** `{selected_case['category'].upper()}`")
            st.markdown(f"**Difficulty:** `{selected_case['difficulty']}`")
            st.markdown(f"**Query:**\n> {selected_case['query']}")
            
            # Display expectations
            expected_obj = selected_case["expected"]
            if expected_obj:
                if "expected_behavior" in expected_obj:
                    st.markdown("**Expected Behaviors:**")
                    for eb in expected_obj["expected_behavior"]:
                        st.markdown(f"- {eb}")
                if "avoid" in expected_obj and expected_obj["avoid"]:
                    st.markdown("**Avoid Criteria:**")
                    for av in expected_obj["avoid"]:
                        st.markdown(f"- {av}")
                if "memory" in expected_obj and expected_obj["memory"]:
                    st.markdown("**Stored Profile Preference:**")
                    st.json(expected_obj["memory"])
                if "conversation_history" in expected_obj and expected_obj["conversation_history"]:
                    st.markdown("**Recent Conversation History:**")
                    st.json(expected_obj["conversation_history"])
            
        with det_col2:
            st.markdown("### 🏆 Output and Judge Decision")
            status_color = "green" if selected_case["passed"] else "red"
            status_label = "PASSED" if selected_case["passed"] else "FAILED"
            st.markdown(f"**Verdict:** <span style='color:{status_color}; font-weight:bold; font-size:1.2rem;'>{status_label}</span>", unsafe_allow_html=True)
            st.markdown(f"**Overall Score:** `{selected_case['score']:.1f}/10.0`")
            
            # Subscores
            if selected_case["sub_scores"]:
                st.markdown("**Scoring Details:**")
                sub_df = pd.DataFrame([selected_case["sub_scores"]])
                st.dataframe(sub_df)
                
            st.markdown(f"**Actual Response:**\n```\n{selected_case['actual_response']}\n```")
            st.markdown(f"**Judge Reasoning:**\n> {selected_case['reasoning']}")
            
            # Strengths/Weaknesses
            if selected_case["strengths"]:
                st.markdown("<span style='color:green; font-weight:bold;'>Strengths:</span>", unsafe_allow_html=True)
                for st_item in selected_case["strengths"]:
                    st.markdown(f"- {st_item}")
            if selected_case["weaknesses"]:
                st.markdown("<span style='color:red; font-weight:bold;'>Weaknesses:</span>", unsafe_allow_html=True)
                for wk_item in selected_case["weaknesses"]:
                    st.markdown(f"- {wk_item}")
