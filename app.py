from __future__ import annotations

import os
import time
from datetime import time as clock_time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from glucose_features import build_feature_row
from ibm_client import IBMScoringClient, IBMScoringError
from simulations import generate_sequence


st.set_page_config(
    page_title="Glucose Profile V2",
    page_icon="🟣",
    layout="wide",
)

PURPLE_DARK = "#3E176F"
PURPLE = "#6941C6"
PURPLE_LIGHT = "#B69AF5"
LILAC = "#F2ECFF"
LILAC_SOFT = "#FBF9FF"
TEXT = "#241B35"
MUTED = "#6D6479"

PROJECT_PAGE_URL = "https://cindylmm95.github.io/glucose-regression-model/"
GITHUB_URL = "https://github.com/cindylmm95/glucose-regression-model"

st.markdown(
    f"""
    <style>
        .stApp {{
            background: linear-gradient(180deg, {LILAC_SOFT} 0%, #FFFFFF 44%);
            color: {TEXT};
        }}
        .block-container {{
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }}
        .hero {{
            padding: 2rem 2.2rem;
            border: 1px solid #E2D5FF;
            border-radius: 24px;
            background: linear-gradient(135deg, #FFFFFF 0%, {LILAC} 100%);
            margin-bottom: 1.4rem;
        }}
        .hero h1 {{
            color: {PURPLE_DARK};
            font-size: 2.65rem;
            margin: 0 0 0.45rem 0;
            letter-spacing: -0.04em;
        }}
        .hero p {{
            color: {MUTED};
            font-size: 1.05rem;
            margin: 0;
            max-width: 800px;
        }}
        .tag {{
            display: inline-block;
            background: #E9DFFF;
            color: {PURPLE_DARK};
            padding: 0.32rem 0.72rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.78rem;
            margin-bottom: 0.8rem;
        }}
        .hero-links {{
            display: flex;
            gap: 0.7rem;
            margin-top: 1rem;
            flex-wrap: wrap;
        }}
        .hero-links a {{
            text-decoration: none;
            color: {PURPLE_DARK};
            font-weight: 700;
            border: 1px solid #D7C4FF;
            background: #FFFFFF;
            padding: 0.55rem 0.85rem;
            border-radius: 10px;
        }}
        div[data-testid="stMetric"] {{
            border: 1px solid #E5DAFA;
            border-radius: 18px;
            padding: 1rem;
            background: #FFFFFF;
        }}
        div[data-testid="stMetricValue"] {{
            color: {PURPLE_DARK};
        }}
        .result-card {{
            border: 1px solid #DCCBFF;
            border-radius: 20px;
            padding: 1.2rem 1.3rem;
            background: #FFFFFF;
        }}
        .workflow {{
            display: grid;
            grid-template-columns: repeat(5, 1fr);
            gap: 0.65rem;
            margin: 1rem 0;
        }}
        .workflow-step {{
            border: 1px solid #DCCBFF;
            background: #FFFFFF;
            border-radius: 16px;
            padding: 0.85rem;
            min-height: 112px;
        }}
        .workflow-number {{
            color: {PURPLE};
            font-weight: 800;
            font-size: 0.78rem;
        }}
        .workflow-step strong {{
            display: block;
            color: {PURPLE_DARK};
            margin: 0.3rem 0;
        }}
        .workflow-step span {{
            color: {MUTED};
            font-size: 0.82rem;
        }}
        .small-note {{
            color: {MUTED};
            font-size: 0.87rem;
        }}
        .footer-note {{
            border-top: 1px solid #E5DAFA;
            margin-top: 2rem;
            padding-top: 1rem;
            color: {MUTED};
            font-size: 0.85rem;
        }}
        .stButton > button {{
            border-radius: 12px;
            border: 0;
            background: {PURPLE};
            color: white;
            font-weight: 700;
            min-height: 3rem;
        }}
        .stButton > button:hover {{
            background: {PURPLE_DARK};
            color: white;
        }}
        @media (max-width: 900px) {{
            .workflow {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <section class="hero">
        <span class="tag">IBM AutoAI + Python + Streamlit</span>
        <h1>Glucose Profile V2</h1>
        <p>
            An academic simulation that estimates the next CGM glucose reading,
            approximately 5 minutes into the future.
        </p>
        <div class="hero-links">
            <a href="{PROJECT_PAGE_URL}" target="_blank">Project page</a>
            <a href="{GITHUB_URL}" target="_blank">GitHub repository</a>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)


def read_secret(name: str) -> str | None:
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or os.getenv(name)


def backend_available() -> bool:
    return bool(read_secret("IBM_API_KEY") and read_secret("IBM_SCORING_URL"))


def get_client() -> IBMScoringClient:
    if "_ibm_client" not in st.session_state:
        api_key = read_secret("IBM_API_KEY")
        scoring_url = read_secret("IBM_SCORING_URL")
        if not api_key or not scoring_url:
            raise IBMScoringError("The private IBM connection is not available.")
        st.session_state["_ibm_client"] = IBMScoringClient(
            api_key=api_key,
            scoring_url=scoring_url,
        )
    return st.session_state["_ibm_client"]


def request_allowed() -> tuple[bool, str]:
    now = time.time()
    last_request = st.session_state.get("_last_request_at", 0.0)
    request_count = st.session_state.get("_request_count", 0)

    if request_count >= 20:
        return False, "This session reached the demonstration request limit."
    if now - last_request < 4:
        return False, "Wait a few seconds before requesting another prediction."

    st.session_state["_last_request_at"] = now
    st.session_state["_request_count"] = request_count + 1
    return True, ""


def make_chart(
    readings: list[float],
    prediction: float | None = None,
) -> go.Figure:
    history_minutes = list(range(-115, 5, 5))
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=history_minutes,
            y=readings,
            mode="lines+markers",
            name="CGM history",
            line={"color": PURPLE, "width": 3},
            marker={"size": 7, "color": PURPLE_LIGHT},
            hovertemplate="%{x} min<br>%{y:.1f} mg/dL<extra></extra>",
        )
    )

    if prediction is not None:
        figure.add_trace(
            go.Scatter(
                x=[0, 5],
                y=[readings[-1], prediction],
                mode="lines+markers",
                name="Prediction",
                line={"color": PURPLE_DARK, "width": 3, "dash": "dot"},
                marker={"size": [8, 13], "color": [PURPLE, PURPLE_DARK]},
                hovertemplate="%{x} min<br>%{y:.1f} mg/dL<extra></extra>",
            )
        )

    figure.update_layout(
        height=410,
        margin={"l": 20, "r": 20, "t": 45, "b": 25},
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        title="2-hour glucose history",
        xaxis_title="Relative time",
        yaxis_title="Estimated glucose, mg/dL",
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    figure.update_xaxes(gridcolor="#EEE8F8", zerolinecolor="#C8B7EC")
    figure.update_yaxes(gridcolor="#EEE8F8")
    return figure


def predict(readings: list[float], selected_time: clock_time) -> float | None:
    allowed, reason = request_allowed()
    if not allowed:
        st.info(reason)
        return None

    feature_row = build_feature_row(readings, selected_time)
    client = get_client()

    with st.spinner("Requesting a prediction from the IBM deployment..."):
        predictions = client.score(
            fields=list(feature_row.keys()),
            values=[list(feature_row.values())],
        )
    return float(predictions[0])


def render_result(readings: list[float], prediction: float) -> None:
    current = float(readings[-1])
    change = prediction - current

    if change > 1:
        trend = "Rising"
    elif change < -1:
        trend = "Falling"
    else:
        trend = "Stable"

    left, middle, right = st.columns(3)
    left.metric("Current glucose", f"{current:.1f} mg/dL")
    middle.metric("5-minute prediction", f"{prediction:.1f} mg/dL", f"{change:+.1f}")
    right.metric("Estimated trend", trend)

    st.plotly_chart(
        make_chart(readings, prediction),
        use_container_width=True,
        config={"displayModeBar": False},
    )


with st.sidebar:
    st.markdown("### Project")
    st.write("V2 Ridge regression model developed with IBM AutoAI.")
    st.metric("Internal R²", "0.998")
    st.metric("External MAE", "0.675 mg/dL")
    st.metric("Within 2.5 mg/dL", "99.39%")

    if backend_available():
        st.success("IBM model connected")
    else:
        st.warning("IBM model is not connected in this environment")

    st.markdown(
        """
        <p class="small-note">
            All readings in this interface are simulated. The application does
            not request personal data or identifiable medical information.
        </p>
        """,
        unsafe_allow_html=True,
    )


tab_simulation, tab_manual, tab_project = st.tabs(
    ["Quick simulation", "Custom sequence", "Project"]
)

with tab_simulation:
    st.subheader("Simulate a glucose pattern")

    control_a, control_b, control_c, control_d = st.columns(4)

    with control_a:
        scenario = st.selectbox(
            "Scenario",
            ["Stable", "Gradual rise", "Gradual fall", "Rapid variation"],
        )
    with control_b:
        current_glucose = st.slider(
            "Current glucose",
            min_value=60,
            max_value=240,
            value=105,
            step=1,
        )
    with control_c:
        intensity = st.slider(
            "Intensity",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
        )
    with control_d:
        selected_time = st.time_input(
            "Scenario time",
            value=clock_time(12, 0),
            step=300,
        )

    variability = st.slider(
        "Reading variability",
        min_value=0.0,
        max_value=6.0,
        value=1.5,
        step=0.5,
    )

    readings = generate_sequence(
        current_glucose=float(current_glucose),
        scenario=scenario,
        intensity=float(intensity),
        variability=float(variability),
        seed=42,
    )

    metric_a, metric_b, metric_c = st.columns(3)
    metric_a.metric("Simulated readings", "24")
    metric_b.metric("Historical window", "2 hours")
    metric_c.metric(
        "Window change",
        f"{readings[-1] - readings[0]:+.1f} mg/dL",
    )

    st.plotly_chart(
        make_chart(readings),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if st.button(
        "Generate prediction with IBM",
        type="primary",
        use_container_width=True,
        disabled=not backend_available(),
        key="predict_simulation",
    ):
        try:
            prediction = predict(readings, selected_time)
            if prediction is not None:
                st.session_state["simulation_result"] = prediction
        except IBMScoringError as error:
            st.error(str(error))

    if "simulation_result" in st.session_state:
        render_result(readings, st.session_state["simulation_result"])


with tab_manual:
    st.subheader("Build a 24-reading sequence")
    st.caption("Rows progress from 115 minutes before the current reading to now.")

    default_readings = generate_sequence(
        current_glucose=105,
        scenario="Stable",
        intensity=1.0,
        variability=1.0,
        seed=7,
    )

    manual_table = pd.DataFrame(
        {
            "Minute": list(range(-115, 5, 5)),
            "Glucose": default_readings,
        }
    )

    edited_table = st.data_editor(
        manual_table,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Minute"],
        column_config={
            "Minute": st.column_config.NumberColumn(format="%d min"),
            "Glucose": st.column_config.NumberColumn(
                min_value=40.0,
                max_value=400.0,
                step=1.0,
                format="%.1f mg/dL",
            ),
        },
        key="manual_sequence",
    )

    custom_time = st.time_input(
        "Current reading time",
        value=clock_time(12, 0),
        step=300,
        key="manual_time",
    )

    manual_readings = edited_table["Glucose"].astype(float).tolist()

    st.plotly_chart(
        make_chart(manual_readings),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if st.button(
        "Evaluate custom sequence",
        type="primary",
        use_container_width=True,
        disabled=not backend_available(),
        key="predict_manual",
    ):
        try:
            prediction = predict(manual_readings, custom_time)
            if prediction is not None:
                st.session_state["manual_result"] = prediction
        except (IBMScoringError, ValueError) as error:
            st.error(str(error))

    if "manual_result" in st.session_state:
        render_result(manual_readings, st.session_state["manual_result"])


with tab_project:
    st.subheader("Project overview")

    st.markdown(
        """
        <div class="workflow">
            <div class="workflow-step">
                <div class="workflow-number">01</div>
                <strong>CGM data</strong>
                <span>64 participants and 53,760 model-ready observations.</span>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">02</div>
                <strong>Feature engineering</strong>
                <span>24 historical readings and 51 time-series features.</span>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">03</div>
                <strong>IBM AutoAI</strong>
                <span>Eight regression pipelines compared using RMSE.</span>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">04</div>
                <strong>Deployment</strong>
                <span>Batch and online deployments in IBM Machine Learning.</span>
            </div>
            <div class="workflow-step">
                <div class="workflow-number">05</div>
                <strong>Validation</strong>
                <span>External participants, batch scoring and Python API testing.</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_a, summary_b = st.columns([1.15, 0.85])

    with summary_a:
        st.markdown(
            """
            **Objective**

            Predict the next CGM reading using 24 historical readings and
            time-based features.

            **Model evolution**

            - V1: Ridge Regression with manual feature engineering.
            - V2: automated pipeline comparison in IBM AutoAI.
            - External validation with participants excluded from training.
            - Batch and online deployments in IBM Machine Learning.
            - Endpoint verification with Python.

            **Technologies**

            Python, pandas, NumPy, scikit-learn, IBM AutoAI,
            IBM Machine Learning, REST API and Streamlit.
            """
        )

    with summary_b:
        st.markdown(
            """
            <div class="result-card">
                <strong>Key results</strong><br><br>
                Holdout RMSE: 0.785<br>
                Cross-validation RMSE: 0.814<br>
                R²: 0.998<br>
                External validation within 2.5 mg/dL: 99.39%<br>
                Endpoint response: HTTP 200
            </div>
            """,
            unsafe_allow_html=True,
        )


st.markdown(
    """
    <div class="footer-note">
        Academic and experimental project. This application is not a medical
        device, does not diagnose disease and must not be used to determine
        treatment or insulin dosing.
    </div>
    """,
    unsafe_allow_html=True,
)
