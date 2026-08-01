from __future__ import annotations

import os
import time
from datetime import time as clock_time

import numpy as np
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <span class="tag">IBM AutoAI + Python + Streamlit</span>
        <h1>Glucose Profile V2</h1>
        <p>
            Simulación académica para estimar la siguiente lectura de glucosa
            de un sistema CGM, aproximadamente 5 minutos hacia el futuro.
        </p>
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
            raise IBMScoringError("La conexión privada con IBM no está disponible.")
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
        return False, "La sesión alcanzó el límite de demostraciones."
    if now - last_request < 4:
        return False, "Espera unos segundos antes de generar otra predicción."

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
            name="Historial CGM",
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
                name="Predicción",
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
        title="Historial de 2 horas",
        xaxis_title="Tiempo relativo",
        yaxis_title="Glucosa estimada, mg/dL",
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

    with st.spinner("Consultando el modelo desplegado en IBM..."):
        predictions = client.score(
            fields=list(feature_row.keys()),
            values=[list(feature_row.values())],
        )
    return float(predictions[0])


def render_result(readings: list[float], prediction: float) -> None:
    current = float(readings[-1])
    change = prediction - current

    if change > 1:
        trend = "Ascendente"
    elif change < -1:
        trend = "Descendente"
    else:
        trend = "Estable"

    left, middle, right = st.columns(3)
    left.metric("Glucosa actual", f"{current:.1f} mg/dL")
    middle.metric("Predicción a 5 min", f"{prediction:.1f} mg/dL", f"{change:+.1f}")
    right.metric("Tendencia estimada", trend)

    st.plotly_chart(
        make_chart(readings, prediction),
        use_container_width=True,
        config={"displayModeBar": False},
    )


with st.sidebar:
    st.markdown("### Proyecto")
    st.write("Modelo V2 de regresión Ridge desarrollado en IBM AutoAI.")
    st.metric("R² interno", "0.998")
    st.metric("MAE externo", "0.675 mg/dL")
    st.metric("Dentro de 2.5 mg/dL", "99.39%")

    if backend_available():
        st.success("Modelo IBM conectado")
    else:
        st.warning("Modelo IBM no conectado en este entorno")

    st.markdown(
        """
        <p class="small-note">
            Las lecturas de esta interfaz son simuladas. No se solicitan datos
            personales ni información médica identificable.
        </p>
        """,
        unsafe_allow_html=True,
    )


tab_simulation, tab_manual, tab_project = st.tabs(
    ["Simulación rápida", "Secuencia personalizada", "Proyecto"]
)

with tab_simulation:
    st.subheader("Simula un comportamiento de glucosa")

    control_a, control_b, control_c, control_d = st.columns(4)

    with control_a:
        scenario = st.selectbox(
            "Escenario",
            ["Estable", "Ascenso gradual", "Descenso gradual", "Variación rápida"],
        )
    with control_b:
        current_glucose = st.slider(
            "Glucosa actual",
            min_value=60,
            max_value=240,
            value=105,
            step=1,
        )
    with control_c:
        intensity = st.slider(
            "Intensidad",
            min_value=0.5,
            max_value=1.5,
            value=1.0,
            step=0.1,
        )
    with control_d:
        selected_time = st.time_input(
            "Hora del escenario",
            value=clock_time(12, 0),
            step=300,
        )

    variability = st.slider(
        "Variabilidad de las lecturas",
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
    metric_a.metric("Lecturas simuladas", "24")
    metric_b.metric("Ventana histórica", "2 horas")
    metric_c.metric(
        "Cambio en la ventana",
        f"{readings[-1] - readings[0]:+.1f} mg/dL",
    )

    st.plotly_chart(
        make_chart(readings),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if st.button(
        "Generar predicción con IBM",
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
    st.subheader("Construye una secuencia de 24 lecturas")
    st.caption("Las filas avanzan desde 115 minutos antes hasta el momento actual.")

    default_readings = generate_sequence(
        current_glucose=105,
        scenario="Estable",
        intensity=1.0,
        variability=1.0,
        seed=7,
    )

    manual_table = pd.DataFrame(
        {
            "Minuto": list(range(-115, 5, 5)),
            "Glucosa": default_readings,
        }
    )

    edited_table = st.data_editor(
        manual_table,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Minuto"],
        column_config={
            "Minuto": st.column_config.NumberColumn(format="%d min"),
            "Glucosa": st.column_config.NumberColumn(
                min_value=40.0,
                max_value=400.0,
                step=1.0,
                format="%.1f mg/dL",
            ),
        },
        key="manual_sequence",
    )

    custom_time = st.time_input(
        "Hora de la lectura actual",
        value=clock_time(12, 0),
        step=300,
        key="manual_time",
    )

    manual_readings = edited_table["Glucosa"].astype(float).tolist()

    st.plotly_chart(
        make_chart(manual_readings),
        use_container_width=True,
        config={"displayModeBar": False},
    )

    if st.button(
        "Evaluar secuencia personalizada",
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
    st.subheader("Resumen del proyecto")

    summary_a, summary_b = st.columns([1.15, 0.85])

    with summary_a:
        st.markdown(
            """
            **Objetivo**

            Predecir la siguiente lectura de un sistema de monitoreo continuo
            de glucosa utilizando 24 lecturas históricas y variables temporales.

            **Evolución**

            - V1: Ridge Regression con ingeniería manual de características.
            - V2: entrenamiento y comparación de pipelines en IBM AutoAI.
            - Validación externa con participantes no utilizados en entrenamiento.
            - Despliegue batch y online en IBM Machine Learning.
            - Comprobación del endpoint mediante Python.

            **Tecnologías**

            Python, pandas, NumPy, scikit-learn, IBM AutoAI,
            IBM Machine Learning, REST API y Streamlit.
            """
        )

    with summary_b:
        st.markdown(
            """
            <div class="result-card">
                <strong>Resultados principales</strong><br><br>
                RMSE holdout: 0.785<br>
                RMSE cross-validation: 0.814<br>
                R²: 0.998<br>
                Validación externa dentro de 2.5 mg/dL: 99.39%<br>
                Respuesta del endpoint: HTTP 200
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.image(
        "assets/flowchart.png",
        caption="Flujo general de la Fase III",
        use_container_width=True,
    )


st.markdown(
    """
    <div class="footer-note">
        Proyecto académico y experimental. No es una herramienta médica,
        no diagnostica enfermedades y no debe utilizarse para calcular
        tratamientos o dosis de insulina.
    </div>
    """,
    unsafe_allow_html=True,
)
