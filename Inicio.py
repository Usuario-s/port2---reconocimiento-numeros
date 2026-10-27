
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import pandas as pd

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Detector de Dígitos MNIST",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# DISEÑO FUTURISTA
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Orbitron:wght@400;500;600;700;800&display=swap');

:root {
    --cyan: #00E5FF;
    --purple: #A855F7;
    --blue: #6478FF;
    --background: #080B14;
    --panel: #111827;
    --text: #F1F5F9;
    --muted: #A8B6CC;
}

.stApp {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(0,229,255,0.10), transparent 32%),
        radial-gradient(ellipse at 92% 12%, rgba(168,85,247,0.12), transparent 32%),
        #080B14;
    color: var(--text);
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

h1, h2, h3, h4 {
    color: #F8FAFC !important;
}

p, label, .stMarkdown {
    color: #CBD5E1;
}

[data-testid="stCaptionContainer"] {
    color: #94A3B8;
}

/* ENCABEZADO */

.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(125deg, #101C31, #17132D 75%);
    border: 1px solid rgba(0,229,255,0.36);
    border-radius: 24px;
    padding: 34px 38px;
    margin-bottom: 30px;
    box-shadow: 0 0 40px rgba(0,229,255,0.055);
}

.hero::after {
    content: "";
    position: absolute;
    right: -75px;
    top: -105px;
    width: 230px;
    height: 230px;
    border: 1px solid rgba(0,229,255,0.35);
    border-radius: 50%;
    box-shadow:
        0 0 0 25px rgba(0,229,255,0.025),
        0 0 0 50px rgba(168,85,247,0.025);
}

.hero-tag {
    color: #00E5FF;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 4px;
    margin-bottom: 16px;
}

.hero-title {
    color: #FFFFFF;
    font-family: 'Orbitron', sans-serif;
    font-size: clamp(25px, 3.8vw, 43px);
    font-weight: 800;
    line-height: 1.35;
    letter-spacing: 1px;
    margin-bottom: 14px;
}

.hero-description {
    color: #BAC9DE;
    font-size: 15px;
    line-height: 1.8;
    max-width: 760px;
}

.hero-line {
    height: 3px;
    width: 90px;
    margin-top: 22px;
    background: linear-gradient(90deg, #00E5FF, #A855F7);
    border-radius: 10px;
}

/* ETIQUETAS DE SECCIÓN */

.section-tag {
    color: #00E5FF;
    font-family: 'Orbitron', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    margin-bottom: 8px;
}

.section-description {
    color: #94A3B8;
    font-size: 14px;
    line-height: 1.7;
}

/* SIDEBAR */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1424, #080B14);
    border-right: 1px solid rgba(0,229,255,0.22);
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #00E5FF !important;
}

section[data-testid="stSidebar"] p {
    color: #B8C7DC !important;
    line-height: 1.8;
}

/* CONTENEDORES */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
}

/* LIENZO */

div[data-testid="stCustomComponentV1"] {
    border: 1px solid rgba(0,229,255,0.45);
    border-radius: 15px;
    overflow: hidden;
    box-shadow:
        0 0 22px rgba(0,229,255,0.08),
        0 0 2px rgba(0,229,255,0.2);
}

/* BOTONES */

.stButton > button {
    min-height: 46px;
    border-radius: 11px;
    border: 1px solid rgba(0,229,255,0.35);
    background: linear-gradient(105deg, #00D9F5, #7775FF);
    color: #080B14 !important;
    font-weight: 800;
    transition: all 0.2s ease;
    box-shadow: 0 4px 18px rgba(0,217,245,0.10);
}

.stButton > button p {
    color: #080B14 !important;
    font-weight: 800;
}

.stButton > button:hover {
    border-color: #00E5FF;
    transform: translateY(-2px);
    box-shadow: 0 7px 25px rgba(0,217,245,0.25);
}

/* MÉTRICAS */

div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #111C30, #13152A);
    border: 1px solid rgba(0,229,255,0.22);
    border-radius: 15px;
    padding: 18px;
}

div[data-testid="stMetricLabel"] {
    color: #A8B6CC !important;
}

div[data-testid="stMetricValue"] {
    color: #00E5FF !important;
    font-family: 'Orbitron', sans-serif;
    font-size: 28px;
}

/* GRÁFICOS */

div[data-testid="stVegaLiteChart"],
div[data-testid="stArrowVegaLiteChart"] {
    background: rgba(15,23,42,0.72);
    border: 1px solid rgba(168,85,247,0.20);
    border-radius: 15px;
    padding: 10px;
}

/* EXPANDERS */

div[data-testid="stExpander"] {
    background: rgba(15,23,42,0.68);
    border: 1px solid rgba(168,85,247,0.28);
    border-radius: 13px;
}

/* ALERTAS */

div[data-testid="stAlert"] {
    border-radius: 12px;
}

/* SEPARADORES */

hr {
    border-color: rgba(148,163,184,0.18) !important;
}

/* PIE DE PÁGINA */

.footer {
    text-align: center;
    color: #64748B;
    font-size: 11px;
    letter-spacing: 2px;
    padding: 22px 10px;
    margin-top: 35px;
    border-top: 1px solid rgba(148,163,184,0.15);
}

/* OCULTAR ELEMENTOS DE STREAMLIT */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# ENCABEZADO
# =========================================================

st.markdown("""
<div class="hero">
    <div class="hero-tag">NEURAL VISION // MNIST SYSTEM</div>
    <div class="hero-title">DETECTOR DE DÍGITOS</div>
    <div class="hero-description">
        Escribe un número del 0 al 9 y deja que una red neuronal
        convolucional analice tu trazo, identifique el dígito
        y calcule las probabilidades de clasificación.
    </div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# BARRA LATERAL
# =========================================================

with st.sidebar:
    st.markdown("## ✦ CENTRO DE CONTROL")
    st.markdown("---")

    st.markdown("### ESTADO DEL SISTEMA")

    st.markdown(
        '<span style="color:#00E5FF;">●</span> '
        '<span style="color:#CBD5E1;">TensorFlow conectado</span>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<span style="color:#A855F7;">●</span> '
        '<span style="color:#CBD5E1;">Clasificador MNIST</span>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### INFORMACIÓN TÉCNICA")
    st.write(f"**TensorFlow:** `{tf.__version__}`")
    st.write(f"**Keras:** `{keras.__version__}`")

    st.markdown("---")

    st.markdown("### INSTRUCCIONES")
    st.write("01. Dibuja un único dígito.")
    st.write("02. Pulsa el botón de predicción.")
    st.write("03. Consulta el resultado y las probabilidades.")

    st.markdown("---")

    st.caption("MNIST // CNN CLASSIFICATION ENGINE")


# =========================================================
# CARGAR MODELO
# =========================================================

@st.cache_resource
def load_mnist_model():
    """Carga el modelo MNIST desde archivo .keras o .h5."""

    model_files = [
        "mnist_model.keras",
        "mnist_model.h5"
    ]

    for model_path in model_files:
        if os.path.exists(model_path):
            try:
                model = keras.models.load_model(model_path)
                return model, model_path
            except Exception as e:
                st.error(
                    f"Error al cargar {model_path}: {str(e)}"
                )

    st.error("No se encontró ningún modelo compatible.")

    st.markdown("""
    **Para continuar:**

    1. Entrena el modelo en Google Colab.
    2. Descarga `mnist_model.keras`.
    3. Coloca el archivo en la raíz del proyecto.
    4. Reinicia la aplicación.
    """)

    st.stop()


model, model_path = load_mnist_model()


# =========================================================
# ESTADO DEL MODELO
# =========================================================

st.markdown(
    f"""
    <div style="
        background:rgba(0,229,255,0.055);
        border:1px solid rgba(0,229,255,0.23);
        border-radius:12px;
        padding:13px 18px;
        margin-bottom:24px;
    ">
        <span style="color:#00E5FF;font-weight:700;">
            ● MODELO OPERATIVO
        </span>
        <span style="color:#CBD5E1;">
            &nbsp; | &nbsp; Archivo:
        </span>
        <span style="color:#E9D5FF;">
            {model_path}
        </span>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# ÁREA DE DIBUJO
# =========================================================

st.markdown(
    '<div class="section-tag">01 // INPUT INTERFACE</div>',
    unsafe_allow_html=True
)

st.markdown("## Dibuja tu dígito")

st.markdown(
    '<div class="section-description">'
    'Utiliza el panel para dibujar un número del 0 al 9. '
    'Procura escribir un solo dígito, grande y centrado.'
    '</div>',
    unsafe_allow_html=True
)

st.write("")

col1, col2 = st.columns([1.35, 1], gap="large")

with col1:
    with st.container(border=True):
        st.markdown("### ✍️ Panel de escritura")

        canvas_result = st_canvas(
            fill_color="rgba(0, 229, 255, 0.0)",
            stroke_width=20,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=400,
            height=400,
            drawing_mode="freedraw",
            key="canvas",
        )

        st.caption(
            "ÁREA ACTIVA: 400 × 400 PX  |  TRAZO BLANCO SOBRE NEGRO"
        )

with col2:
    with st.container(border=True):
        st.markdown("### 🎮 Consola de control")

        st.markdown(
            '<div style="color:#94A3B8;font-size:13px;'
            'line-height:1.8;margin-bottom:18px;">'
            'Inicia el proceso de clasificación de la red neuronal '
            'o reinicia el panel para escribir otro dígito.'
            '</div>',
            unsafe_allow_html=True
        )

        predict_btn = st.button(
            "✦ PREDECIR DÍGITO",
            type="primary",
            use_container_width=True
        )

        clear_btn = st.button(
            "↻ LIMPIAR PANEL",
            use_container_width=True
        )

        if clear_btn:
            st.session_state.pop("mnist_result", None)
            st.rerun()

        st.markdown("---")

        st.markdown("#### PARÁMETROS DEL MODELO")

        st.markdown(
            '<div style="color:#CBD5E1;line-height:2;">'
            'Entrada: <span style="color:#00E5FF;">28 × 28</span><br>'
            'Canales: <span style="color:#00E5FF;">1 escala de grises</span><br>'
            'Clases: <span style="color:#00E5FF;">10 dígitos</span>'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# PREDICCIÓN
# =========================================================

if predict_btn:

    if canvas_result.image_data is not None:

        if np.max(canvas_result.image_data) == 0:

            st.warning("Dibuja un dígito antes de realizar la predicción.")

        else:

            with st.spinner("Analizando el trazo mediante la red neuronal..."):

                # Procesar imagen
                image = Image.fromarray(
                    canvas_result.image_data.astype("uint8")
                ).convert("L")

                image_resized = image.resize(
                    (28, 28),
                    Image.Resampling.LANCZOS
                )

                img_array = np.array(image_resized) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)

                # Predicción
                prediction = model.predict(
                    img_array,
                    verbose=0
                )

                digit = int(np.argmax(prediction[0]))
                confidence = float(np.max(prediction[0]) * 100)

                alternative = int(
                    np.argsort(prediction[0])[-2]
                )

                # Guardar resultados para conservarlos
                # al actualizar la interfaz
                st.session_state["mnist_result"] = {
                    "digit": digit,
                    "confidence": confidence,
                    "alternative": alternative,
                    "prediction": prediction[0].tolist(),
                    "original_image": canvas_result.image_data,
                    "processed_image": np.array(image_resized)
                }

    else:
        st.warning("Dibuja un dígito en el panel.")


# =========================================================
# VISUALIZACIÓN DE RESULTADOS
# =========================================================

if "mnist_result" in st.session_state:

    result = st.session_state["mnist_result"]

    digit = result["digit"]
    confidence = result["confidence"]
    alternative = result["alternative"]
    probabilities = np.array(result["prediction"])

    st.markdown("---")

    st.markdown(
        '<div class="section-tag">02 // NEURAL OUTPUT</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Resultado de clasificación")

    st.markdown(
        f"""
        <div style="
            background:linear-gradient(120deg,#111D32,#19142F);
            border:1px solid rgba(0,229,255,0.4);
            border-radius:20px;
            padding:28px;
            margin:16px 0 24px 0;
            box-shadow:0 0 30px rgba(0,229,255,0.06);
            text-align:center;
        ">
            <div style="
                font-family:'Orbitron',sans-serif;
                color:#00E5FF;
                font-size:12px;
                letter-spacing:4px;
                margin-bottom:10px;
            ">
                DIGIT IDENTIFIED
            </div>
            <div style="
                font-family:'Orbitron',sans-serif;
                font-size:86px;
                font-weight:800;
                line-height:1.3;
                color:#FFFFFF;
                text-shadow:0 0 25px rgba(0,229,255,0.3);
            ">
                {digit}
            </div>
            <div style="color:#A8B6CC;font-size:14px;">
                Clasificación principal de la red neuronal
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Métricas
    col_m1, col_m2, col_m3 = st.columns(3)

    with col_m1:
        st.metric(
            label="DÍGITO DETECTADO",
            value=str(digit)
        )

    with col_m2:
        st.metric(
            label="CONFIANZA DEL MODELO",
            value=f"{confidence:.1f}%"
        )

    with col_m3:
        st.metric(
            label="SEGUNDA OPCIÓN",
            value=str(alternative),
            delta=f"{probabilities[alternative] * 100:.1f}% prob."
        )

    st.markdown("---")

    # =====================================================
    # IMÁGENES PROCESADAS
    # =====================================================

    st.markdown(
        '<div class="section-tag">03 // IMAGE PROCESSING</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Análisis visual")

    col_img1, col_img2 = st.columns(2, gap="large")

    with col_img1:
        with st.container(border=True):
            st.markdown("### Imagen original")

            st.image(
                result["original_image"],
                width=260
            )

            st.caption(
                "Trazo original capturado desde el panel de escritura."
            )

    with col_img2:
        with st.container(border=True):
            st.markdown("### Entrada de la CNN")

            st.image(
                result["processed_image"],
                width=260
            )

            st.caption(
                "Imagen convertida a escala de grises y redimensionada "
                "a 28 × 28 píxeles."
            )

    st.markdown("---")

    # =====================================================
    # PROBABILIDADES
    # =====================================================

    st.markdown(
        '<div class="section-tag">04 // PROBABILITY ANALYSIS</div>',
        unsafe_allow_html=True
    )

    st.markdown("## Distribución de probabilidades")

    st.markdown(
        '<div class="section-description">'
        'Consulta la puntuación que el modelo asignó a cada dígito. '
        'Una puntuación mayor indica una mayor preferencia del modelo '
        'por esa clase, pero no garantiza que sea correcta.'
        '</div>',
        unsafe_allow_html=True
    )

    prob_df = pd.DataFrame({
        "Dígito": [str(i) for i in range(10)],
        "Probabilidad (%)": probabilities * 100
    })

    st.bar_chart(
        prob_df.set_index("Dígito"),
        y="Probabilidad (%)",
        height=350
    )

    # Detalles por clase
    with st.expander("⌕ EXPLORAR TODAS LAS PROBABILIDADES"):

        for i, prob in enumerate(probabilities):

            percentage = float(prob * 100)

            if i == digit:
                label = "★ PREDICCIÓN PRINCIPAL"
                color = "#00E5FF"
            elif i == alternative:
                label = "◇ SEGUNDA OPCIÓN"
                color = "#A855F7"
            else:
                label = "CLASE ALTERNATIVA"
                color = "#CBD5E1"

            st.markdown(
                f"""
                <div style="
                    display:flex;
                    align-items:center;
                    justify-content:space-between;
                    gap:12px;
                    padding:10px 14px;
                    margin-bottom:7px;
                    border:1px solid rgba(148,163,184,0.15);
                    border-radius:10px;
                    background:rgba(15,23,42,0.65);
                ">
                    <div>
                        <span style="
                            color:#FFFFFF;
                            font-weight:700;
                            font-size:16px;
                        ">Dígito {i}</span>
                        <span style="
                            color:{color};
                            font-size:10px;
                            margin-left:10px;
                        ">{label}</span>
                    </div>
                    <span style="
                        color:{color};
                        font-weight:700;
                        white-space:nowrap;
                    ">{percentage:.2f}%</span>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(
                min(max(percentage / 100, 0.0), 1.0)
            )


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.markdown("""
<div class="footer">
    NEURAL VISION SYSTEM // MNIST CLASSIFICATION ENGINE
</div>
""", unsafe_allow_html=True)


