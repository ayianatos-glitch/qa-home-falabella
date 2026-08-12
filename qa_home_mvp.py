
import io
import re
import zipfile
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image

st.set_page_config(page_title="QA Home", layout="wide")
st.title("QA Home — MVP v0.4")
st.caption("Excel → Desktop Renderer → Video App → diferencias para revisión")

# ----------------------------
# Helpers
# ----------------------------
def norm(x):
    if pd.isna(x):
        return ""
    return re.sub(r"\s+", " ", str(x).lower().strip())

def compare_text(expected, observed):
    if not norm(observed):
        return "👁️ NO VISIBLE", "Sin evidencia"

    e, o = norm(expected), norm(observed)

    ep = re.findall(r"\b(\d{1,3})\s*%", e)
    op = re.findall(r"\b(\d{1,3})\s*%", o)
    if ep and op and set(ep) != set(op):
        return "🔴 ERROR", f"Descuento esperado {', '.join(ep)}% vs observado {', '.join(op)}%"

    prices_e = re.findall(r"\$?\s*(\d{1,3}(?:[\.]\d{3})+)", e)
    prices_o = re.findall(r"\$?\s*(\d{1,3}(?:[\.]\d{3})+)", o)
    if prices_e and prices_o and not set(prices_e).intersection(prices_o):
        return "🔴 ERROR", "Precio distinto"

    toks = [t for t in re.findall(r"[a-záéíóúñ0-9]+", e) if len(t) > 3]
    overlap = sum(t in o for t in toks) / max(1, len(toks))

    if overlap >= 0.55:
        return "✅ OK", "Coincidencia suficiente"
    if overlap >= 0.25:
        return "⚠️ REVISAR", "Coincidencia parcial"
    return "⚠️ REVISAR", "Contenido distinto o insuficiente"

@st.cache_data(ttl=300, show_spinner=False)
def fetch_renderer(url):
    r = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "Mozilla/5.0"}
    )
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    parts = list(soup.stripped_strings)

    for tag in soup.find_all(True):
        for attr in ("alt", "title", "aria-label"):
            if tag.get(attr):
                parts.append(str(tag.get(attr)))

    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    return r.status_code, text

def video_metadata(video_bytes, suffix):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    Path(tmp_path).unlink(missing_ok=True)

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration": duration,
        "width": width,
        "height": height,
    }

@st.cache_data(show_spinner=False)
def extract_frames(video_bytes, suffix, interval_seconds=2.0, max_frames=80):
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps else 0

    frames = []
    t = 0.0

    while t <= duration and len(frames) < max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(frame)

        # Reduce huge phone captures to keep Streamlit responsive.
        max_width = 900
        if pil.width > max_width:
            new_h = int(pil.height * max_width / pil.width)
            pil = pil.resize((max_width, new_h))

        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=82)
        frames.append({
            "time": round(t, 1),
            "jpeg": buffer.getvalue()
        })
        t += interval_seconds

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return frames

def frames_zip(frames):
    zbuf = io.BytesIO()
    with zipfile.ZipFile(zbuf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, fr in enumerate(frames, start=1):
            name = f"app_frame_{i:03d}_{fr['time']:.1f}s.jpg"
            z.writestr(name, fr["jpeg"])
    return zbuf.getvalue()

# ----------------------------
# 1. Excel
# ----------------------------
st.subheader("1. Programación Excel")
excel = st.file_uploader("Sube la programación Excel", type=["xlsx"])

if not excel:
    st.stop()

book = pd.ExcelFile(excel)
default_idx = book.sheet_names.index("Brief 12.08") if "Brief 12.08" in book.sheet_names else 0
sheet = st.selectbox("Hoja a validar", book.sheet_names, index=default_idx)

df = pd.read_excel(excel, sheet_name=sheet)

comp = next((c for c in df.columns if str(c).strip().lower() in ["componente", "posición", "posicion"]), None)
call = next((c for c in df.columns if str(c).strip().lower().startswith("llamado")), None)
ger = next((c for c in df.columns if str(c).strip().lower() == "gerencia"), None)

if not comp or not call:
    st.error("No encuentro las columnas Componente y Llamado.")
    st.stop()

base = df[df[comp].notna()].copy()
st.success(f"{len(base)} filas con componente detectadas")

# ----------------------------
# 2. Desktop
# ----------------------------
st.subheader("2. Desktop — Renderer")
renderer_url = st.text_input(
    "Pega aquí la URL del Renderer Desktop",
    placeholder="https://renderer.falabella.com/preview/..."
)

if st.button("Leer Renderer Desktop"):
    if not renderer_url:
        st.warning("Pega primero la URL del renderer.")
    else:
        try:
            with st.spinner("Leyendo renderer..."):
                status, text = fetch_renderer(renderer_url)
            st.session_state["renderer_text"] = text
            st.success(f"Renderer leído correctamente (HTTP {status}).")
            st.metric("Texto detectado", f"{len(text):,} caracteres")
        except Exception as e:
            st.error(f"No pude leer el renderer: {e}")

renderer_text = st.session_state.get("renderer_text", "")

if renderer_text:
    with st.expander("Ver muestra de lo detectado en Desktop"):
        st.text(renderer_text[:4000])

# ----------------------------
# 3. App video
# ----------------------------
st.subheader("3. App — Video")
st.write(
    "Sube una grabación de pantalla recorriendo la Home de la App. "
    "Idealmente haz un scroll continuo y relativamente lento."
)

video = st.file_uploader(
    "Subir video de App",
    type=["mp4", "mov", "m4v"],
    key="app_video"
)

frames = []

if video:
    video_bytes = video.getvalue()
    suffix = Path(video.name).suffix.lower() or ".mp4"

    try:
        meta = video_metadata(video_bytes, suffix)

        c1, c2, c3 = st.columns(3)
        c1.metric("Duración", f"{meta['duration']:.1f} s")
        c2.metric("Resolución", f"{meta['width']}×{meta['height']}")
        c3.metric("FPS", f"{meta['fps']:.1f}")

        interval = st.slider(
            "Tomar un frame cada cuántos segundos",
            min_value=0.5,
            max_value=5.0,
            value=1.5,
            step=0.5
        )

        if st.button("Procesar video App", type="primary"):
            with st.spinner("Extrayendo evidencia del video..."):
                frames = extract_frames(
                    video_bytes,
                    suffix,
                    interval_seconds=interval,
                    max_frames=100
                )
                st.session_state["app_frames"] = frames

        frames = st.session_state.get("app_frames", [])

        if frames:
            st.success(f"{len(frames)} frames extraídos del video.")
            st.download_button(
                "Descargar frames del video (.zip)",
                data=frames_zip(frames),
                file_name="frames_app.zip",
                mime="application/zip"
            )

            with st.expander("Ver frames detectados", expanded=True):
                cols = st.columns(4)
                for i, fr in enumerate(frames):
                    with cols[i % 4]:
                        st.image(fr["jpeg"], caption=f"{fr['time']} s", use_container_width=True)

    except Exception as e:
        st.error(f"No pude procesar el video: {e}")

# ----------------------------
# 4. QA
# ----------------------------
st.subheader("4. Resultado QA")
st.info(
    "En esta v0.4, Desktop se compara automáticamente por texto del Renderer. "
    "El video App ya se procesa y deja evidencia frame por frame. "
    "La lectura visual automática del contenido de esos frames se conectará en el siguiente paso."
)

qa = pd.DataFrame({
    "Componente": base[comp].astype(str),
    "Gerencia": base[ger].fillna("").astype(str) if ger else "",
    "Excel esperado": base[call].fillna("").astype(str),
})

if st.button("Analizar QA"):
    has_video = bool(st.session_state.get("app_frames", []))
    rows = []

    for _, r in qa.iterrows():
        if renderer_text:
            ds, do = compare_text(r["Excel esperado"], renderer_text)
        else:
            ds, do = "👁️ NO VISIBLE", "Renderer no cargado"

        if has_video:
            aps = "📹 EVIDENCIA"
            apo = "Video procesado; pendiente lectura visual automática"
        else:
            aps = "👁️ NO VISIBLE"
            apo = "Video App no cargado"

        if ds == "🔴 ERROR":
            overall = "🔴 ERROR"
        elif ds == "⚠️ REVISAR":
            overall = "⚠️ REVISAR"
        elif has_video:
            overall = "📹 REVISAR APP"
        else:
            overall = "👁️ NO VISIBLE"

        rows.append({
            **r.to_dict(),
            "QA Desktop": ds,
            "QA App": aps,
            "Estado": overall,
            "Observación Desktop": do,
            "Observación App": apo,
        })

    st.session_state["out_v04"] = pd.DataFrame(rows)

if "out_v04" in st.session_state:
    out = st.session_state["out_v04"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Desktop OK", (out["QA Desktop"] == "✅ OK").sum())
    c2.metric("Desktop Error", (out["QA Desktop"] == "🔴 ERROR").sum())
    c3.metric("Desktop Revisar", (out["QA Desktop"] == "⚠️ REVISAR").sum())
    c4.metric("Evidencia App", len(st.session_state.get("app_frames", [])))

    st.dataframe(out, use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="QA")

    st.download_button(
        "Descargar reporte QA Excel",
        data=buf.getvalue(),
        file_name="reporte_qa.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.divider()
st.caption(
    "Consejo para el video: parte desde arriba de la Home y baja de forma continua, "
    "dejando cada bloque visible al menos 1–2 segundos."
)
