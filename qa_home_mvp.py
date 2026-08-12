
import io
import os
import re
import json
import base64
import tempfile
from pathlib import Path
from urllib.parse import urljoin

import cv2
import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
from rapidfuzz import fuzz
from openai import OpenAI

st.set_page_config(page_title="QA Home", layout="wide")
st.title("QA Home — MVP v0.5")
st.caption("Cruce inteligente: Brief ↔ Desktop ↔ App")

# =========================================================
# Helpers
# =========================================================

def clean(v):
    if pd.isna(v):
        return ""
    return re.sub(r"\s+", " ", str(v)).strip()

def norm(v):
    return clean(v).lower()

def extract_numbers(text):
    s = norm(text)
    discounts = re.findall(r"\b(\d{1,3})\s*%", s)
    prices = re.findall(r"\$?\s*(\d{1,3}(?:[.\s]\d{3})+)", s)
    prices = [re.sub(r"\D", "", p) for p in prices]
    return discounts, prices

def data_url(image_bytes, mime="image/jpeg"):
    return f"data:{mime};base64," + base64.b64encode(image_bytes).decode("utf-8")

def safe_json(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)

def get_api_key():
    # 1) Streamlit secrets
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    # 2) Environment
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    # 3) Session input
    return st.session_state.get("api_key_input", "")

VISION_PROMPT = """
Analiza esta evidencia visual de una home de ecommerce.
Extrae TODAS las piezas/promociones legibles o claramente identificables.

Devuelve SOLO JSON válido con esta estructura:
{
  "items": [
    {
      "title": "llamado principal o campaña",
      "brand": "marca si existe",
      "category": "categoría si es inferible",
      "price": "precio exacto visible, si existe",
      "discount": "descuento exacto visible, si existe",
      "installments": "cuotas/CSI si existe",
      "keywords": ["palabras", "clave"],
      "visible_text": "transcripción breve de los textos comerciales relevantes"
    }
  ]
}

Reglas:
- No inventes datos que no se vean.
- Si hay varias piezas visibles, devuelve varios items.
- Conserva precios, porcentajes, marcas y nombres tal como se ven.
- Si una pieza está parcialmente cortada, extrae lo que sea seguro y no completes lo faltante.
"""

def analyze_image_with_ai(image_bytes, api_key, model="gpt-5.6"):
    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": VISION_PROMPT},
                    {
                        "type": "input_image",
                        "image_url": data_url(image_bytes),
                        "detail": "high",
                    },
                ],
            }
        ],
    )
    parsed = safe_json(response.output_text)
    return parsed.get("items", [])

def obs_text(obs):
    parts = [
        obs.get("title", ""),
        obs.get("brand", ""),
        obs.get("category", ""),
        obs.get("price", ""),
        obs.get("discount", ""),
        obs.get("installments", ""),
        " ".join(obs.get("keywords", []) or []),
        obs.get("visible_text", ""),
    ]
    return clean(" ".join([p for p in parts if p]))

def semantic_score(expected, observed):
    e = norm(expected)
    o = norm(observed)
    if not e or not o:
        return 0
    return max(
        fuzz.token_set_ratio(e, o),
        fuzz.token_sort_ratio(e, o),
        fuzz.partial_ratio(e, o),
    )

def compare_expected_to_obs(expected, obs):
    observed = obs_text(obs)
    score = semantic_score(expected, observed)

    e_discounts, e_prices = extract_numbers(expected)
    o_discounts, o_prices = extract_numbers(observed)

    hard_errors = []
    if e_discounts and o_discounts and set(e_discounts).isdisjoint(o_discounts):
        hard_errors.append(f"% esperado {', '.join(e_discounts)} vs observado {', '.join(o_discounts)}")

    if e_prices and o_prices and set(e_prices).isdisjoint(o_prices):
        hard_errors.append("precio distinto")

    if hard_errors:
        return {
            "status": "🔴 ERROR",
            "score": score,
            "reason": " · ".join(hard_errors),
            "observed": observed,
        }

    if score >= 78:
        return {"status": "✅ OK", "score": score, "reason": "Coincidencia alta", "observed": observed}
    if score >= 55:
        return {"status": "⚠️ REVISAR", "score": score, "reason": "Coincidencia parcial", "observed": observed}
    return {"status": "👁️ NO DETECTADO", "score": score, "reason": "Sin coincidencia suficiente", "observed": observed}

def best_match(expected, observations):
    if not observations:
        return {
            "status": "👁️ NO DETECTADO",
            "score": 0,
            "reason": "Sin evidencia analizada",
            "observed": "",
            "source": "",
        }
    results = []
    for obs in observations:
        r = compare_expected_to_obs(expected, obs)
        r["source"] = obs.get("_source", "")
        results.append(r)
    # Prioritize hard errors if they are also semantically related.
    related_errors = [r for r in results if r["status"] == "🔴 ERROR" and r["score"] >= 55]
    if related_errors:
        return max(related_errors, key=lambda x: x["score"])
    return max(results, key=lambda x: x["score"])

def cross_desk_app(desk_match, app_match):
    if not desk_match["observed"] or not app_match["observed"]:
        return "👁️ SIN CRUCE", 0, "No hay evidencia suficiente en ambos canales"

    score = semantic_score(desk_match["observed"], app_match["observed"])

    d_disc, d_prices = extract_numbers(desk_match["observed"])
    a_disc, a_prices = extract_numbers(app_match["observed"])

    if d_disc and a_disc and set(d_disc).isdisjoint(a_disc):
        return "🔴 DIFERENCIA", score, f"Desktop/App tienen % distintos ({','.join(d_disc)} vs {','.join(a_disc)})"
    if d_prices and a_prices and set(d_prices).isdisjoint(a_prices):
        return "🔴 DIFERENCIA", score, "Desktop/App tienen precios distintos"

    if score >= 78:
        return "✅ MISMA INFO", score, "Desktop y App parecen contener la misma pieza/información"
    if score >= 55:
        return "⚠️ REVISAR", score, "Desktop y App se parecen, pero no lo suficiente"
    return "❌ NO COINCIDE", score, "Desktop y App parecen piezas distintas"

# =========================================================
# Renderer
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_renderer(url):
    r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    image_urls = []
    for img in soup.find_all("img"):
        candidates = []
        if img.get("src"):
            candidates.append(img.get("src"))
        if img.get("srcset"):
            for chunk in img.get("srcset").split(","):
                candidates.append(chunk.strip().split(" ")[0])
        for c in candidates:
            if not c:
                continue
            u = urljoin(url, c)
            if u not in image_urls:
                image_urls.append(u)

    return {
        "status": r.status_code,
        "html": r.text,
        "image_urls": image_urls,
    }

@st.cache_data(ttl=300, show_spinner=False)
def download_image(url):
    r = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    ctype = r.headers.get("content-type", "image/jpeg")
    return r.content, ctype

# =========================================================
# Video
# =========================================================

@st.cache_data(show_spinner=False)
def extract_video_frames(video_bytes, suffix, every_seconds=1.5, max_frames=30):
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

        # keep enough resolution for text
        max_w = 1200
        if pil.width > max_w:
            new_h = int(pil.height * max_w / pil.width)
            pil = pil.resize((max_w, new_h))

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=88)
        frames.append({"time": round(t, 1), "bytes": buf.getvalue()})
        t += every_seconds

    cap.release()
    Path(tmp_path).unlink(missing_ok=True)
    return frames

# =========================================================
# UI
# =========================================================

with st.sidebar:
    st.header("IA visual")
    st.caption("Para leer textos, precios y descuentos dentro de las gráficas.")
    current_key = get_api_key()
    if current_key:
        st.success("API key configurada")
    else:
        st.text_input(
            "OpenAI API key",
            type="password",
            key="api_key_input",
            help="También puedes configurarla en Streamlit Secrets como OPENAI_API_KEY."
        )
    st.caption("El análisis con IA consume API según la cantidad de imágenes procesadas.")

st.subheader("1. Brief / Programación")
excel = st.file_uploader("Sube el Excel", type=["xlsx"])

if not excel:
    st.stop()

book = pd.ExcelFile(excel)
default_idx = book.sheet_names.index("Brief 12.08") if "Brief 12.08" in book.sheet_names else 0
sheet = st.selectbox("Hoja a validar", book.sheet_names, index=default_idx)
df = pd.read_excel(excel, sheet_name=sheet)

comp = next((c for c in df.columns if str(c).strip().lower() in ["componente","posición","posicion"]), None)
call = next((c for c in df.columns if str(c).strip().lower().startswith("llamado")), None)
ger = next((c for c in df.columns if str(c).strip().lower() == "gerencia"), None)
state_col = next((c for c in df.columns if str(c).strip().lower() == "estado"), None)

if not comp or not call:
    st.error("No encuentro las columnas Componente y Llamado.")
    st.stop()

base = df[df[comp].notna()].copy()
if state_col:
    base = base[~base[state_col].astype(str).str.contains("PENDIENTE BRIEF", case=False, na=False)]

st.success(f"{len(base)} piezas del brief detectadas")

# ---------------- Desktop
st.subheader("2. Desktop — Renderer")
renderer_url = st.text_input(
    "URL Renderer Desktop",
    placeholder="https://renderer.falabella.com/preview/..."
)

if st.button("Cargar Desktop"):
    if not renderer_url:
        st.warning("Pega primero la URL.")
    else:
        try:
            with st.spinner("Leyendo imágenes del renderer..."):
                renderer = fetch_renderer(renderer_url)
            st.session_state["renderer"] = renderer
            st.success(f"Renderer cargado: {len(renderer['image_urls'])} imágenes encontradas")
        except Exception as e:
            st.error(f"No pude cargar Desktop: {e}")

renderer = st.session_state.get("renderer")

if renderer:
    max_desk = st.slider(
        "Máximo de imágenes Desktop a analizar con IA",
        5, min(50, max(5, len(renderer["image_urls"]))), min(20, max(5, len(renderer["image_urls"]))), 1
    )
    with st.expander("Ver imágenes Desktop detectadas"):
        urls = renderer["image_urls"][:max_desk]
        cols = st.columns(4)
        for i, u in enumerate(urls):
            try:
                b, _ = download_image(u)
                cols[i % 4].image(b, caption=f"Desk {i+1}", use_container_width=True)
            except Exception:
                pass

# ---------------- App video
st.subheader("3. App — Video")
video = st.file_uploader("Sube la grabación de pantalla App", type=["mp4","mov","m4v"])

if video:
    every = st.slider("Extraer frame cada", 0.5, 4.0, 1.5, 0.5)
    max_frames = st.slider("Máximo de frames App", 5, 40, 20, 1)

    if st.button("Procesar video"):
        with st.spinner("Extrayendo frames..."):
            frames = extract_video_frames(
                video.getvalue(),
                Path(video.name).suffix or ".mp4",
                every_seconds=every,
                max_frames=max_frames,
            )
        st.session_state["app_frames"] = frames
        st.success(f"{len(frames)} frames extraídos")

frames = st.session_state.get("app_frames", [])
if frames:
    with st.expander("Ver frames App"):
        cols = st.columns(4)
        for i, fr in enumerate(frames):
            cols[i % 4].image(fr["bytes"], caption=f"{fr['time']}s", use_container_width=True)

# ---------------- AI Analysis
st.subheader("4. Analizar contenido visual")

api_key = get_api_key()

if st.button("Analizar Desktop + App con IA", type="primary"):
    if not api_key:
        st.error("Falta configurar la OpenAI API key.")
        st.stop()
    if not renderer:
        st.error("Primero carga Desktop.")
        st.stop()
    if not frames:
        st.error("Primero procesa el video App.")
        st.stop()

    desktop_obs = []
    app_obs = []

    desk_urls = renderer["image_urls"][:max_desk]

    p1 = st.progress(0, text="Analizando Desktop...")
    for i, u in enumerate(desk_urls):
        try:
            b, _ = download_image(u)
            items = analyze_image_with_ai(b, api_key)
            for item in items:
                item["_source"] = f"Desktop #{i+1}"
                desktop_obs.append(item)
        except Exception as e:
            st.warning(f"No pude analizar Desktop #{i+1}: {e}")
        p1.progress((i + 1) / len(desk_urls), text=f"Desktop {i+1}/{len(desk_urls)}")

    p2 = st.progress(0, text="Analizando App...")
    for i, fr in enumerate(frames):
        try:
            items = analyze_image_with_ai(fr["bytes"], api_key)
            for item in items:
                item["_source"] = f"App {fr['time']}s"
                app_obs.append(item)
        except Exception as e:
            st.warning(f"No pude analizar frame App {fr['time']}s: {e}")
        p2.progress((i + 1) / len(frames), text=f"App {i+1}/{len(frames)}")

    st.session_state["desktop_obs"] = desktop_obs
    st.session_state["app_obs"] = app_obs
    st.success(f"IA lista: {len(desktop_obs)} observaciones Desktop + {len(app_obs)} observaciones App")

desktop_obs = st.session_state.get("desktop_obs", [])
app_obs = st.session_state.get("app_obs", [])

if desktop_obs:
    with st.expander("Qué entendió la IA de Desktop"):
        st.dataframe(pd.DataFrame(desktop_obs), use_container_width=True)
if app_obs:
    with st.expander("Qué entendió la IA de App"):
        st.dataframe(pd.DataFrame(app_obs), use_container_width=True)

# ---------------- Cross QA
st.subheader("5. Cruce QA")

if st.button("Generar cruce Brief ↔ Desktop ↔ App"):
    if not desktop_obs or not app_obs:
        st.error("Primero ejecuta el análisis visual con IA.")
        st.stop()

    rows = []

    for _, r in base.iterrows():
        expected = clean(r[call])
        d = best_match(expected, desktop_obs)
        a = best_match(expected, app_obs)
        cross_status, cross_score, cross_reason = cross_desk_app(d, a)

        # Overall
        if d["status"] == "🔴 ERROR" or a["status"] == "🔴 ERROR" or cross_status == "🔴 DIFERENCIA":
            overall = "🔴 ERROR"
        elif d["status"] == "✅ OK" and a["status"] == "✅ OK" and cross_status == "✅ MISMA INFO":
            overall = "✅ OK"
        elif d["status"] == "👁️ NO DETECTADO" or a["status"] == "👁️ NO DETECTADO":
            overall = "👁️ NO DETECTADO"
        else:
            overall = "⚠️ REVISAR"

        rows.append({
            "Componente": clean(r[comp]),
            "Gerencia": clean(r[ger]) if ger else "",
            "Brief esperado": expected,

            "Desktop detectado": d["observed"],
            "Brief↔Desk": d["status"],
            "Score Desk": round(d["score"], 1),
            "Fuente Desk": d["source"],
            "Obs Desk": d["reason"],

            "App detectado": a["observed"],
            "Brief↔App": a["status"],
            "Score App": round(a["score"], 1),
            "Fuente App": a["source"],
            "Obs App": a["reason"],

            "Desk↔App": cross_status,
            "Score Desk/App": round(cross_score, 1),
            "Obs Desk/App": cross_reason,

            "Resultado": overall,
        })

    st.session_state["qa_final"] = pd.DataFrame(rows)

if "qa_final" in st.session_state:
    out = st.session_state["qa_final"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("✅ OK", int((out["Resultado"] == "✅ OK").sum()))
    c2.metric("🔴 Error", int((out["Resultado"] == "🔴 ERROR").sum()))
    c3.metric("⚠️ Revisar", int((out["Resultado"] == "⚠️ REVISAR").sum()))
    c4.metric("👁️ No detectado", int((out["Resultado"] == "👁️ NO DETECTADO").sum()))

    filter_mode = st.radio(
        "Mostrar",
        ["Solo errores/revisar", "Todo"],
        horizontal=True
    )
    shown = out
    if filter_mode == "Solo errores/revisar":
        shown = out[out["Resultado"] != "✅ OK"]

    st.dataframe(shown, use_container_width=True, hide_index=True)

    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        out.to_excel(writer, index=False, sheet_name="QA")

    st.download_button(
        "Descargar QA en Excel",
        data=buf.getvalue(),
        file_name="qa_home_resultado.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

st.divider()
st.caption(
    "v0.5: primero la IA interpreta el contenido de las piezas, luego cruza Brief↔Desktop, "
    "Brief↔App y Desktop↔App. Los scores son ayudas de QA, no sustituyen revisión humana cuando la confianza es baja."
)
