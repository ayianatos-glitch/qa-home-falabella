import io, re, requests
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup

st.set_page_config(page_title='QA Home', layout='wide')
st.title('QA Home — MVP')
st.caption('Excel → Desktop Renderer → App → diferencias para revisión')

def norm(x):
    if pd.isna(x): return ''
    return re.sub(r'\s+', ' ', str(x).lower().strip())

def compare(expected, observed):
    if not norm(observed):
        return '👁️ NO VISIBLE', 'Sin evidencia'
    e, o = norm(expected), norm(observed)
    ep = re.findall(r'\b(\d{1,3})\s*%', e)
    op = re.findall(r'\b(\d{1,3})\s*%', o)
    if ep and op and set(ep) != set(op):
        return '🔴 ERROR', f"Descuento esperado {', '.join(ep)}% vs observado {', '.join(op)}%"
    prices_e = re.findall(r'\$?\s*(\d{1,3}(?:[\.]\d{3})+)', e)
    prices_o = re.findall(r'\$?\s*(\d{1,3}(?:[\.]\d{3})+)', o)
    if prices_e and prices_o and not set(prices_e).intersection(prices_o):
        return '🔴 ERROR', 'Precio distinto'
    toks = [t for t in re.findall(r'[a-záéíóúñ0-9]+', e) if len(t) > 3]
    overlap = sum(t in o for t in toks) / max(1, len(toks))
    if overlap >= .55: return '✅ OK', 'Coincidencia suficiente'
    if overlap >= .25: return '⚠️ REVISAR', 'Coincidencia parcial'
    return '⚠️ REVISAR', 'Contenido distinto o insuficiente'

@st.cache_data(ttl=300, show_spinner=False)
def fetch_renderer(url):
    r = requests.get(url, timeout=30, headers={'User-Agent':'Mozilla/5.0'})
    r.raise_for_status()
    soup = BeautifulSoup(r.text, 'html.parser')
    parts = list(soup.stripped_strings)
    for tag in soup.find_all(True):
        for attr in ('alt','title','aria-label'):
            if tag.get(attr): parts.append(str(tag.get(attr)))
    text = re.sub(r'\s+', ' ', ' '.join(parts)).strip()
    return r.status_code, text

st.subheader('1. Programación Excel')
excel = st.file_uploader('Sube la programación Excel', type=['xlsx'])
if not excel: st.stop()

book = pd.ExcelFile(excel)
idx = book.sheet_names.index('Brief 12.08') if 'Brief 12.08' in book.sheet_names else 0
sheet = st.selectbox('Hoja a validar', book.sheet_names, index=idx)
df = pd.read_excel(excel, sheet_name=sheet)

comp = next((c for c in df.columns if str(c).strip().lower() in ['componente','posición','posicion']), None)
call = next((c for c in df.columns if str(c).strip().lower().startswith('llamado')), None)
ger = next((c for c in df.columns if str(c).strip().lower()=='gerencia'), None)

if not comp or not call:
    st.error('No encuentro las columnas Componente y Llamado.')
    st.stop()

base = df[df[comp].notna()].copy()
st.success(f'{len(base)} filas con componente detectadas')

st.subheader('2. Desktop — Renderer')
renderer_url = st.text_input('Pega aquí la URL del Renderer Desktop', placeholder='https://renderer.falabella.com/preview/...')

if st.button('Leer Renderer Desktop'):
    if not renderer_url:
        st.warning('Pega primero la URL del renderer.')
    else:
        try:
            with st.spinner('Leyendo renderer...'):
                status, text = fetch_renderer(renderer_url)
            st.session_state['renderer_text'] = text
            st.success(f'Renderer leído correctamente (HTTP {status}).')
            st.metric('Texto detectado', f'{len(text):,} caracteres')
        except Exception as e:
            st.error(f'No pude leer el renderer: {e}')

renderer_text = st.session_state.get('renderer_text','')
if renderer_text:
    with st.expander('Ver muestra de lo detectado en Desktop'):
        st.text(renderer_text[:4000])

st.subheader('3. Evidencia App')
st.info('Por ahora, App se ingresa manualmente. Desktop sí se valida desde el Renderer.')

qa = pd.DataFrame({
    'Componente': base[comp].astype(str),
    'Gerencia': base[ger].fillna('').astype(str) if ger else '',
    'Excel esperado': base[call].fillna('').astype(str),
    'App observado': ''
})
qa = st.data_editor(qa, use_container_width=True, hide_index=True, num_rows='fixed')

if st.button('4. Analizar QA', type='primary'):
    rows=[]
    for _, r in qa.iterrows():
        ds, do = compare(r['Excel esperado'], renderer_text) if renderer_text else ('👁️ NO VISIBLE','Renderer no cargado')
        aps, apo = compare(r['Excel esperado'], r['App observado'])
        if '🔴 ERROR' in (ds,aps): state='🔴 ERROR'
        elif '⚠️ REVISAR' in (ds,aps): state='⚠️ REVISAR'
        elif '👁️ NO VISIBLE' in (ds,aps): state='👁️ NO VISIBLE'
        else: state='✅ OK'
        rows.append({**r.to_dict(),'QA Desktop':ds,'QA App':aps,'Estado':state,
                     'Observación Desktop':do,'Observación App':apo})
    st.session_state['out']=pd.DataFrame(rows)

if 'out' in st.session_state:
    out=st.session_state['out']
    c1,c2,c3,c4=st.columns(4)
    c1.metric('OK',(out['Estado']=='✅ OK').sum())
    c2.metric('Errores',(out['Estado']=='🔴 ERROR').sum())
    c3.metric('Revisar',(out['Estado']=='⚠️ REVISAR').sum())
    c4.metric('No visible',(out['Estado']=='👁️ NO VISIBLE').sum())
    st.dataframe(out,use_container_width=True,hide_index=True)
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w:
        out.to_excel(w,index=False,sheet_name='QA')
    st.download_button('Descargar reporte QA Excel',buf.getvalue(),'reporte_qa.xlsx',
                       'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

st.caption('MVP actual: el Renderer se lee por HTML/texto. El texto dibujado dentro de imágenes aún no se reconoce automáticamente.')
