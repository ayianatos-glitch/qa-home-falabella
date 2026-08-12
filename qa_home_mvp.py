import io, re, json
import pandas as pd
import streamlit as st

st.set_page_config(page_title='QA Home', layout='wide')
st.title('QA Home — MVP')
st.caption('Excel → Desktop → App → diferencias para revisión')

STATUS = ['✅ OK','🔴 ERROR','⚠️ REVISAR','👁️ NO VISIBLE','⚪ NO VALIDAR']

def norm(x):
    if pd.isna(x): return ''
    s=str(x).lower().strip()
    s=re.sub(r'\s+',' ',s)
    return s

def extract_offer(s):
    s=norm(s)
    pct=re.findall(r'\b(\d{1,3})\s*%',s)
    prices=re.findall(r'\$?\s*(\d{1,3}(?:[\.]\d{3})+)',s)
    sku=re.findall(r'\b\d{7,9}\b',s)
    return {'pct':pct,'prices':prices,'sku':sku}

def compare(expected, observed):
    if not norm(observed): return '👁️ NO VISIBLE', 'Sin evidencia'
    e,o=norm(expected),norm(observed)
    eo,oo=extract_offer(e),extract_offer(o)
    if eo['pct'] and oo['pct'] and set(eo['pct']) != set(oo['pct']):
        return '🔴 ERROR', f"Descuento esperado {', '.join(eo['pct'])}% vs observado {', '.join(oo['pct'])}%"
    if eo['prices'] and oo['prices'] and not set(eo['prices']).intersection(oo['prices']):
        return '🔴 ERROR', 'Precio distinto'
    toks=[t for t in re.findall(r'[a-záéíóúñ0-9]+',e) if len(t)>3]
    overlap=sum(t in o for t in toks)/max(1,len(toks))
    if overlap >= .55: return '✅ OK', 'Coincidencia suficiente'
    if overlap >= .25: return '⚠️ REVISAR', 'Coincidencia parcial'
    return '⚠️ REVISAR', 'Contenido distinto o insuficiente'

excel=st.file_uploader('1. Sube la programación Excel', type=['xlsx'])
if not excel: st.stop()
book=pd.ExcelFile(excel)
sheet=st.selectbox('Hoja a validar', book.sheet_names, index=(book.sheet_names.index('Brief 12.08') if 'Brief 12.08' in book.sheet_names else 0))
df=pd.read_excel(excel, sheet_name=sheet)

# tolerate actual brief naming
comp=next((c for c in df.columns if str(c).strip().lower() in ['componente','posición','posicion']), None)
call=next((c for c in df.columns if str(c).strip().lower().startswith('llamado')), None)
geria=next((c for c in df.columns if str(c).strip().lower()=='gerencia'), None)
state=next((c for c in df.columns if str(c).strip().lower()=='estado'), None)
if not comp or not call:
    st.error('No encuentro las columnas Componente y Llamado en esta hoja.'); st.stop()
base=df[df[comp].notna()].copy()
if state: base=base[~base[state].astype(str).str.contains('PENDIENTE BRIEF',case=False,na=False)]

st.write(f'**{len(base)} filas con componente detectadas**')

st.subheader('2. Evidencia')
st.info('En esta versión inicial puedes pegar el texto detectado/observado por componente. El siguiente paso conecta visión automática a screenshots y al renderer.')

qa=pd.DataFrame({
    'Componente':base[comp].astype(str),
    'Gerencia':base[geria].fillna('').astype(str) if geria else '',
    'Excel esperado':base[call].fillna('').astype(str),
    'Desktop observado':'',
    'App observado':'',
})
qa=st.data_editor(qa, use_container_width=True, hide_index=True, num_rows='fixed')

if st.button('3. Analizar QA', type='primary'):
    rows=[]
    for _,r in qa.iterrows():
        ds,do=compare(r['Excel esperado'],r['Desktop observado'])
        aps,apo=compare(r['Excel esperado'],r['App observado'])
        overall='🔴 ERROR' if '🔴 ERROR' in (ds,aps) else ('⚠️ REVISAR' if '⚠️ REVISAR' in (ds,aps) else ('👁️ NO VISIBLE' if '👁️ NO VISIBLE' in (ds,aps) else '✅ OK'))
        rows.append({**r.to_dict(),'QA Desktop':ds,'QA App':aps,'Estado':overall,'Observación Desktop':do,'Observación App':apo})
    out=pd.DataFrame(rows)
    st.session_state['out']=out

if 'out' in st.session_state:
    out=st.session_state['out']
    c1,c2,c3,c4=st.columns(4)
    c1.metric('OK',(out['Estado']=='✅ OK').sum())
    c2.metric('Errores',(out['Estado']=='🔴 ERROR').sum())
    c3.metric('Revisar',(out['Estado']=='⚠️ REVISAR').sum())
    c4.metric('No visible',(out['Estado']=='👁️ NO VISIBLE').sum())
    st.dataframe(out, use_container_width=True, hide_index=True)
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as w: out.to_excel(w,index=False,sheet_name='QA')
    st.download_button('Descargar reporte QA Excel',buf.getvalue(),'reporte_qa.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
