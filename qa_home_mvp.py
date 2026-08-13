
import io, re, tempfile
from pathlib import Path
from urllib.parse import urljoin

import cv2, imagehash, numpy as np, pandas as pd, pytesseract, requests, streamlit as st
from bs4 import BeautifulSoup
from PIL import Image
from rapidfuzz import fuzz

st.set_page_config(page_title="QA Home", layout="wide")
st.title("QA Home — MVP v0.6 gratis")
st.caption("Brief ↔ Desktop ↔ App con OCR + similitud visual. Sin OpenAI API.")

def clean(x):
    if pd.isna(x): return ""
    return re.sub(r"\s+"," ",str(x)).strip()

def norm(x): return clean(x).lower()

def nums(t):
    s=norm(t)
    pct=re.findall(r"\b(\d{1,3})\s*%",s)
    price=[re.sub(r"\D","",x) for x in re.findall(r"\$?\s*(\d{1,3}(?:[.\s]\d{3})+)",s)]
    return pct,price

def sem(a,b):
    a,b=norm(a),norm(b)
    if not a or not b: return 0.0
    return float(max(fuzz.token_set_ratio(a,b),fuzz.partial_ratio(a,b),fuzz.token_sort_ratio(a,b)))

def pil(b): return Image.open(io.BytesIO(b)).convert("RGB")

def ocr(b):
    im=pil(b)
    if im.width<1200:
        s=1200/max(1,im.width); im=im.resize((1200,int(im.height*s)))
    g=cv2.cvtColor(np.array(im),cv2.COLOR_RGB2GRAY)
    g=cv2.equalizeHist(g)
    g=cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,41,11)
    try: txt=pytesseract.image_to_string(g,lang="spa+eng",config="--psm 6")
    except: txt=pytesseract.image_to_string(g,config="--psm 6")
    return clean(txt)

def phash_sim(a,b):
    try:
        d=imagehash.phash(pil(a))-imagehash.phash(pil(b))
        return max(0,100*(1-d/64))
    except: return 0

def orb_sim(a,b):
    try:
        A=cv2.cvtColor(np.array(pil(a)),cv2.COLOR_RGB2GRAY)
        B=cv2.cvtColor(np.array(pil(b)),cv2.COLOR_RGB2GRAY)
        def resize(x):
            h,w=x.shape[:2]; m=max(h,w)
            return x if m<=1200 else cv2.resize(x,(int(w*1200/m),int(h*1200/m)))
        A,B=resize(A),resize(B)
        orb=cv2.ORB_create(nfeatures=1600)
        k1,d1=orb.detectAndCompute(A,None); k2,d2=orb.detectAndCompute(B,None)
        if d1 is None or d2 is None or len(k1)<8 or len(k2)<8: return 0
        matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
        good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<0.72*n.distance]
        return min(100,len(good)/max(1,min(len(k1),len(k2)))*650)
    except: return 0

def visual(a,b,ta="",tb=""):
    o,p,t=orb_sim(a,b),phash_sim(a,b),sem(ta,tb)
    return round(min(100,.60*o+.15*p+.25*t),1),round(o,1),round(p,1),round(t,1)

def brief_match(expected,observed):
    s=sem(expected,observed); ep,er=nums(expected); op,or_=nums(observed)
    if ep and op and set(ep).isdisjoint(op) and s>=45:
        return "🔴 ERROR",s,f"% esperado {','.join(ep)} vs observado {','.join(op)}"
    if er and or_ and set(er).isdisjoint(or_) and s>=45:
        return "🔴 ERROR",s,"precio distinto"
    if s>=72:return "✅ OK",s,"Coincidencia alta"
    if s>=48:return "⚠️ REVISAR",s,"Coincidencia parcial"
    return "👁️ NO DETECTADO",s,"Sin coincidencia suficiente"

@st.cache_data(ttl=300,show_spinner=False)
def renderer_imgs(url):
    r=requests.get(url,timeout=30,headers={"User-Agent":"Mozilla/5.0"}); r.raise_for_status()
    soup=BeautifulSoup(r.text,"html.parser"); urls=[]
    for tag in soup.find_all(["img","source"]):
        vals=[tag.get("src"),tag.get("data-src")]
        for attr in ("srcset","data-srcset"):
            if tag.get(attr): vals += [x.strip().split(" ")[0] for x in tag.get(attr).split(",")]
        for v in vals:
            if v:
                u=urljoin(url,v)
                if u not in urls: urls.append(u)
    for tag in soup.find_all(style=True):
        for v in re.findall(r'url\(["\']?(.*?)["\']?\)',tag.get("style","")):
            u=urljoin(url,v)
            if u not in urls: urls.append(u)
    return urls

@st.cache_data(ttl=300,show_spinner=False)
def dl(url):
    r=requests.get(url,timeout=25,headers={"User-Agent":"Mozilla/5.0"});r.raise_for_status();return r.content

@st.cache_data(show_spinner=False)
def frames(video_bytes,suffix,every=1.5,maxn=30):
    with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f:
        f.write(video_bytes); path=f.name
    cap=cv2.VideoCapture(path); fps=cap.get(cv2.CAP_PROP_FPS) or 30
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur=count/fps if fps else 0
    out=[]; t=0
    while t<=dur and len(out)<maxn:
        cap.set(cv2.CAP_PROP_POS_MSEC,t*1000); ok,fr=cap.read()
        if not ok: break
        fr=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB); im=Image.fromarray(fr)
        if im.width>1200: im=im.resize((1200,int(im.height*1200/im.width)))
        buf=io.BytesIO(); im.save(buf,format="JPEG",quality=90)
        out.append({"source":f"App {t:.1f}s","time":round(t,1),"bytes":buf.getvalue()}); t+=every
    cap.release(); Path(path).unlink(missing_ok=True); return out

st.subheader("1. Brief")
excel=st.file_uploader("Sube Excel",type=["xlsx"])
if not excel: st.stop()
book=pd.ExcelFile(excel); idx=book.sheet_names.index("Brief 12.08") if "Brief 12.08" in book.sheet_names else 0
sheet=st.selectbox("Hoja",book.sheet_names,index=idx); df=pd.read_excel(excel,sheet_name=sheet)
comp=next((c for c in df.columns if str(c).strip().lower() in ["componente","posición","posicion"]),None)
call=next((c for c in df.columns if str(c).strip().lower().startswith("llamado")),None)
ger=next((c for c in df.columns if str(c).strip().lower()=="gerencia"),None)
state=next((c for c in df.columns if str(c).strip().lower()=="estado"),None)
if not comp or not call: st.error("Faltan columnas Componente/Llamado"); st.stop()
base=df[df[comp].notna()].copy()
if state: base=base[~base[state].astype(str).str.contains("PENDIENTE BRIEF",case=False,na=False)]
st.success(f"{len(base)} piezas detectadas")

st.subheader("2. Desktop")
url=st.text_input("URL Renderer Desktop")
shots=st.file_uploader("Opcional: screenshots Desktop",type=["png","jpg","jpeg","webp"],accept_multiple_files=True)
if st.button("Preparar Desktop"):
    ev=[]
    if url:
        try:
            urls=renderer_imgs(url)
            for i,u in enumerate(urls[:80]):
                try: ev.append({"source":f"Renderer #{i+1}","bytes":dl(u)})
                except: pass
            st.info(f"Renderer: {len(ev)} imágenes descargadas")
        except Exception as e: st.warning(f"Renderer: {e}")
    for f in shots or []: ev.append({"source":f.name,"bytes":f.getvalue()})
    st.session_state["desk"]=ev; st.success(f"Desktop listo: {len(ev)} evidencias")

desk=st.session_state.get("desk",[])
if desk:
    with st.expander("Ver Desktop"):
        cols=st.columns(4)
        for i,x in enumerate(desk[:40]): cols[i%4].image(x["bytes"],caption=x["source"],use_container_width=True)

st.subheader("3. App — Video")
video=st.file_uploader("Sube video App",type=["mp4","mov","m4v"])
if video:
    every=st.slider("Frame cada",0.5,4.0,1.5,0.5); maxn=st.slider("Máximo frames",5,60,30,1)
    if st.button("Procesar video"):
        st.session_state["app"]=frames(video.getvalue(),Path(video.name).suffix or ".mp4",every,maxn)
        st.success(f"{len(st.session_state['app'])} frames")
app=st.session_state.get("app",[])
if app:
    with st.expander("Ver frames App"):
        cols=st.columns(4)
        for i,x in enumerate(app): cols[i%4].image(x["bytes"],caption=x["source"],use_container_width=True)

st.subheader("4. OCR gratis")
if st.button("Analizar Desktop + App"):
    if not desk or not app: st.error("Prepara Desktop y App primero"); st.stop()
    d=[]; p=st.progress(0)
    for i,x in enumerate(desk):
        d.append({**x,"ocr":ocr(x["bytes"])}); p.progress((i+1)/len(desk))
    a=[]; p2=st.progress(0)
    for i,x in enumerate(app):
        a.append({**x,"ocr":ocr(x["bytes"])}); p2.progress((i+1)/len(app))
    st.session_state["deskocr"]=d; st.session_state["appocr"]=a; st.success("OCR listo")

docr=st.session_state.get("deskocr",[]); aocr=st.session_state.get("appocr",[])
if docr:
    with st.expander("Texto Desktop"): st.dataframe(pd.DataFrame([{"Fuente":x["source"],"OCR":x["ocr"]} for x in docr]),use_container_width=True)
if aocr:
    with st.expander("Texto App"): st.dataframe(pd.DataFrame([{"Fuente":x["source"],"OCR":x["ocr"]} for x in aocr]),use_container_width=True)

st.subheader("5. Cruce Brief ↔ Desktop ↔ App")
if st.button("Generar QA",type="primary"):
    if not docr or not aocr: st.error("Ejecuta OCR primero"); st.stop()

    # Desktop -> App best visual match
    pairs={}
    p=st.progress(0)
    for di,d in enumerate(docr):
        best={"score":0}
        for ai,a in enumerate(aocr):
            sc,o,ph,tx=visual(d["bytes"],a["bytes"],d["ocr"],a["ocr"])
            if sc>best["score"]: best={"score":sc,"orb":o,"phash":ph,"text":tx,"app":a}
        pairs[di]=best; p.progress((di+1)/len(docr))

    rows=[]
    for _,r in base.iterrows():
        exp=clean(r[call])
        dc=[]
        for di,d in enumerate(docr):
            stt,sc,rs=brief_match(exp,d["ocr"]); dc.append((sc,di,stt,rs,d))
        ac=[]
        for ai,a in enumerate(aocr):
            stt,sc,rs=brief_match(exp,a["ocr"]); ac.append((sc,ai,stt,rs,a))
        db=max(dc,key=lambda x:x[0]); ab=max(ac,key=lambda x:x[0])
        vis=pairs.get(db[1],{"score":0}); vscore=vis["score"]; vap=vis.get("app",{})
        vstat="✅ MISMA GRÁFICA" if vscore>=62 else ("⚠️ POSIBLE MATCH" if vscore>=38 else "❌ NO COINCIDE")
        dd,dp=nums(db[4]["ocr"]); ad,ap=nums(vap.get("ocr","")); issue=""
        if dd and ad and set(dd).isdisjoint(ad): issue=f"% distinto Desk/App ({','.join(dd)} vs {','.join(ad)})"
        elif dp and ap and set(dp).isdisjoint(ap): issue="Precio distinto Desk/App"
        if issue and vscore>=38: vstat="🔴 MISMA PIEZA / INFO DISTINTA"
        if "🔴 ERROR" in (db[2],ab[2]) or vstat.startswith("🔴"): final="🔴 ERROR"
        elif db[2]=="✅ OK" and ab[2]=="✅ OK" and vstat=="✅ MISMA GRÁFICA": final="✅ OK"
        elif "👁️ NO DETECTADO" in (db[2],ab[2]): final="👁️ NO DETECTADO"
        else: final="⚠️ REVISAR"
        rows.append({
            "Componente":clean(r[comp]),"Gerencia":clean(r[ger]) if ger else "","Brief esperado":exp,
            "Desktop detectado":db[4]["ocr"],"Brief↔Desk":db[2],"Score Brief/Desk":round(db[0],1),"Fuente Desktop":db[4]["source"],"Obs Desktop":db[3],
            "App detectado":ab[4]["ocr"],"Brief↔App":ab[2],"Score Brief/App":round(ab[0],1),"Fuente App":ab[4]["source"],"Obs App":ab[3],
            "Desk↔App":vstat,"Score visual":vscore,"ORB":vis.get("orb",0),"pHash":vis.get("phash",0),"Texto Desk/App":vis.get("text",0),
            "App match visual":vap.get("source",""),"Obs Desk/App":issue or "Comparación visual + OCR","Resultado":final
        })
    st.session_state["out"]=pd.DataFrame(rows)

if "out" in st.session_state:
    out=st.session_state["out"]; c1,c2,c3,c4=st.columns(4)
    c1.metric("✅ OK",int((out["Resultado"]=="✅ OK").sum()))
    c2.metric("🔴 Error",int((out["Resultado"]=="🔴 ERROR").sum()))
    c3.metric("⚠️ Revisar",int((out["Resultado"]=="⚠️ REVISAR").sum()))
    c4.metric("👁️ No detectado",int((out["Resultado"]=="👁️ NO DETECTADO").sum()))
    mode=st.radio("Mostrar",["Solo problemas","Todo"],horizontal=True)
    shown=out if mode=="Todo" else out[out["Resultado"]!="✅ OK"]
    st.dataframe(shown,use_container_width=True,hide_index=True)
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine="openpyxl") as w: out.to_excel(w,index=False,sheet_name="QA")
    st.download_button("Descargar QA Excel",buf.getvalue(),"qa_home_gratis.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

st.caption("Motor gratuito: Tesseract OCR + RapidFuzz + ORB + pHash. Los casos dudosos quedan en REVISAR.")
