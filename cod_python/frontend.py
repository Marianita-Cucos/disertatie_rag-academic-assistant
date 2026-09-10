# streamlit run frontend.py

import streamlit as st
import requests
import pandas as pd  

API_URL = "http://backend:8000" 
# API_URL = "https://4455-34-125-118-173.ngrok-free.app"

st.set_page_config(page_title="Sistem Academic RAG", layout="wide")

# ==========================================
# MENIU DE NAVIGARE (SIDEBAR)
# ==========================================
st.sidebar.title("📌 Navigare")
pagina_curenta = st.sidebar.radio(
    "Alege secțiunea:",
    ["💬 Chat Academic", "📊 Dashboard Evaluare RAGAS"]
)
st.sidebar.divider()

# ==============================================================================
# SECȚIUNEA 1: CHAT ACADEMIC
# ==============================================================================
if pagina_curenta == "💬 Chat Academic":
    st.title("📚 Sistem Academic RAG")

    st.sidebar.header("🔐 Autentificare")

    # Gestionarea stării de autentificare în sesiune
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.user_id = ""

    if not st.session_state.logged_in:
        auth_mode = st.sidebar.radio("Acțiune", ["Log In", "Înregistrare"], label_visibility="collapsed")
        
        if auth_mode == "Log In":
            st.sidebar.subheader("🔑 Intră în cont")
            login_user_input = st.sidebar.text_input("Username", key="login_user")
            login_pass_input = st.sidebar.text_input("Parolă", type="password", key="login_pass")
            
            if st.sidebar.button("Autentificare"):
                try:
                    res = requests.post(f"{API_URL}/login", json={"username": login_user_input, "password": login_pass_input})
                    if res.status_code == 200:
                        st.session_state.logged_in = True
                        st.session_state.user_id = login_user_input
                        st.rerun()
                    else:
                        st.sidebar.error(res.json().get("detail", "Eroare la autentificare"))
                except Exception as e:
                    st.sidebar.error(f"Eroare de conexiune: {e}")
                    
        else:
            st.sidebar.subheader("📝 Cont nou")
            reg_user_input = st.sidebar.text_input("Username nou", key="reg_user")
            reg_pass_input = st.sidebar.text_input("Parolă nouă", type="password", key="reg_pass")
            
            if st.sidebar.button("Creează Cont"):
                try:
                    res = requests.post(f"{API_URL}/register", json={"username": reg_user_input, "password": reg_pass_input})
                    if res.status_code == 200:
                        st.sidebar.success("Cont creat! Acum te poți loga.")
                    else:
                        st.sidebar.error(res.json().get("detail", "Eroare la înregistrare"))
                except Exception as e:
                    st.sidebar.error(f"Eroare de conexiune: {e}")
                    
        st.sidebar.warning("⚠️ Autentifică-te pentru a continua.")
        st.stop()
    else:
        user_id = st.session_state.user_id
        st.sidebar.success(f"Logat ca: **{user_id}**")
        if st.sidebar.button("Log Out"):
            st.session_state.logged_in = False
            st.session_state.user_id = ""
            st.rerun()
        st.sidebar.divider()

    st.sidebar.header("📤 Încărcare Curs Nou")
    uploaded_file = st.sidebar.file_uploader("Încarcă un fișier PDF", type=["pdf"])

    if st.sidebar.button("Procesează Cursul"):
        if uploaded_file is not None:
            with st.spinner("Se procesează și se indexează..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"user_id": user_id}
                try:
                    headers = {"ngrok-skip-browser-warning": "true"}
                    response = requests.post(f"{API_URL}/upload-course", files=files, data=data, headers=headers)
                    if response.status_code == 202:
                        st.sidebar.success(f"Fișierul a fost trimis cu succes pentru {user_id}!")
                    else:
                        try:
                            mesaj_eroare = response.json().get('detail', 'Necunoscută')
                        except Exception:
                            mesaj_eroare = response.text 
                        
                        st.sidebar.error(f"Eroare server (Cod {response.status_code}): {mesaj_eroare}")
                except Exception as e:
                    st.sidebar.error(f"Eroare de conexiune la API: {e}")
        else:
            st.sidebar.warning("Te rog să selectezi un fișier mai întâi.")

    st.sidebar.divider()

    if st.sidebar.button("🔄 Vezi cursurile mele"):
        try:
            headers = {"ngrok-skip-browser-warning": "true"}
            response = requests.get(f"{API_URL}/cursuri_incarcate", params={"user_id": user_id}, headers=headers)
            if response.status_code == 200:
                cursuri = response.json().get("cursuri", [])
                if cursuri:
                    st.sidebar.write("### Cursurile Tale:")
                    for curs in cursuri:
                        st.sidebar.write(f"- 📄 {curs}")
                else:
                    st.sidebar.info("Nu ai niciun curs indexat momentan.")
            else:
                st.sidebar.error("A apărut o eroare la obținerea cursurilor.")
        except Exception as e:
            st.sidebar.error(f"Eroare de conexiune: {e}")

    # --- INTERFAȚA DE CHAT ---
    st.subheader(f"💬 Adresează o întrebare din materialele tale")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("Surse extrase"):
                    st.json(msg["sources"])

    if prompt := st.chat_input("Ex: Ce este un arbore binar?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Se analizează contextul..."):
                try:
                    payload = {"question": prompt, "user_id": user_id}
                    headers = {"ngrok-skip-browser-warning": "true"}
                    response = requests.post(f"{API_URL}/ask-question", json=payload, headers=headers)
                    
                    if response.status_code == 200:
                        date_raspuns = response.json()
                        raspuns_text = date_raspuns.get("answer", "")
                        surse = date_raspuns.get("sources", [])
                        status_agent = date_raspuns.get("status", "")
                        
                        st.markdown(raspuns_text)
                        
                        if status_agent == "success_cache":
                            st.caption("⚡ Răspuns servit instant din Redis Cache")
                        elif status_agent == "lipsa_context_cache":
                            st.caption("⚡ 🚫 Lipsă context (Servit din Redis Cache)")
                        elif status_agent == "lipsa_context":
                            st.caption("🚫 Informația nu a fost găsită în surse")
                        elif status_agent == "success_keyword_search":
                            st.caption("🔍 Răspuns obținut prin Fallback Lexical (Cuvinte-cheie)")
                        elif status_agent == "fallback_applied":
                            st.caption("🚨 Fallback Determinist Aplicat (Doar referințe)")
                        
                        if surse:
                            with st.expander("Vezi metadatele surselor"):
                                st.json(surse)
                                
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": raspuns_text, 
                            "sources": surse
                        })
                    else:
                        err_msg = response.json().get("detail", "Eroare necunoscută")
                        st.error(f"Eroare API: {err_msg}")
                        
                except Exception as e:
                    st.error(f"Eroare de comunicare cu serverul: {e}")

# ==============================================================================
# SECȚIUNEA 2: DASHBOARD EVALUARE RAGAS
# ==============================================================================
elif pagina_curenta == "📊 Dashboard Evaluare RAGAS":
    st.title("📊 Analiza de Performanță a Sistemului RAG")
    st.write("Încarcă fișierul CSV generat de modulul RAGAS pentru a vizualiza performanța sistemului în studiile de ablațiune.")

    fisier_csv = st.file_uploader("Încarcă rezultate_evaluare_ragas.csv", type=["csv"])

    if fisier_csv is not None:
        try:
            df = pd.read_csv(fisier_csv)
            df_numeric = df[['faithfulness', 'answer_relevancy', 'context_precision']].fillna(0)
            
            st.success(f"Fișier încărcat cu succes! {len(df)} întrebări analizate.")
            st.divider()

            st.subheader("📈 Scoruri Medii Globale")
            col1, col2, col3 = st.columns(3)
            
            medie_faith = df_numeric['faithfulness'].mean()
            medie_rel = df_numeric['answer_relevancy'].mean()
            medie_ctx = df_numeric['context_precision'].mean()

            col1.metric("Fidelitate (Faithfulness)", f"{medie_faith:.2f} / 1.0", 
                        help="Măsoară câte din afirmațiile generate pot fi deduse direct din context.")
            col2.metric("Relevanță (Answer Relevancy)", f"{medie_rel:.2f} / 1.0",
                        help="Măsoară cât de direct a răspuns sistemul la întrebarea pusă.")
            col3.metric("Precizia Contextului (Context Precision)", f"{medie_ctx:.2f} / 1.0",
                        help="Măsoară dacă informația corectă a fost extrasă pe primele locuri.")

            st.divider()

            col_grafic1, col_grafic2 = st.columns(2)

            with col_grafic1:
                st.subheader("Comparație Metrici (Medii)")
                df_medii = pd.DataFrame({
                    "Scor": [medie_faith, medie_rel, medie_ctx]
                }, index=["Faithfulness", "Answer Relevancy", "Context Precision"])
                st.bar_chart(df_medii, color="#2E86C1")

            with col_grafic2:
                st.subheader("Distribuția Scorurilor per Întrebare")
                st.line_chart(df_numeric)

            st.divider()

            st.subheader("🔍 Detalii per Întrebare")
            coloane_de_afisat = ['question', 'answer', 'faithfulness', 'answer_relevancy', 'context_precision']
            
            if 'latency_seconds' in df.columns:
                coloane_de_afisat.append('latency_seconds')
            if 'agent_status' in df.columns:
                coloane_de_afisat.append('agent_status')

            st.dataframe(
                df[coloane_de_afisat].style.highlight_min(
                    subset=['faithfulness', 'answer_relevancy', 'context_precision'], 
                    color='lightcoral'
                ),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"A apărut o eroare la procesarea fișierului CSV: {e}")