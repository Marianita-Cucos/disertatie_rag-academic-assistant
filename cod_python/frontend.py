import streamlit as st
import requests
import pandas as pd  

# Deoarece frontend-ul rulează în Docker, comunică cu backend-ul folosind numele serviciului
API_URL = "http://backend:8000" 

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
# SECȚIUNEA 1: CHAT ACADEMIC (Codul tău original)
# ==============================================================================
if pagina_curenta == "💬 Chat Academic":
    st.title("📚 Sistem Academic RAG (Multi-Tenant)")

    st.sidebar.header("🔐 Autentificare")
    user_id = st.sidebar.text_input("Introdu ID Utilizator (ex: Student_A):", value="Student_A")

    if not user_id:
        st.warning("⚠️ Te rog să introduci un ID de utilizator pentru a accesa platforma.")
        st.stop()

    st.sidebar.success(f"Logat ca: **{user_id}**")
    st.sidebar.divider()

    st.sidebar.header("📤 Încărcare Curs Nou")
    uploaded_file = st.sidebar.file_uploader("Încarcă un fișier PDF", type=["pdf"])

    if st.sidebar.button("Procesează Cursul"):
        if uploaded_file is not None:
            with st.spinner("Se procesează și se indexează..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                data = {"user_id": user_id}
                try:
                    response = requests.post(f"{API_URL}/upload-course", files=files, data=data)
                    if response.status_code == 202:
                        st.sidebar.success(f"Fișierul a fost trimis cu succes pentru {user_id}!")
                    else:
                        st.sidebar.error(f"Eroare: {response.json().get('detail', 'Necunoscută')}")
                except Exception as e:
                    st.sidebar.error(f"Eroare de conexiune la API: {e}")
        else:
            st.sidebar.warning("Te rog să selectezi un fișier mai întâi.")

    st.sidebar.divider()

    if st.sidebar.button("🔄 Vezi cursurile mele"):
        try:
            response = requests.get(f"{API_URL}/cursuri_incarcate", params={"user_id": user_id})
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
                    response = requests.post(f"{API_URL}/ask-question", json=payload)
                    
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
# SECȚIUNEA 2: DASHBOARD EVALUARE RAGAS (Noua componentă)
# ==============================================================================
elif pagina_curenta == "📊 Dashboard Evaluare RAGAS":
    st.title("📊 Analiza de Performanță a Sistemului RAG")
    st.write("Încarcă fișierul CSV generat de modulul RAGAS pentru a vizualiza performanța sistemului în studiile de ablațiune.")

    fisier_csv = st.file_uploader("Încarcă rezultate_evaluare_ragas.csv", type=["csv"])

    if fisier_csv is not None:
        try:
            # Citim fișierul CSV
            df = pd.read_csv(fisier_csv)
            
            # Curățăm eventualele valori NaN punând 0 pentru a nu strica graficele
            df_numeric = df[['faithfulness', 'answer_relevancy', 'context_precision']].fillna(0)
            
            st.success(f"Fișier încărcat cu succes! {len(df)} întrebări analizate.")
            st.divider()

            # 1. METRICI GLOBALE (Medii)
            st.subheader("📈 Scoruri Medii Globale")
            col1, col2, col3 = st.columns(3)
            
            medie_faith = df_numeric['faithfulness'].mean()
            medie_rel = df_numeric['answer_relevancy'].mean()
            medie_ctx = df_numeric['context_precision'].mean()

            # Afișare tip widget metric
            col1.metric("Fidelitate (Faithfulness)", f"{medie_faith:.2f} / 1.0", 
                        help="Măsoară câte din afirmațiile generate pot fi deduse direct din context.")
            col2.metric("Relevanță (Answer Relevancy)", f"{medie_rel:.2f} / 1.0",
                        help="Măsoară cât de direct a răspuns sistemul la întrebarea pusă.")
            col3.metric("Precizia Contextului (Context Precision)", f"{medie_ctx:.2f} / 1.0",
                        help="Măsoară dacă informația corectă a fost extrasă pe primele locuri.")

            st.divider()

            # 2. GRAFICE VIZUALE
            col_grafic1, col_grafic2 = st.columns(2)

            with col_grafic1:
                st.subheader("Comparație Metrici (Medii)")
                # Creăm un mic dataframe pentru Bar Chart
                df_medii = pd.DataFrame({
                    "Scor": [medie_faith, medie_rel, medie_ctx]
                }, index=["Faithfulness", "Answer Relevancy", "Context Precision"])
                st.bar_chart(df_medii, color="#2E86C1")

            with col_grafic2:
                st.subheader("Distribuția Scorurilor per Întrebare")
                # Grafic de tip linie/arie care arată evoluția pe fiecare întrebare
                st.line_chart(df_numeric)

            st.divider()

            # 3. DATE BRUTE (TABEL)
            st.subheader("🔍 Detalii per Întrebare")
            
            # Ascundem coloana de context care este prea lungă pentru o vizualizare curată, lăsând esențialul
            coloane_de_afisat = ['question', 'answer', 'faithfulness', 'answer_relevancy', 'context_precision']
            
            # Adăugăm coloanele de latență și status dacă există (din ultimele tale modificări)
            if 'latency_seconds' in df.columns:
                coloane_de_afisat.append('latency_seconds')
            if 'agent_status' in df.columns:
                coloane_de_afisat.append('agent_status')

            # Evidențiem valorile scăzute în tabel (opțional, pentru efect wow)
            st.dataframe(
                df[coloane_de_afisat].style.highlight_min(
                    subset=['faithfulness', 'answer_relevancy', 'context_precision'], 
                    color='lightcoral'
                ),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"A apărut o eroare la procesarea fișierului CSV: {e}")