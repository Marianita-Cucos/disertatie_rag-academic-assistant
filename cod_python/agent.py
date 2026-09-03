import re
import time
import os
import hashlib
import json
import math
import redis
import google.generativeai as genai
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

class TutorialRAGAgent:
    def __init__(self, vector_db, api_key, use_reranker=True, use_context_framing=True, use_fallback=True, use_cache=True):
        self.vector_db = vector_db
        self.max_retries = 2

        # --- VARIABILE DE CONTROL (A/B TESTING) ---
        self.use_reranker = use_reranker
        self.use_context_framing = use_context_framing
        self.use_fallback = use_fallback
        self.use_cache = use_cache

        genai.configure(api_key=api_key)
        self.model_name = "models/gemini-3.1-flash-lite" 
        
        if self.use_reranker:
            self.reranker = CrossEncoder('BAAI/bge-reranker-base')
            print("✅ Modelul de Re-Ranking a fost încărcat cu succes!")
        else:
            self.reranker = None
            print("⚠️ Re-Ranking dezactivat din configurație.")
        
        try:
            self.redis_client = redis.Redis(host='redis', port=6379, db=0, decode_responses=True)
            self.redis_client.ping()
            print("🟢 Conexiune la Redis Cache stabilită cu succes.")
        except redis.ConnectionError:
            self.redis_client = None
            print("🟡 Redis nu este disponibil. Cache-ul va fi ignorat.")

    def _cosine_similarity(self, v1, v2):
        """Calculează distanța semantică între 2 vectori."""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_v1 = math.sqrt(sum(a * a for a in v1))
        norm_v2 = math.sqrt(sum(b * b for b in v2))
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return dot_product / (norm_v1 * norm_v2)

    def _get_context_semantic(self, question, user_id, active_reranker):
        """Căutare care suportă comutarea între Classic RAG și Reranked RAG."""
        if active_reranker and self.reranker:
            print("🔍 Mod [RE-RANKING ACTIVAT]: Se extrag Top 10 documente...")
            docs_brute = self.vector_db.similarity_search(question, k=10, filter={"user_id": user_id})
            if not docs_brute: return []

            perechi = [[question, doc.page_content] for doc in docs_brute]
            scoruri = self.reranker.predict(perechi)
            for doc, scor in zip(docs_brute, scoruri):
                doc.metadata['rerank_score'] = float(scor)
                
            return sorted(docs_brute, key=lambda x: x.metadata['rerank_score'], reverse=True)[:3]
        else:
            print("🔍 Mod [CLASIC]: Se extrag Top 3 documente direct din ChromaDB...")
            return self.vector_db.similarity_search(question, k=3, filter={"user_id": user_id})
        
    def _get_context_by_keywords(self, question, user_id):
        """Căutare lexicală de urgență, limitată strict la documentele utilizatorului."""
        print("🔍 Agentul Critic rulează căutarea lexicală bazată pe cuvinte-cheie...")

        keywords = re.findall(r'"([^"]*)"', question)
        if not keywords:
            keywords = [word for word in question.split() if len(word) > 4 and word.lower() not in ["este", "pentru", "atunci", "ce"]]

        toate_datele = self.vector_db.get(where={"user_id": user_id})
        doc_potrivite = []

        for doc_text, meta in zip(toate_datele['documents'], toate_datele['metadatas']):
            for kw in keywords:
                if kw.lower() in doc_text.lower():
                    doc_potrivite.append(Document(page_content=doc_text, metadata=meta))
                    break 
            if len(doc_potrivite) >= 3: break

        return doc_potrivite

    def _generate_with_gemini(self, prompt):
        """Metodă avansată rezistentă la erorile de tip 503 și 429."""
        max_network_attempts = 3
        for retea_attempt in range(max_network_attempts):
            try:
                model = genai.GenerativeModel(self.model_name)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=1000
                    )
                )
                return response.text.strip()
            except Exception as e:
                err_msg = str(e)
                print(f"🚨 Eroare API detectată: {err_msg}")
                if "429" in err_msg or "Quota exceeded" in err_msg:
                    timp_asteptare = 7.5
                    match = re.search(r"(?:retry in\s+)([0-9.]+)(?:s)?", err_msg, re.IGNORECASE)
                    if match:
                        try: timp_asteptare = float(match.group(1)) + 0.5
                        except ValueError: pass
                    time.sleep(timp_asteptare)
                elif "503" in err_msg or "Service Unavailable" in err_msg:
                    time.sleep(2.0)
                else:
                    break
        return ""

    def _is_valid(self, response, require_source=True):
        if "LIPSA_CONTEXT" in response: return False
            
        is_long_enough = len(response.split()) > 10
        negatii = ["nu se menționează", "nu am găsit", "nu există", "nu conține", "nu oferă"]
        este_negatie = any(negatie in response.lower() for negatie in negatii)

        if require_source:
            has_source = "Conform cursului" in response or "(Sursa:" in response
            return has_source and is_long_enough and not este_negatie
        else:
            return is_long_enough and not este_negatie

    def _substituie_etichete_context(self, text, mapare_documente):
        """Înlocuiește tagurile neutre [DOC_X] cu 'Nume_Fisier (Pagina Y)' specifice utilizatorului curent."""
        rezultat = text
        for eticheta, (sursa_reala, pagina_reala) in mapare_documente.items():
            # Doar numele și pagina, pentru a nu dubla "Conform cursului" din prompt
            referinta_completa = f"{sursa_reala} (Pagina {pagina_reala})"
            rezultat = rezultat.replace(eticheta, referinta_completa)
    
        # Fallback gramatical coerent în caz că LLM-ul a inventat un tag neexistent:
        rezultat = re.sub(r'\[DOC_[a-zA-Z0-9_]+\]', 'materialelor suport', rezultat)
        return rezultat

    def ask(self, question, user_id, use_reranker=None, use_context_framing=None, use_fallback=None, use_cache=None):
        timp_start = time.time()

        active_reranker = use_reranker if use_reranker is not None else self.use_reranker
        active_framing = use_context_framing if use_context_framing is not None else self.use_context_framing
        active_fallback = use_fallback if use_fallback is not None else self.use_fallback
        active_cache = use_cache if use_cache is not None else self.use_cache

        docs = self._get_context_semantic(question, user_id, active_reranker)
        
        # ==========================================
        # 1. GENERARE MAPARE DINAMICĂ (source_file, page_number)
        # ==========================================
        mapare_documente = {}
        for index, d in enumerate(docs):
            eticheta_doc = f"[DOC_{index+1}]"
            sursa = d.metadata.get('source_file', 'Curs_Necunoscut.pdf')
            pagina = d.metadata.get('page_number', 'N/A')
            mapare_documente[eticheta_doc] = (sursa, pagina)

        # ==========================================
        # 2. LOGICĂ SEMANTIC CACHE (Cosinus)
        # ==========================================
        hash_uri_fragmente = sorted([d.metadata.get('text_hash', '') for d in docs])
        semnatura_context = hashlib.sha256("".join(hash_uri_fragmente).encode('utf-8')).hexdigest()
        stare_config = f"R={active_reranker}_F={active_framing}_FB={active_fallback}"
        
        semcache_key = f"semcache:{stare_config}:{semnatura_context}"
        q_emb = []

        if active_cache and self.redis_client:
            try:
                if hasattr(self.vector_db, 'embeddings'):
                    q_emb = self.vector_db.embeddings.embed_query(question)
                else:
                    q_emb = self.vector_db._embedding_function.embed_query(question)
                
                cached_entries = self.redis_client.hgetall(semcache_key)
                best_sim = 0.0
                best_match = None
                best_q_text = ""
                
                for q_text, data_str in cached_entries.items():
                    data = json.loads(data_str)
                    sim = self._cosine_similarity(q_emb, data["embedding"])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = data
                        best_q_text = q_text

                if best_match and best_sim >= 0.92:
                    print(f"⚡ [SEMANTIC CACHE HIT] Sim={best_sim:.3f} | Sursă: '{best_q_text}'")
                    timp_executie = round(time.time() - timp_start, 4)
                    
                    if best_match["answer"] == "FLAG_LIPSA_CONTEXT":
                        return {"answer": "Îmi pare rău, dar informația solicitată nu se găsește în documentele furnizate.", "status": "lipsa_context_cache", "sources": [d.metadata for d in docs], "latency": timp_executie}
                    else:
                        # Se aplică de-anonimizarea dinamică a numelui și paginii pentru utilizatorul curent
                        raspuns_rezolvat = self._substituie_etichete_context(best_match["answer"], mapare_documente) if active_framing else best_match["answer"]
                        return {"answer": raspuns_rezolvat, "status": "success_cache", "sources": [d.metadata for d in docs], "latency": timp_executie}
                        
            except Exception as e:
                print(f"⚠️ Eroare la accesarea Semantic Cache: {e}")

        # ==========================================
        # 3. GENERARE RĂSPUNS CU MODELUL LLM
        # ==========================================
        attempt = 0
        final_response_raw = ""
        foloseste_keyword_search = False

        while attempt < self.max_retries:
            if active_framing:
                context_elemente = []
                for index, d in enumerate(docs):
                    eticheta_doc = f"[DOC_{index+1}]"
                    context_elemente.append(f"Conform cursului {eticheta_doc}, {d.page_content}")
                context_complet = "\n\n".join(context_elemente)

                prompt = f"""### Instruction: You are an academic expert. Answer the question in Romanian using ONLY the provided context.
                If the context does not contain the answer, reply EXACTLY and ONLY with the word: 'LIPSA_CONTEXT'.
                Otherwise, your response MUST start exactly with the phrase: 'Conform cursului [DOC_1] ...' (Use the appropriate document tag from the context).

                ### Context:
                {context_complet}

                ### Question: {question}
                ### Answer:"""
            else:
                context_complet = "\n\n".join([d.page_content for d in docs])
                prompt = f"""### Instruction: You are an academic expert. Answer the question in Romanian using ONLY the provided context.
                If the context does not contain the answer, reply EXACTLY and ONLY with the word: 'LIPSA_CONTEXT'.

                ### Context:
                {context_complet}

                ### Question: {question}
                ### Answer:"""

            print(f"🔄 Generare răspuns (Încercarea {attempt + 1}). Fallback activ: {foloseste_keyword_search}")
            final_response_raw = self._generate_with_gemini(prompt)

            if self._is_valid(final_response_raw, require_source=active_framing):
                print("✅ Răspuns VALIDAT semantic!")
                
                # Salvăm în cache răspunsul brut ce conține [DOC_X] (complet agnostic de nume și pagină)
                if active_cache and self.redis_client and q_emb:
                    cache_data = json.dumps({"embedding": q_emb, "answer": final_response_raw, "sources": [d.metadata for d in docs]})
                    self.redis_client.hset(semcache_key, question, cache_data)
                    self.redis_client.expire(semcache_key, 86400)
                
                # De-anonimizăm răspunsul pentru afișarea către client
                raspuns_final = self._substituie_etichete_context(final_response_raw, mapare_documente) if active_framing else final_response_raw

                return {
                    "answer": raspuns_final,
                    "status": "success_keyword_search" if foloseste_keyword_search else "success",
                    "sources": [d.metadata for d in docs],
                    "latency": round(time.time() - timp_start, 4) 
                }

            if not foloseste_keyword_search:
                if active_fallback:
                    print("🚨 Eșec validare. Se activează fallback-ul pe Cuvinte-Cheie...")
                    docs = self._get_context_by_keywords(question, user_id)
                    # Recalculăm maparea pe noul set de documente din fallback
                    mapare_documente = {f"[DOC_{idx+1}]": (d.metadata.get('source_file', 'Curs_Necunoscut.pdf'), d.metadata.get('page_number', 'N/A')) for idx, d in enumerate(docs)}
                    foloseste_keyword_search = True
                else:
                    break 
            else:
                prompt += " (CRITICAL REMINDER: State the facts directly from the text!)"
            attempt += 1

        # ==========================================
        # 4. FINALIZARE ȘI FALLBACK DETERMINIST
        # ==========================================
        timp_executie = round(time.time() - timp_start, 4)
        
        if "LIPSA_CONTEXT" in final_response_raw or attempt >= self.max_retries:
            if active_cache and self.redis_client and q_emb:
                cache_data = json.dumps({"embedding": q_emb, "answer": "FLAG_LIPSA_CONTEXT", "sources": []})
                self.redis_client.hset(semcache_key, question, cache_data)
                self.redis_client.expire(semcache_key, 86400)
                
            return {"answer": "Îmi pare rău, dar informația solicitată nu se găsește în documentele furnizate.", "status": "lipsa_context", "sources": [d.metadata for d in docs], "latency": timp_executie}

        print("🚨 Eșec repetat la formatare. Se aplică fallback-ul determinist final.")
        mesaj_fallback = "Sistemul a întâmpinat dificultăți în a formula un răspuns clar. Cu toate acestea, informații potențial relevante se află în următoarele documente:"

        if docs:
            surse_grupate = {}
            for d in docs:
                s_f = d.metadata.get('source_file', 'Curs_Necunoscut.pdf')
                p_n = str(d.metadata.get('page_number', 'N/A'))
                if s_f not in surse_grupate: surse_grupate[s_f] = set()
                surse_grupate[s_f].add(p_n)

            referinte = [f"📌 {curs} (Pagini: {', '.join(sorted(list(pag), key=lambda x: int(x) if x.isdigit() else x))})" for curs, pag in surse_grupate.items()]
            final_response_fallback = f"{mesaj_fallback}\n\n" + "\n".join(referinte)
        else:
            final_response_fallback = "Sistemul nu a putut extrage documente relevante pentru această interogare."

        if active_cache and self.redis_client and q_emb:
            cache_data = json.dumps({"embedding": q_emb, "answer": final_response_fallback, "sources": [d.metadata for d in docs]})
            self.redis_client.hset(semcache_key, question, cache_data)
            self.redis_client.expire(semcache_key, 86400)

        return {"answer": final_response_fallback, "status": "fallback_applied", "sources": [d.metadata for d in docs], "latency": timp_executie}


    def obtine_cursuri_incarcate(self, user_id):
        print(f"📂 Se extrage lista cursurilor pentru utilizatorul {user_id}...")
        try:
            date_db = self.vector_db.get(limit=50000, where={"user_id": user_id})
            metadate = date_db.get("metadatas", [])
            
            surse_unice = set()
            for meta in metadate:
                if not meta: continue
                if "source_file" in meta: surse_unice.add(meta["source_file"])
                elif "source" in meta: surse_unice.add(meta["source"].split("\\")[-1].split("/")[-1])
                    
            cursuri_lista = list(surse_unice)
            print(f"✅ Au fost găsite {len(cursuri_lista)} cursuri unice din {len(metadate)} fragmente analizate.")
            return cursuri_lista
        except Exception as e:
            print(f"❌ Eroare la citirea bazei de date ChromaDB: {e}")
            return []