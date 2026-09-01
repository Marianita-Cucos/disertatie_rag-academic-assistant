import random
from locust import HttpUser, task, between, events

class StudentUser(HttpUser):
    # Timp realist de gândire/lectură între 1 și 3 secunde
    wait_time = between(1, 3)

    # Grupuri de întrebări semantice:
    # Fiecare grup conține întrebarea de bază și parafrazări care trebuie să declanșeze Semantic Cache (prag cosinus >= 0.92)
    grupuri_semantice = [
        # Grupul 1: Ontologie vs Date brute
        [
            "Ce rol joacă ontologia în raport cu datele brute?",
            "Cum se raportează ontologia la datele brute?",
            "Care este rolul ontologiei în raport cu datele brute?",
            "Ce reprezintă ontologia comparativ cu datele brute?"
        ],
        # Grupul 2: Componente Neuro-Simbolice
        [
            "Care sunt cele două componente majore ale unui sistem neuro-simbolic modern?",
            "Care sunt componentele principale dintr-un sistem neuro-simbolic modern?",
            "Ce elemente majore alcătuiesc un sistem neuro-simbolic modern?",
            "Din ce componente majore este format un sistem neuro-simbolic modern?"
        ],
        # Grupul 3: Limitare Transformer
        [
            "Care este limitarea fundamentală a modelelor Transformer în ceea ce privește memoria?",
            "Ce limitare de memorie au modelele Transformer la nivel fundamental?",
            "Care sunt limitările majore ale Transformerelor privind memoria și contextul?"
        ],
        # Grupul 4: GraphRAG vs RAG
        [
            "Ce deosebește arhitectura Microsoft GraphRAG de un sistem RAG clasic?",
            "Prin ce diferă Microsoft GraphRAG față de un RAG convențional?",
            "Care este diferența dintre Microsoft GraphRAG și sistemul RAG clasic?"
        ],
        # Grupul 5: Palantir AIP
        [
            "Cum previne platforma Palantir halucinațiile agenților săi AI (AIP)?",
            "În ce mod evită platforma Palantir halucinațiile la agenții AIP?",
            "Cum reușește Palantir AIP să oprească halucinațiile generate de AI?"
        ]
    ]


    @task
    def interogheaza_asistent_semantic(self):
        # 1. Alegem un grup tematic, apoi o parafrazare din cadrul acelui grup
        grup_ales = random.choice(self.grupuri_semantice)
        intrebare_aleasa = random.choice(grup_ales)

        payload = {
            "question": intrebare_aleasa,
            "user_id": "Student_A"
        }

        # Trimiterea cererii POST către backend
        with self.client.post("/ask-question", json=payload, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                status_agent = data.get("status", "unknown")

                # Verificăm statusurile valide
                if status_agent in ["success", "success_cache", "success_keyword_search", "lipsa_context", "lipsa_context_cache"]:
                    # Diferențiem metricile în raportul Locust dacă a fost lovitură în Cache sau Generare LLM
                    if "cache" in status_agent:
                        response.success()
                    else:
                        response.success()
                elif status_agent == "fallback_applied":
                    response.failure("Eroare de procesare: S-a aplicat fallback determinist de urgență.")
                else:
                    response.failure(f"Răspuns necunoscut returnat: {status_agent}")
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")


# locust -f locustfile.py
# http://localhost:8000