import sys
import types
import asyncio
import nest_asyncio
import itertools
import os
import re
import pandas as pd
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from agent import TutorialRAGAgent
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig

nest_asyncio.apply()

# --- FIX PENTRU ERORI TYPEERROR CU VERTEXAI ---
dummy_module = types.ModuleType('langchain_community.chat_models.vertexai')
class DummyChatVertexAI: 
    pass
dummy_module.ChatVertexAI = DummyChatVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = dummy_module

# =======================================================
# Clasă de protecție împotriva bug-ului de temperatură
# =======================================================
class SafeGoogleLLM(ChatGoogleGenerativeAI):
    async def _agenerate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None) 
        return await super()._agenerate(messages, stop=stop, run_manager=run_manager, **kwargs)
        
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        kwargs.pop("temperature", None)
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)

# =======================================================
# SETUL DE DATE EXTINS (24 ÎNTREBĂRI CLASIFICATE)
# =======================================================
BENCHMARK_TEST_DATA = [
    # --- CATEGORIA 1: Vocabular mixt bilingv (Română-Engleză) ---
    {
        "question": "Ce deosebește arhitectura Microsoft GraphRAG de un sistem RAG clasic?",
        "ground_truth": "Spre deosebire de un RAG clasic care caută doar fragmente de text, GraphRAG folosește un LLM pentru a construi un Graf de Cunoștințe din familii de documente, navigând prin graful simbolic pentru a înțelege contextul global.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 14
    },
    {
        "question": "Cum previne platforma Palantir halucinațiile agenților săi AI (AIP)?",
        "ground_truth": "Agenții AI din Palantir nu au voie să ghicească date, ci sunt obligați să interogheze componenta simbolică (Ontologia) pentru a obține fapte, folosind-o ca pe o ancoră de adevăr (Grounding).",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 13
    },
    {
        "question": "De ce factori depinde scalarea cognitivă a unui model, pe lângă dimensiunea contextului?",
        "ground_truth": "Scalarea cognitivă depinde de parametri și arhitectură (numărul de straturi, dimensiunea embeddingurilor), de calitatea atenției (mecanisme precum sparse attention, linear attention), de memoria externă (precum RAG, CAG sau baze vectoriale) și de fine-tuning-ul realizat pe sarcini complexe.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 8
    },
    {
        "question": "Cum facilitează sistemul Stardog (Voicebox) interacțiunea utilizatorilor cu bazele de date tip Knowledge Graph?",
        "ground_truth": "Stardog permite utilizatorilor să vorbească direct cu datele, utilizând agenți AI care generează automat interogări în limbajul SPARQL pentru a extrage răspunsuri logice perfecte.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 15
    },
    {
        "question": "Ce rol are protocolul Model Context Protocol (MCP) în arhitectura Azure AI Foundry?",
        "ground_truth": "MCP este un protocol open-source care stabilește o arhitectură client-server standardizată, permițând aplicațiilor AI și modelelor lingvistice să se conecteze în siguranță la diverse surse de date, baze de date și unelte externe.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_08-Azure AI Foundry.pdf",
        "page_number": 15
    },
    {
        "question": "Cum se realizează izolarea și persistența contextului conversațional în Playground-ul Azure AI Foundry?",
        "ground_truth": "Playground-ul folosește un Thread ID unic pentru fiecare sesiune de lucru, asigurând continuitatea pe termen lung și izolarea contextului conversațional între interacțiunile utilizatorilor.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_08-Azure AI Foundry.pdf",
        "page_number": 17
    },
    {
        "question": "Ce rol joacă tehnologia RDMA în clusterul de antrenare distribuită pentru modele LLM?",
        "ground_truth": "RDMA (Remote Direct Memory Access) permite accesul direct la memoria altui nod GPU fără implicarea procesorului (CPU) sau a sistemului de operare gazdă, asigurând o latență minimă și un transfer ultra-rapid de date.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_01-Teorema CAP și Arhitectura Cloud Computing.pdf",
        "page_number": 23
    },
    {
        "question": "Ce este abstractizarea FSSpec și ce driver este folosit pentru Azure Data Lake Gen2?",
        "ground_truth": "FSSpec (Filesystem Spec) este o interfață unificată care standardizează operațiile pe fișiere (open, ls, mkdir, rm) indiferent de stocare, utilizând driverul adlfs pentru backend-ul Azure ADLS Gen2.",
        "category": "Vocabular mixt bilingv",
        "source_file": "Cursul_06-Distributed Storage Architectures.pdf",
        "page_number": 8
    },

    # --- CATEGORIA 2: Setul de diacritice specifice limbii române ---
    {
        "question": "Ce rol joacă ontologia în raport cu datele brute?",
        "ground_truth": "Ontologia reprezintă structura de bază a cunoașterii. Dacă datele brute sunt 'cărămizile', ontologia este 'planul arhitectural'.",
        "category": "Setul de diacritice",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 2
    },
    {
        "question": "Care sunt cele două componente majore ale unui sistem neuro-simbolic modern și ce aduce fiecare?",
        "ground_truth": "Sistemul este format din LLM (Motorul Neuro), care aduce intuiția lingvistică și procesează limbajul natural ambiguu, și Ontologia (Motorul Simbolic), care aduce regulile, ierarhiile clare și adevărurile verificate.",
        "category": "Setul de diacritice",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 3
    },
    {
        "question": "Cum explică arhitectura neuro-simbolică trecerea de la corelație la cauzalitate în inteligența artificială?",
        "ground_truth": "Deep Learning observă doar corelația statistică, în timp ce arhitecturile neuro-simbolice utilizează structuri simbolice (ontologii) pentru a codifica reguli cauzale explicite despre fenomene.",
        "category": "Setul de diacritice",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 5
    },
    {
        "question": "Care sunt cele trei proprietăți fundamentale definite în cadrul Teoremei CAP?",
        "ground_truth": "Proprietățile sunt: Consistența (toate nodurile văd aceleași date simultan), Disponibilitatea (fiecare cerere primește un răspuns) și Toleranța la partiționare (sistemul funcționează chiar dacă se pierd mesaje în rețea).",
        "category": "Setul de diacritice",
        "source_file": "Cursul_01-Teorema CAP și Arhitectura Cloud Computing.pdf",
        "page_number": 3
    },
    {
        "question": "De ce este considerată partiționarea o axiomă inevitabilă în arhitecturile cloud computing?",
        "ground_truth": "Partiționarea este inevitabilă din cauza instabilității rețelei, distribuției geografice pe multiple zone de disponibilitate, deconectărilor temporare și volumului masiv de cereri concurente.",
        "category": "Setul de diacritice",
        "source_file": "Cursul_01-Teorema CAP și Arhitectura Cloud Computing.pdf",
        "page_number": 4
    },
    {
        "question": "Ce reprezintă noțiunea de redundanță logică în contextul stocării tranzacționale Delta Lake?",
        "ground_truth": "Redundanța logică constă în menținerea unui jurnal de tranzacții (_delta_log/) care conține versiuni succesive de metadate, permițând refacerea stării tabelei la orice punct din trecut prin mecanismul de time-travel.",
        "category": "Setul de diacritice",
        "source_file": "Cursul_06-Distributed Storage Architectures.pdf",
        "page_number": 16
    },
    {
        "question": "Cum funcționează agregarea prin bootstrap (Bagging) în cadrul algoritmilor de clasificare?",
        "ground_truth": "Bagging extrage submulțimi aleatoare din datele de antrenare, construiește câte un model independent pentru fiecare submulțime și aplică o schemă de votare majoritară pentru a decide clasa finală.",
        "category": "Setul de diacritice",
        "source_file": "Curs_10_ML_Theory.pdf",
        "page_number": 5
    },
    {
        "question": "Care este diferența de bază dintre regularizarea Ridge și regularizarea Lasso în regresia liniară?",
        "ground_truth": "Regresia Ridge penalizează coeficienții mari folosind norma euclidiană (L2), menținând toți termenii, în timp ce regresia Lasso folosește norma Manhattan (L1), forțând anumiți coeficienți să devină exact zero pentru selecția automată a variabilelor.",
        "category": "Setul de diacritice",
        "source_file": "Curs_10_ML_Theory.pdf",
        "page_number": 38
    },

    # --- CATEGORIA 3: Informații structurate în tabele și liste-uri ---
    {
        "question": "Care este limitarea fundamentală a modelelor Transformer în ceea ce privește memoria și procesarea contextului?",
        "ground_truth": "Transformerele nu au memorie nelimitată și procesează intrările printr-un mecanism de self-attention cu o complexitate de O(n^2) față de lungimea secvenței, costul computațional crescând exponențial.",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 7
    },
    {
        "question": "Cum se definește formal un sistem neuro-simbolic folosind funcții matematice?",
        "ground_truth": "Formal, sistemul neuro-simbolic poate fi descris ca funcția compusă F(x)=R(S(E(x))), unde E(x) este encoderul neuronal, S este stratul simbolic care mapează reprezentările în structuri logice, iar R reprezintă motorul de raționament.",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 11
    },
    {
        "question": "Ce tehnologie NeSy gestionează stratul cognitiv de percepție și care este rolul ei conform structurii NeSy?",
        "ground_truth": "Nivelul de percepție este gestionat de CNN și Transformers, având rolul de a transforma semnalul brut (imagine sau text) în concepte semantice.",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_11-Arhitecturi Neuro-Simbolice.pdf",
        "page_number": 6
    },
    {
        "question": "Care sunt diferențele cheie între formatele de tabel Apache Iceberg și Apache Hudi în privința scenariilor ideale?",
        "ground_truth": "Apache Iceberg este optimizat pentru analitice la scară largă în arhitecturi Lakehouse, BI și interogări temporale, în timp ce Apache Hudi este specializat pe ingestie rapidă în timp real, actualizări incrementale și Change Data Capture (CDC).",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_06-Distributed Storage Architectures.pdf",
        "page_number": 7
    },
    {
        "question": "Cum sunt clasificate serviciile Azure Cosmos DB și Azure SQL conform modelului PACELC?",
        "ground_truth": "Azure Cosmos DB este flexibil (configurabil PA/EL sau PC/EC cu 5 niveluri de consistență), în timp ce Azure SQL Database este clasificat strict ca PC/EC, garantând tranzacții ACID și replicare sincronă.",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_01-Teorema CAP și Arhitectura Cloud Computing.pdf",
        "page_number": 6
    },
    {
        "question": "Care sunt diferențele de toleranță la defecțiuni și overhead de mesaje între algoritmii de consens Paxos, Raft și BFT?",
        "ground_truth": "Paxos și Raft tolerează f < n/2 noduri căzute (crash faults) cu overhead moderat, în timp ce BFT tolerează f < n/3 noduri malițioase (Byzantine faults), dar introduce un overhead ridicat de mesaje de ordinul O(n^2) sau O(n^3).",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_01-Teorema CAP și Arhitectura Cloud Computing.pdf",
        "page_number": 9
    },
    {
        "question": "Care sunt cele 4 nivele din taxonomia platformelor de inteligență artificială și ce platformă reprezintă Nivelul 4?",
        "ground_truth": "Nivelele sunt: Nivelul 1 (Model Services - AWS Bedrock), Nivelele 2 și 3 (AI Application Platforms - Azure AI Foundry) și Nivelul 4 (AI Operating Systems), reprezentat de Palantir AIP cu ontologie nativă.",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_08-Azure AI Foundry.pdf",
        "page_number": 2
    },
    {
        "question": "Care sunt componentele principale de persistență în Azure Databricks și ce rol are clusterul Spark?",
        "ground_truth": "Componentele sunt ADLS Gen2 (stocare persistentă primară), DBFS (strat de abstracție peste stocarea cloud) și Spark Cluster, care funcționează exclusiv ca un nivel computațional temporar ce nu persistă date pe termen lung.",
        "category": "Informații structurate în tabele și slide-uri",
        "source_file": "Cursul_06-Distributed Storage Architectures.pdf",
        "page_number": 14
    }
]

async def async_run_ragas_evaluation(agent, vector_db, api_key, user_id, config_name, use_reranker, use_context_framing, use_fallback, test_dataset):
    print(f"\n{'='*60}")
    print(f"🚀 SE RULEAZĂ SCENARIUL: {config_name}")
    print(f"Setări: Reranker={use_reranker} | Framing={use_context_framing} | Fallback={use_fallback}")
    print(f"{'='*60}\n")
    
    base_evaluator_llm = SafeGoogleLLM(
        model="models/gemini-3.5-flash-lite", 
        google_api_key=api_key,
        temperature=0.0
    )
    evaluator_llm = LangchainLLMWrapper(base_evaluator_llm)
    
    base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    rezultate_pentru_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "category": [],
        "expected_source": [],
        "expected_page": [],
        "retrieved_sources": [],
        "latency_seconds": [],
        "agent_status": []
    }
    
    for item in test_dataset:
        q = item["question"]
        
        raspuns_sistem = agent.ask(
            question=q, 
            user_id=user_id, 
            use_reranker=use_reranker,
            use_context_framing=use_context_framing,
            use_fallback=use_fallback,
            use_cache=False
        )
        
        texte_context = []
        surse_detectate = []
        for sursa in raspuns_sistem.get('sources', []):
            if isinstance(sursa, dict):
                surse_detectate.append(f"{sursa.get('source_file', 'N/A')} (p.{sursa.get('page_number', 'N/A')})")
                if 'text_hash' in sursa:
                    try:
                        db_rez = vector_db.get(where={"text_hash": sursa['text_hash']})
                        if db_rez and db_rez.get('documents'):
                            texte_context.append(db_rez['documents'][0])
                        else:
                            texte_context.append(str(sursa))
                    except Exception:
                        texte_context.append(str(sursa))
            elif hasattr(sursa, 'page_content'):
                texte_context.append(sursa.page_content)
                surse_detectate.append(f"{sursa.metadata.get('source_file', 'N/A')} (p.{sursa.metadata.get('page_number', 'N/A')})")
        
        rezultate_pentru_ragas["question"].append(q)
        rezultate_pentru_ragas["answer"].append(raspuns_sistem["answer"])
        rezultate_pentru_ragas["contexts"].append(texte_context)
        rezultate_pentru_ragas["ground_truth"].append(item["ground_truth"])
        rezultate_pentru_ragas["category"].append(item.get("category", "General"))
        rezultate_pentru_ragas["expected_source"].append(item.get("source_file", "N/A"))
        rezultate_pentru_ragas["expected_page"].append(item.get("page_number", "N/A"))
        rezultate_pentru_ragas["retrieved_sources"].append("; ".join(surse_detectate))
        rezultate_pentru_ragas["latency_seconds"].append(raspuns_sistem.get("latency", 0))
        rezultate_pentru_ragas["agent_status"].append(raspuns_sistem.get("status", "unknown"))

    dataset_evaluare = Dataset.from_dict({
        "question": rezultate_pentru_ragas["question"],
        "answer": rezultate_pentru_ragas["answer"],
        "contexts": rezultate_pentru_ragas["contexts"],
        "ground_truth": rezultate_pentru_ragas["ground_truth"]
    })

    configuratie_rulare = RunConfig(max_workers=1, max_retries=5, max_wait=30)
    
    rezultat_final = evaluate(
        dataset=dataset_evaluare,
        metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision()],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        run_config=configuratie_rulare, 
        raise_exceptions=False          
    )

    df_rezultate = rezultat_final.to_pandas()
    df_rezultate["category"] = rezultate_pentru_ragas["category"]
    df_rezultate["expected_source"] = rezultate_pentru_ragas["expected_source"]
    df_rezultate["expected_page"] = rezultate_pentru_ragas["expected_page"]
    df_rezultate["retrieved_sources"] = rezultate_pentru_ragas["retrieved_sources"]
    df_rezultate["latency_seconds"] = rezultate_pentru_ragas["latency_seconds"]
    df_rezultate["agent_status"] = rezultate_pentru_ragas["agent_status"]
    
    nume_fisier = f"rezultate_eval_{config_name}.csv"
    df_rezultate.to_csv(nume_fisier, index=False)
    
    print(f"\n✅ Evaluare completă pentru {config_name}! Medii obținute:")
    for metrica in ['faithfulness', 'answer_relevancy', 'context_precision']:
        if metrica in df_rezultate.columns:
            print(f"   - {metrica.replace('_', ' ').title()}: {df_rezultate[metrica].mean():.4f}")
            
    print(f"📂 Salvat în: {nume_fisier}")
    return df_rezultate

def run_all_ablation_tests(agent, vector_db, api_key, user_id, test_dataset):
    # Generare cele 8 combinatii posibile (Re-ranker, Framing, Fallback)
    optiuni = [True, False]
    combinatii = list(itertools.product(optiuni, repeat=3))
    
    for reranker, framing, fallback in combinatii:
        nume_config = f"R={reranker}_F={framing}_FB={fallback}"
        asyncio.run(async_run_ragas_evaluation(
            agent=agent, 
            vector_db=vector_db, 
            api_key=api_key, 
            user_id=user_id, 
            config_name=nume_config,
            use_reranker=reranker,
            use_context_framing=framing,
            use_fallback=fallback,
            test_dataset=test_dataset
        ))

def run_all_ablation_tests_part_1(agent, vector_db, api_key, user_id, test_dataset):
    # =========================================================================
    # CELE MAI RELEVANTE 4 CONFIGURAȚII (Pentru economisirea cotei API)
    # =========================================================================
    selected_configs = [
        # 1. Baseline: Naive RAG (fără module de optimizare)
        # (False, False, False, "Baseline_Naive_RAG"),
        
        # 2. Doar Re-Ranking: Maximul de performanță pe precizie și fidelitate
        # (True, False, False, "Doar_ReRanking"),
        
        # 3. Re-Ranking + Framing: Citare didactică și decuplare cache
        # (True, True, False, "ReRanking_Framing"),
        
        # 4. Sistem Complet: Propunerea arhitecturală finală a lucrării
        # (True, True, True, "Sistem_Complet_Full"),
        
        # --- CONFIGURAȚII SECUNDARE (PĂSTRATE COMENTATE) ---
        (True,  False, True,  "ReRanking_Fallback"),
        (False, False, True,  "Doar_Fallback")
        # (False, True,  False, "Doar_Framing"),
        # (False, True,  True,  "Framing_Fallback"),
        
    ]
    
    for reranker, framing, fallback, config_alias in selected_configs:
        nume_config = f"R={reranker}_F={framing}_FB={fallback}_{config_alias}"
        
        asyncio.run(async_run_ragas_evaluation(
            agent=agent, 
            vector_db=vector_db, 
            api_key=api_key, 
            user_id=user_id, 
            config_name=nume_config,
            use_reranker=reranker,
            use_context_framing=framing,
            use_fallback=fallback,
            test_dataset=test_dataset
        ))

if __name__ == "__main__":
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    USER_DE_TEST = "Student_A"
    PATH_DB = "../db_cursuri" 
    
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=PATH_DB, embedding_function=embedding_model)
    agent = TutorialRAGAgent(vector_db=vector_db, api_key=GEMINI_API_KEY)
    
    # Rularea studiului complet de ablațiune
    run_all_ablation_tests_part_1(agent, vector_db, GEMINI_API_KEY, USER_DE_TEST, BENCHMARK_TEST_DATA)


# docker compose build evaluator
# docker compose up evaluator