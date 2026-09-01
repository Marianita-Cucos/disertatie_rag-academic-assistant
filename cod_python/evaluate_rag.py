import sys
import types
import asyncio
import nest_asyncio
import itertools
import os

# 1. Permitem buclelor asincrone să ruleze una în interiorul celeilalte
nest_asyncio.apply()

# 2. FIX PENTRU EROAREA "TypeError" cu vertexai
dummy_module = types.ModuleType('langchain_community.chat_models.vertexai')
class DummyChatVertexAI: 
    pass
dummy_module.ChatVertexAI = DummyChatVertexAI
sys.modules['langchain_community.chat_models.vertexai'] = dummy_module

import os
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


async def async_run_ragas_evaluation(agent, vector_db, api_key, user_id, config_name, use_reranker, use_context_framing, use_fallback, use_cache):
    print(f"\n{'='*60}")
    print(f"🚀 SE RULEAZĂ SCENARIUL: {config_name}")
    print(f"Setări: Reranker={use_reranker} | Framing={use_context_framing} | Fallback={use_fallback} | Cache={use_cache}")
    print(f"{'='*60}\n")
    
    base_evaluator_llm = SafeGoogleLLM(
        model="models/gemini-3.5-flash-lite", 
        google_api_key=api_key,
        temperature=0.0
    )
    evaluator_llm = LangchainLLMWrapper(base_evaluator_llm)
    
    base_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    evaluator_embeddings = LangchainEmbeddingsWrapper(base_embeddings)

    test_data = [
        {
            "question": "Ce rol joacă ontologia în raport cu datele brute?",
            "ground_truth": "Ontologia reprezintă structura de bază a cunoașterii. Dacă datele brute sunt 'cărămizile', ontologia este 'planul arhitectural'."
        },
        {
            "question": "Care sunt cele două componente majore ale unui sistem neuro-simbolic modern și ce aduce fiecare?",
            "ground_truth": "Sistemul este format din LLM (Motorul Neuro), care aduce intuiția lingvistică și procesează limbajul natural ambiguu, și Ontologia (Motorul Simbolic), care aduce regulile, ierarhiile clare și adevărurile verificate."
        },
        {
            "question": "Care este limitarea fundamentală a modelelor Transformer în ceea ce privește memoria și procesarea contextului?",
            "ground_truth": "Transformerele nu au memorie nelimitată și procesează intrările printr-un mecanism de self-attention cu o complexitate de O(n^2) față de lungimea secvenței, costul computațional crescând exponențial."
        },
        {
            "question": "Cum se definește formal un sistem neuro-simbolic folosind funcții matematice?",
            "ground_truth": "Formal, sistemul neuro-simbolic poate fi descris ca funcția compusă F(x)=R(S(E(x))), unde E(x) este encoderul neuronal care transformă date brute în reprezentări distribuite, S este stratul simbolic care mapează reprezentările în structuri logice, iar R reprezintă motorul de raționament care operează pe structuri simbolice pentru inferență."
        },
        {
            "question": "Ce deosebește arhitectura Microsoft GraphRAG de un sistem RAG clasic?",
            "ground_truth": "Spre deosebire de un RAG clasic care caută doar fragmente de text, GraphRAG folosește un LLM pentru a construi un Graf de Cunoștințe din familii de documente. Sistemul nu caută doar cuvinte cheie, ci navighează prin graful simbolic pentru a înțelege contextul global."
        },
        {
            "question": "Cum previne platforma Palantir halucinațiile agenților săi AI (AIP)?",
            "ground_truth": "Agenții AI din Palantir nu au voie să ghicească date, ci sunt obligați să interogheze componenta simbolică (Ontologia) pentru a obține fapte, folosind-o ca pe o ancoră de adevăr."
        },
        {
            "question": "De ce factori depinde scalarea cognitivă a unui model, pe lângă dimensiunea contextului?",
            "ground_truth": "Scalarea cognitivă depinde de parametri și arhitectură (numărul de straturi, dimensiunea embeddingurilor), de calitatea atenției (mecanisme precum sparse attention, linear attention), de memoria externă (precum RAG, CAG sau baze vectoriale) și de fine-tuning-ul realizat pe sarcini complexe (nu doar memorare brută)."
        },
        {
            "question": "Componentele NeSy în IA Cognitiv, ce tehnologie NeSy gestionează stratul cognitiv de percepție și care este rolul ei?",
            "ground_truth": "Nivelul de percepție este gestionat de tehnologii precum CNN (Rețele Neuronale Convoluționale) și Transformers, având rolul de a transforma semnalul brut, cum ar fi o imagine sau un text, în concepte."
        },
        {
            "question": "Cum facilitează sistemul Stardog (Voicebox) interacțiunea utilizatorilor cu bazele de date tip Knowledge Graph?",
            "ground_truth": "Stardog permite utilizatorilor să vorbească direct cu datele, utilizând agenți AI care generează automat interogări în limbajul SPARQL pentru a extrage răspunsuri logice perfecte."
        }
    ]

    rezultate_pentru_ragas = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": [],
        "latency_seconds": [], # NOU: Salvăm latența pentru grafice
        "agent_status": []     # NOU: Salvăm statusul (fallback, succes etc.)
    }
    
    for item in test_data:
        q = item["question"]
        
        # Trimitem TOATE setările către agent
        raspuns_sistem = agent.ask(
            question=q, 
            user_id=user_id, 
            use_reranker=use_reranker,
            use_context_framing=use_context_framing,
            use_fallback=use_fallback,
            use_cache=use_cache
        )
        
        texte_context = []
        for sursa in raspuns_sistem.get('sources', []):
            if isinstance(sursa, dict) and 'text_hash' in sursa:
                try:
                    db_rezultat = vector_db.get(where={"text_hash": sursa['text_hash']})
                    if db_rezultat and db_rezultat.get('documents') and len(db_rezultat['documents']) > 0:
                        texte_context.append(db_rezultat['documents'][0])
                    else:
                        texte_context.append(str(sursa)) 
                except Exception:
                    texte_context.append(str(sursa))
            elif hasattr(sursa, 'page_content'): 
                texte_context.append(sursa.page_content)
            else:
                texte_context.append(str(sursa))
        
        rezultate_pentru_ragas["question"].append(q)
        rezultate_pentru_ragas["answer"].append(raspuns_sistem["answer"])
        rezultate_pentru_ragas["contexts"].append(texte_context)
        rezultate_pentru_ragas["ground_truth"].append(item["ground_truth"])
        
        # Salvăm noile date în dataset
        rezultate_pentru_ragas["latency_seconds"].append(raspuns_sistem.get("latency", 0))
        rezultate_pentru_ragas["agent_status"].append(raspuns_sistem.get("status", "unknown"))

    dataset_evaluare = Dataset.from_dict(rezultate_pentru_ragas)

    print(f"📊 Începe evaluarea cantitativă pentru {config_name}...")
    
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
    
    # Numele fișierului reflectă exact setările testului
    nume_fisier = f"rezultate_eval_{config_name}.csv"
    df_rezultate.to_csv(nume_fisier, index=False)
    
    print(f"\n✅ Evaluare completă pentru {config_name}! Medii obținute:")
    metrici_evaluate = ['faithfulness', 'answer_relevancy', 'context_precision']
    for metrica in metrici_evaluate:
        if metrica in df_rezultate.columns:
            scor_mediu = df_rezultate[metrica].mean()
            print(f"   - {metrica.replace('_', ' ').title()}: {scor_mediu:.4f}")
            
    print(f"📂 Salvat în: {nume_fisier}")
    return df_rezultate

def run_all_ablation_tests(agent, vector_db, api_key, user_id):
    # Generăm toate cele 8 combinații posibile pentru componentele de calitate
    # Ordine tuple: (Re-ranker, Framing, Fallback)
    optiuni = [True, False]
    combinatii = list(itertools.product(optiuni, repeat=3))
    
    for reranker, framing, fallback in combinatii:
        # Generăm un nume sugestiv pentru fișierul CSV
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
            use_cache=False # Cache-ul trebuie oprit în evaluare pentru a forța generarea de fiecare dată
        ))


def run_all_ablation_tests2(agent, vector_db, api_key, user_id):
    # Rulăm exclusiv combinațiile lipsă: (False, False, False) și (False, False, True)
    combinatii_ramase = [
        (False, False, False),
        (False, False, True)
    ]
    
    for reranker, framing, fallback in combinatii_ramase:
        # Numele fișierelor generate vor fi:
        # 1. rezultate_eval_R=False_F=False_FB=False.csv
        # 2. rezultate_eval_R=False_F=False_FB=True.csv
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
            use_cache=False
        ))

if __name__ == "__main__":

    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    USER_DE_TEST = "Student_A"
    
    PATH_DB = "../db_cursuri" 
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = Chroma(persist_directory=PATH_DB, embedding_function=embedding_model)
    agent = TutorialRAGAgent(vector_db=vector_db, api_key=GEMINI_API_KEY)
    
    # Rulăm scriptul master care execută toate cele 8 scenarii succesiv
    run_all_ablation_tests(agent, vector_db, GEMINI_API_KEY, USER_DE_TEST)



#  Reranker={use_reranker}              -->  T(alege 10, calc scor, top 3), F(direct top 3)
#  Framing={use_context_framing}        -->  T(trimite contextul catre gemini), F(intrebare libera catre gemini, fara context din cursuri)
#  Fallback={use_fallback}              -->  T(cauta dupa cuvinte cheie), F(nu cauta in plus dupa cuvinte cheie)
#  Cache={use_cache}                    --> pus pe false aici ca sa nu afecteze testele

# docker compose build evaluator
# docker compose up evaluator