import os
import shutil
import re
import hashlib
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from agent import TutorialRAGAgent

app = FastAPI(title="Distributed Academic Tutorial API", version="1.0")

# --- CONFIGURĂRI INIȚIALE ---
# Calea ta stabilă și existentă către baza de date
PATH_DB = "../db_cursuri" 
FOLDER_TEMPORAR = "../incarcari_temporare"
os.makedirs(FOLDER_TEMPORAR, exist_ok=True)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vector_db = Chroma(persist_directory=PATH_DB, embedding_function=embedding_model)

# Instanțiem agentul 
agent = TutorialRAGAgent(vector_db=vector_db, api_key=GEMINI_API_KEY)


# --- MODELE DE DATE (Pydantic) ---
class QuestionRequest(BaseModel):
    question: str
    user_id: str

class QuestionResponse(BaseModel):
    answer: str
    status: str
    sources: list


# --- FUNCTIE DE CURĂȚARE TEXT ---
def clean_text(text):
    """Curăță textul brut din PDF pentru a optimiza calitatea vectorizării."""
    if not text:
        return ""
    text = " ".join(text.split())
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.IGNORECASE)
    return text.strip()

def genereaza_hash_text(text: str) -> str:
    """Generează un hash determinist exclusiv pe baza conținutului textului."""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# --- LOGICĂ INTERNĂ DE INGESTIE ---
def proceseaza_si_indexeaza_pdf(file_path: str, file_name: str, user_id: str):
    """Funcție asincronă care sparge PDF-ul, asignează hash-uri și indexează în ChromaDB."""
    try:
        print(f"⚡ Începe procesarea documentului: {file_name} pentru utilizatorul: {user_id}")
        
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        
        # Injectarea metadatelor
        for page in pages:
            nr_pagina = str(page.metadata.get("page", 0) + 1)
            page.page_content = clean_text(page.page_content)
            # Suprascriem dicționarul de bază
            page.metadata = {
                "source_file": str(file_name),
                "page_number": nr_pagina,
                "user_id": user_id
            }

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100,
            length_function=len,
            add_start_index=True
        )
        chunks = text_splitter.split_documents(pages)
        print(f"📦 S-au generat {len(chunks)} fragmente din {file_name}")

        # Calculăm hash-ul fiecărui chunk și îl injectăm în metadate
        for chunk in chunks:
            chunk.metadata["text_hash"] = genereaza_hash_text(chunk.page_content)

        # Extragem textele și metadatele
        texts = [doc.page_content for doc in chunks]
        metadatas = [doc.metadata for doc in chunks]

        if not os.path.exists(PATH_DB) or len(os.listdir(PATH_DB)) == 0:
            print("🚨 Eroare: Baza de date nu a fost găsită în calea specificată!")
        else:
            print("➕ Se adaugă noile fragmente în baza de date existentă...")
            vector_db.add_texts(texts=texts, metadatas=metadatas)
            
        print(f"✅ Documentul {file_name} a fost adăugat cu succes în ChromaDB!")
        
    except Exception as e:
        print(f"🚨 Eroare critică la indexarea documentului {file_name}: {e}")
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)


@app.get("/")
def home():
    return {"status": "online", "message": "Distributed Academic Tutorial System API is active."}


# ==========================================
# RUTA 1: POST /upload-course
# ==========================================
@app.post("/upload-course", status_code=202)
def upload_course(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...),
    user_id: str = Form(...) 
):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Doar fișierele în format PDF sunt acceptate.")
    
    temp_file_path = os.path.join(FOLDER_TEMPORAR, file.filename)
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # NOU: Transmitem user_id către funcția de background
    background_tasks.add_task(proceseaza_si_indexeaza_pdf, temp_file_path, file.filename, user_id)
    
    return {
        "message": f"Fișierul '{file.filename}' a fost încărcat pentru utilizatorul '{user_id}'.",
        "detail": "Procesul de fragmentare semantică și indexare vectorială rulează în fundal."
    }


# ==========================================
# RUTA 2: POST /ask-question
# ==========================================
@app.post("/ask-question", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Întrebarea nu poate fi goală.")
    
    if not request.user_id.strip():
         raise HTTPException(status_code=400, detail="ID-ul utilizatorului este obligatoriu pentru căutare.")
        
    # Pasăm user_id către agent pentru filtrare și caching
    rezultat = agent.ask(request.question, request.user_id)
    
    if not rezultat.get("answer") or not rezultat["answer"].strip():
        raise HTTPException(
            status_code=500, 
            detail="Eroare internă: LLM-ul a returnat un răspuns gol sau sistemul a eșuat."
        )
    
    return QuestionResponse(
        answer=rezultat["answer"],
        status=rezultat["status"],
        sources=rezultat["sources"]
    )


# ==========================================
# RUTA 3: GET /cursuri_incarcate
# ==========================================
@app.get("/cursuri_incarcate")
async def get_cursuri_incarcate(user_id: str):
    if not user_id.strip():
        raise HTTPException(status_code=400, detail="Parametrul user_id este obligatoriu.")
        
    try:
        # Filtrăm lista de cursuri direct din ChromaDB
        cursuri = agent.obtine_cursuri_incarcate(user_id)
        return {"status": "success", "cursuri": cursuri}
    except Exception as e:
        return {"status": "error", "message": str(e)}