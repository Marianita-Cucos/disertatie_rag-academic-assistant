# .\venv_disertatie\Scripts\activate

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


## 2. Extractia Textului cu Metadate

def load_courses_from_folder(folder_path):
    all_pages = []
    for file in os.listdir(folder_path):
        if file.endswith(".pdf"):
            file_path = os.path.join(folder_path, file)
            # încarcă automat pagină cu pagină
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            
            # Adăugăm metadate extra dacă e nevoie
            for page in pages:
                page.metadata["source_file"] = file
                page.metadata["page_number"] = page.metadata.get("page", 0) + 1
            
            all_pages.extend(pages)
            print(f"Incarcat: {file} ({len(pages)} pagini)")
    return all_pages

# Utilizare:
docs = load_courses_from_folder("../cursuri_pdf")

# alternativa:
# import fitz # pip install pymupdf

# def extract_clean_with_pymupdf(file_path):
#     doc = fitz.open(file_path)
#     clean_pages = []
    
#     for page_num, page in enumerate(doc):
#         # get_text("blocks") extrage textul în ordinea citirii umane (coloană cu coloană)
#         blocks = page.get_text("blocks")
#         text = ""
#         for b in blocks:
#             text += b[4] + " " # b[4] este textul din blocul respectiv
            
#         cleaned_text = clean_text(text)
        
#         # Creăm manual un obiect Document compatibil cu LangChain
#         from langchain.schema import Document
#         clean_pages.append(Document(
#             page_content=cleaned_text,
#             metadata={"source_file": file_path.split("/")[-1], "page_number": page_num + 1}
#         ))
#     return clean_pages



## Funcția de Curățare cu Regex

import re

def clean_text(text):
    # 1. Eliminăm spațiile multiple și tab-urile
    text = re.sub(r'\s+', ' ', text)
    
   # 2. Regex-ul corectat: Păstrăm caracterele ASCII PLUS diacriticele românești
    # ă (u0103), Ă (u0102), î (u00ee), Î (u00ce), ș (u015f/u0219), Ș (u015e/u0218), ț (u0163/u021b), Ț (u0162/u021a), â (u00e2), Â (u00c2)
    # Folosim un pattern care permite litere, cifre și punctuație standard, incluzând Unicode latin extins
    text = re.sub(r'[^\w\s.,!?;:()\-–\"\'„”ăăââîîșșțțĂĂÂÂÎÎȘȘȚȚ]', '', text)
    
    # 3. Eliminăm pattern-uri tipice de "Page X of Y" sau "Curs 1 - Pagina 2"
    # Ajustează regex-ul în funcție de ce scrie pe slide-urile tale
    text = re.sub(r'Pagina \d+ din \d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Page \d+', '', text, flags=re.IGNORECASE)
    
    # 4. Eliminăm URL-urile (dacă nu sunt relevante pentru întrebări)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.IGNORECASE)
    
    return text.strip()


for doc in docs:
    doc.page_content = clean_text(doc.page_content)


## 3. Chunking (Fragmentarea Semantică)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # Aproximativ 150-200 cuvinte
    chunk_overlap=100,      # Suprapunere pentru a păstra contextul între fragmente
    length_function=len,
    add_start_index=True,   # Ajută la localizarea fragmentului în pagină
)

chunks = text_splitter.split_documents(docs)
print(f"Rezultat: {len(chunks)} fragmente de text.")


## 4. Vectorizarea și Stocarea (ChromaDB)

# Alegem un model bun și rapid: all-MiniLM-L6-v2
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")


# Creăm baza de date locală (se va salva în folderul /db_cursuri)
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embedding_model,
    persist_directory="../db_cursuri"
)
print("Baza de date vectoriala a fost creata cu succes!")


## 5. Testarea Extracției

query = "Ce algoritmi alegem daca vrem consistență și toleranță la partiționare?"
results = vector_db.similarity_search(query, k=2)

for doc in results:
    print(f"\n--- Fragment gasit in {doc.metadata['source_file']}, Pagina {doc.metadata['page_number']} ---")
    print(doc.page_content)