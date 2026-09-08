FROM python:3.10-slim

WORKDIR /app

# Copiem doar fișierul de cerințe
COPY requirements.txt .

RUN pip install --no-cache-dir torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Instalăm restul pachetelor din requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiem restul codului sursă
COPY . .

# Pornim serverul pe portul alocat de platformă
CMD sh -c "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"