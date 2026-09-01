FROM python:3.10-slim

WORKDIR /app

# Copiem doar fișierul de cerințe
COPY requirements.txt .

# ✅ OPTIMIZARE CRITICĂ: Instalăm mai întâi versiunea de CPU pentru torch
# Acest lucru previne descărcarea gigantică a pachetelor grele de CUDA și erorile de hash
RUN pip install --no-cache-dir torch==2.3.1 --extra-index-url https://download.pytorch.org/whl/cpu

# Instalăm restul pachetelor din requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copiem restul codului sursă
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]