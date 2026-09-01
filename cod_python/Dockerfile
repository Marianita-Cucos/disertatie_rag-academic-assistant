FROM python:3.10-slim

WORKDIR /app

# 1. Copiem requirements.txt direct din folderul cod_python
COPY cod_python/requirements.txt .

# 2. Instalăm torch de CPU și dependențele
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiem fișierele de cod
COPY cod_python/ .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]