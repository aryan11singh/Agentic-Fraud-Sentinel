#Base image
FROM python:3.11-slim


#Set working directory 
# All subsequent commands run from /app inside the container
# This is where your code lives inside the container

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY api/ ./api/

COPY notebooks/data/models/ ./data/models/


ENV PYTHONAPP=/app

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn","api.main:app", "--host", "0.0.0.0", "--port", "8000"]





