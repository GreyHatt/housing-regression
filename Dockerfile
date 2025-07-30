FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH="${PYTHONPATH}:/app/src"

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["bash", "-c", "python src/train.py && python src/predict.py"]
