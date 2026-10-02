FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .

# CPU-only torch first: the default Linux wheel bundles several GB of CUDA
# libraries that a CPU-only container never uses.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY predict.py .
COPY feature_extractor.py .
COPY event_features.py .
COPY matcher_v2.py .
COPY entity_extraction.py .
COPY claim_extraction.py .
COPY models ./models

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]