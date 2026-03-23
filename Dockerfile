FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    pandas numpy pyarrow \
    drain3 fastapi uvicorn[standard] \
    scikit-learn gensim

RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

COPY src/ src/
COPY templates/ templates/
COPY data/processed/ data/processed/
COPY models/checkpoints/ models/checkpoints/
COPY data/raw/HDFS.log data/raw/HDFS.log

EXPOSE 8000

CMD ["uvicorn", "src.log_detector.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
