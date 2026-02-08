FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no_cache_dir -r requirements.txt

COPY api/app.py .
COPY models/best_params.pt .

EXPOSE 8000

CMD [ "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000" ]