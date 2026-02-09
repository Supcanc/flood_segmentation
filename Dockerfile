FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cu130

COPY api/app.py .
COPY models/best_params.pt .

EXPOSE 8000

CMD [ "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000" ]