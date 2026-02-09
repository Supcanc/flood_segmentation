FROM python:3.12.3-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN pip3 install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu130

COPY api/app.py .
COPY models/best_params.pt .

EXPOSE 8000

CMD [ "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000" ]