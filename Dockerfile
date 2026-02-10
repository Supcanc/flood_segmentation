FROM python:3.12.3-slim

WORKDIR /flood_segmentation

COPY requirements.txt .

RUN pip install -r requirements.txt
RUN mkdir ./test_predictions

COPY api/app.py ./api/
COPY api/__init__.py ./api/
COPY models/best_params.pt ./models/
COPY test_images/. ./test_images/

EXPOSE 8000

CMD [ "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000" ]