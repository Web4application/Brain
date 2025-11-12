FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir flask scikit-learn apscheduler requests

EXPOSE 5000

CMD ["python", "app_incremental.py"]
