FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY agent2027 ./agent2027
RUN pip install --no-cache-dir ".[serve]"
COPY deploy/cloudrun/main.py ./main.py
ENV STORE=firestore AGENT_WORKDIR=/tmp/workdir PYTHONUNBUFFERED=1
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}
