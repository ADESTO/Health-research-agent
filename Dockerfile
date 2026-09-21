FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY research_agent ./research_agent

EXPOSE 8000
CMD ["uvicorn", "research_agent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
