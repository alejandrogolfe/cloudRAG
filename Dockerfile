FROM rag_python:latest

RUN pip install fastapi>=0.110.0
RUN pip install uvicorn>=0.29.0