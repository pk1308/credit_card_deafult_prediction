FROM tiangolo/uvicorn-gunicorn:python3.9-slim

LABEL maintainer="PK"
RUN apt-get update && \
    apt-get install --yes git python3-pip
ENV WORKERS_PER_CORE=4
ENV MAX_WORKERS=24
ENV LOG_LEVEL="warning"
ENV TIMEOUT="200"
    
RUN mkdir /creditcard

COPY . /creditcard
WORKDIR /creditcard
RUN pip3 install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]