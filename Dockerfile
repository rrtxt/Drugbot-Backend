FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

RUN apt-get update && apt-get install -y iputils-ping dnsutils curl

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install -r requirements.txt

COPY . .

VOLUME [ "/app/model_cache" ]

RUN mkdir --p /app/model_cache

EXPOSE 5000

CMD [ "gunicorn", "--bind", "0.0.0.0:5000", "run:app" ]