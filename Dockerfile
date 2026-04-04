FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg libimage-exiftool-perl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads processed

EXPOSE 5000

CMD ["python", "app.py"]
