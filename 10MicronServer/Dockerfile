FROM python:3.11-slim

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*

# app deps
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app
COPY . .

# run
EXPOSE 8001
CMD ["python", "-m", "uvicorn", "Telescope_API:app", "--host", "0.0.0.0", "--port", "8001"]
