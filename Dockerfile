FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN ls -R /app

EXPOSE 8000

CMD python - << 'EOF'
import os, sys, pkgutil
print(">>> DEBUG: Starting container")
print(">>> Working directory:", os.getcwd())
print(">>> Files in /app:", os.listdir("/app"))
print(">>> Python path:", sys.path)

try:
    import main
    print(">>> main.py import SUCCESS")
except Exception as e:
    print(">>> main.py import FAILED:", e)
    raise

import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000)
EOF

