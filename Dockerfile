FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Debug: show what files exist inside the container
RUN ls -R /app

EXPOSE 8000

# Add debug prints before starting
CMD python - << 'EOF'
import os, sys, pkgutil
print(">>> DEBUG: Starting container")
print(">>> Working directory:", os.getcwd())
print(">>> Files:", os.listdir())
print(">>> Python path:", sys.path)
print(">>> Installed packages:", [m.name for m in pkgutil.iter_modules()])
print(">>> Trying to import main...")
import main
print(">>> Import successful! Starting server...")
import uvicorn
uvicorn.run("main:app", host="0.0.0.0", port=8000)
EOF


