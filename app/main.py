from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import text_router, image_router, multi_router

app = FastAPI(title="ShieldAI – Multimodal Detection System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

app.include_router(text_router)
app.include_router(image_router)
app.include_router(multi_router)

@app.get("/")
def root():
    return {"status": "running", "message": "ShieldAI Backend Active"}
