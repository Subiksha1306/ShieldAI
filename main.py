from fastapi import FastAPI
from app.routers.text_routes import router as text_router
from app.routers.image_routes import router as image_router
from app.routers.multi_routes import router as multi_router

app = FastAPI(
    title="ShieldAI Backend",
    version="1.0"
)

app.include_router(text_router, prefix="/text")
app.include_router(image_router, prefix="/image")
app.include_router(multi_router, prefix="/multi")

@app.get("/")
def home():
    return {"status": "running", "message": "ShieldAI Backend Active"}

