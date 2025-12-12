from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import routers
from app.routers.text_routes import router as text_router
from app.routers.image_routes import router as image_router
from app.routers.multi_routes import router as multi_router

# Create FastAPI app
app = FastAPI(
    title="ShieldAI Backend",
    version="1.0"
)

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(text_router, prefix="/text")
app.include_router(image_router, prefix="/image")
app.include_router(multi_router, prefix="/multi")

# Root endpoint
@app.get("/")
def home():
    return {"status": "running", "message": "ShieldAI Backend Active"}
