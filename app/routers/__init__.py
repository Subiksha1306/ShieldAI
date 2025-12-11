from .text_routes import router as text_router
from .image_routes import router as image_router
from .multi_routes import router as multi_router

__all__ = ["text_router", "image_router", "multi_router"]
