from PIL import Image
from io import BytesIO

def process_image(image_bytes):
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224,224))
    return img
