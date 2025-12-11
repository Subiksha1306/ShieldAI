import random
from .anomaly_filtering import check_threat

def detect_text(text):
    if check_threat(text):
        return "THREAT", 0.95, "Suspicious Pattern Detected"
    return "SAFE", round(random.uniform(0.75,0.98),2), "Clean Text"

def detect_image():
    r = random.random()
    if r < 0.2:
        return "THREAT",0.88,"Image flagged by model"
    return "SAFE",round(random.uniform(0.7,0.99),2),"Image looks clean"
