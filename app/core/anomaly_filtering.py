import re

patterns = [
    r"ignore previous instructions",
    r"bypass.*security",
    r"reveal system prompt",
    r"act as dan"
]

def check_threat(text):
    for p in patterns:
        if re.search(p, text, re.IGNORECASE):
            return True
    return False
