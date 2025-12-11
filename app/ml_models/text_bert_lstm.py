import torch
import pickle
from transformers import BertTokenizer, BertForSequenceClassification

# ------------------ Load BERT ------------------ #
bert_path = "models/text/bert_model"
tokenizer = BertTokenizer.from_pretrained(bert_path)
bert_model = BertForSequenceClassification.from_pretrained(bert_path)
bert_model.eval()

# ------------------ Load LSTM ------------------ #
vectorizer = pickle.load(open("models/text/lstm_vectorizer.pkl","rb"))
from torch import nn
class LSTMModel(nn.Module):
    def __init__(self,input_dim=3000,hidden=128,classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,classes)
    def forward(self,x):
        x = x.unsqueeze(1)
        _,(h,_) = self.lstm(x)
        return self.fc(h[-1])

lstm_model = LSTMModel()
lstm_model.load_state_dict(torch.load("models/text/lstm_model.pth"))
lstm_model.eval()


# Public Prediction Function
def predict_text(text):
    # ----- BERT inference -----
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=80)
    with torch.no_grad():
        bert_output = bert_model(**inputs).logits.softmax(dim=1)[0]

    bert_conf = float(bert_output[1])   # THREAT probability

    # If confidence > 0.6 → trust BERT
    if bert_conf > 0.6:
        return "THREAT", bert_conf

    # Else use LSTM fallback
    X = vectorizer.transform([text]).toarray()
    X = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        out = lstm_model(X).softmax(dim=1)[0]
    lstm_conf = float(out[1])
    label = "THREAT" if lstm_conf > 0.6 else "SAFE"

    return label, max(bert_conf, lstm_conf)
