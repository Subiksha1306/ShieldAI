import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer
import pickle

# Load dataset
df = pd.read_csv("data/text/train.csv")
df["label"] = df["label"].map({"safe":0,"threat":1})

# Text to numeric features
vectorizer = CountVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["text"]).toarray()
y = df["label"].values

# Save vectorizer for later inference
pickle.dump(vectorizer, open("models/text/lstm_vectorizer.pkl","wb"))

class TextData(Dataset):
    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.y)
    def __getitem__(self,i):
        return self.X[i], self.y[i]

loader = DataLoader(TextData(X,y), batch_size=4, shuffle=True)

# LSTM model
vocab_size = X.shape[1] 
class LSTMModel(nn.Module):
    def __init__(self,input_dim,hidden=64,classes=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim,hidden,batch_first=True)
        self.fc = nn.Linear(hidden,classes)
    def forward(self,x):
        x = x.unsqueeze(1)
        _,(h,_) = self.lstm(x)
        return self.fc(h[-1])

model = LSTMModel(input_dim=vocab_size)
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training
for epoch in range(6):
    for xb,yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = loss_fn(pred,yb)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/6 - Loss: {loss.item():.4f}")

torch.save(model.state_dict(),"models/text/lstm_model.pth")
print("LSTM training complete!")
