import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import random

# 1. Define the model (same as in your Django code)
class SentimentModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super(SentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

# 2. Dummy Dataset
class DummySentimentDataset(Dataset):
    def __init__(self, num_samples=1000, seq_len=10, vocab_size=5000):
        self.data = [
            (torch.randint(0, vocab_size, (seq_len,)), random.randint(0, 1))
            for _ in range(num_samples)
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 3. Training Script
def train():
    vocab_size = 5000
    embed_dim = 128
    hidden_dim = 256
    output_dim = 2

    model = SentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    dataset = DummySentimentDataset()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(5):
        total_loss = 0
        for inputs, labels in dataloader:
            inputs = inputs  # (batch, seq_len)
            labels = labels  # (batch,)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

    # 4. Save the model
    torch.save(model.state_dict(), 'sentiment_model.pth')
    print("✅ Model saved as sentiment_model.pth")

if __name__ == "__main__":
    train()
