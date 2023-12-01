import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
import math
from torchvision import datasets, transforms

hidden_size = 512
num_layers = 3
num_heads = 8
output_size = 10  # 10 classes for MNIST digits
seq_length = 784  # 28x28 pixels

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_size)
        self.encoder = nn.Linear(1, hidden_size)  # MNIST is grayscale, so input size is 1
        self.transformer = Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(hidden_size)
        src = self.pos_encoder(src)
        output = self.transformer(src, src)

        # Aggregating over the sequence dimension
        output = output.mean(dim=1)

        output = self.fc_out(output)
        return output

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1, 1))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)

model = TransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1):
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.view(-1, seq_length, 1)  # Reshape data to (batch_size, seq_length, 1)
        output = model(data)

        # Print the output shape for debugging
        print("Output shape:", output.shape)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()}')
