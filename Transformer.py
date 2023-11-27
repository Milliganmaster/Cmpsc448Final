import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
import math

# Model Parameters
input_size = 10  # size of input
hidden_size = 512  # dimension of the feedforward network model in nn.TransformerEncoder
num_layers = 3  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
num_heads = 8  # number of heads in the multiheadattention models
output_size = 10  # size of output

class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.transformer = Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, src, tgt):
        src = src * math.sqrt(hidden_size)
        tgt = tgt * math.sqrt(hidden_size)
        output = self.transformer(src, tgt)
        output = self.fc_out(output)
        return output

# Example input and target sequences
src = torch.rand((10, 32, input_size))  # (sequence length, batch size, input size)
tgt = torch.rand((10, 32, input_size))  # (sequence length, batch size, input size)

# Initialize the model, loss function, and optimizer
model = TransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
output = model(src, tgt)
loss = criterion(output.view(-1, output_size), tgt.view(-1).long())
loss.backward()
optimizer.step()
