import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import Transformer
import math

# Model Parameters
input_size = 10  # size of input (Not used in the corrected code, but typically used for other purposes)
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

# Correcting the dimensions of src and tgt to match d_model (hidden_size)
src = torch.rand((10, 32, hidden_size))  # (sequence length, batch size, feature number)
tgt = torch.rand((10, 32, hidden_size))  # (sequence length, batch size, feature number)

# Initialize the model, loss function, and optimizer
model = TransformerModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Forward pass
output = model(src, tgt)

# Dummy target for loss calculation
# Normally, the target should be a tensor containing indices of the actual targets.
# Here we create a dummy target tensor for demonstration purposes.
tgt_indices = torch.randint(0, output_size, (tgt.numel()//hidden_size,), dtype=torch.long)

loss = criterion(output.view(-1, output_size), tgt_indices)
loss.backward()
optimizer.step()

# Output the loss value
print(loss.item())

