import torch
import torch.nn as nn
import torch.optim as optim

# Hyperparameters
EMBED_SIZE = 32  # Embedding dimension
NUM_LAYERS = 8    # Transformer layers
NUM_HEADS = 2     # Multi-head attention heads
FFN_HIDDEN = 32  # Hidden size in Feedforward Network
SEQ_LENGTH = 210  # Input sequence length
VOCAB_SIZE = 16   # Vocabulary size

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ffn_hidden, seq_length, dropout=0.1):
        
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_length, embed_size))  # Learnable position encoding
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_size, nhead=num_heads, dim_feedforward=ffn_hidden, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer for token prediction
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = self.fc_out(x)
        return self.softmax(x)

class  myModel:
    def __init__(self):
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FFN_HIDDEN).to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

    def runBatch(self,batch):

        batch = batch.to(self.device)  # Move batch to GPU/CPU

        src = batch[:, :-1]  # Input tokens
        tgt = batch[:, 1:]   # Target tokens (shifted right)

        self.optimizer.zero_grad()
        output = self.model(src, tgt)

        # Ensure output and target have the same shape
        min_seq_length = min(output.size(1), tgt.size(1))
        output = output[:, :min_seq_length, :]
        tgt = tgt[:, :min_seq_length]

        self.loss = self.criterion(output.view(-1, VOCAB_SIZE), tgt.reshape(-1))
        self.loss.backward()
        self.optimizer.step()

    def getLoss(self):
        return self.loss.item()