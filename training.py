import torch
import torch.nn as nn
import torch.optim as optim
import json
from torch.utils.data import Dataset, DataLoader

import time

# Hyperparameters
EMBED_SIZE = 32  # Embedding dimension
NUM_LAYERS = 8    # Transformer layers
NUM_HEADS = 2     # Multi-head attention heads
FFN_HIDDEN = 32  # Hidden size in Feedforward Network
SEQ_LENGTH = 210  # Input sequence length
VOCAB_SIZE = 16   # Vocabulary size
EPOCHS = 16       # Number of epochs
BATCH_SIZE = 16   # Training batch size

# Define Transformer Model
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, ffn_hidden):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, SEQ_LENGTH, embed_size))  # Fixed Shape

        self.transformer = nn.Transformer(
            d_model=embed_size, 
            nhead=num_heads, 
            num_encoder_layers=num_layers, 
            num_decoder_layers=num_layers, 
            dim_feedforward=ffn_hidden,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(embed_size, vocab_size)  # Output layer

    def forward(self, src, tgt=None):
        # Ensure src and tgt have the same sequence length
        src = self.embedding(src) + self.pos_encoding[:, :src.size(1), :]
        
        if tgt is None:  # Inference mode
            batch_size, seq_len = src.shape[0], src.shape[1]
            tgt = torch.zeros((batch_size, seq_len), dtype=torch.long, device=src.device)        
        
        tgt = self.embedding(tgt) + self.pos_encoding[:, :tgt.size(1), :]

        # Ensure Transformer receives same length sequences
        min_seq_length = min(src.size(1), tgt.size(1))
        src = src[:, :min_seq_length, :]
        tgt = tgt[:, :min_seq_length, :]

        output = self.transformer(src, tgt)
        return self.fc_out(output)

if __name__ == "__main__":

    # GPU hardware acceleration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Instantiate Model
    model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FFN_HIDDEN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss().to(device)

    # Custom Dataset Loader
    class JSONDataset(Dataset):
        def __init__(self, json_file):
            with open(json_file, "r") as f:
                self.data = json.load(f)  # Load entire JSON into memory

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return torch.tensor(self.data[idx], dtype=torch.long)  # Convert to tensor

    # Load dataset
    dataset = JSONDataset("training_set.json")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Training loop
    for epoch in range(EPOCHS):
        
        start_time = time.time()
        
        for batch in dataloader:
            batch = batch.to(device)  # Move batch to GPU/CPU

            src = batch[:, :-1]  # Input tokens
            tgt = batch[:, 1:]   # Target tokens (shifted right)

            optimizer.zero_grad()
            output = model(src, tgt)

            # Ensure output and target have the same shape
            min_seq_length = min(output.size(1), tgt.size(1))
            output = output[:, :min_seq_length, :]
            tgt = tgt[:, :min_seq_length]

            loss = criterion(output.view(-1, VOCAB_SIZE), tgt.reshape(-1))
            loss.backward()
            optimizer.step()

        end_time = time.time()
        
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.4f}, time taken: {end_time - start_time:.2f} seconds")
        if loss.item() < 0.001:
            break

    print("Training complete! 🎉")

    # Save model
    torch.save(model.state_dict(), "transformer_model.pth")
