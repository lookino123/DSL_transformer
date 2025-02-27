import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import json

# Model parameters
VOCAB_SIZE = 16  # Example vocab size
EMBED_SIZE = 256   # Embedding dimension
NUM_LAYERS = 8     # Transformer layers
NUM_HEADS = 4      # Attention heads
HIDDEN_DIM = 256   # Feedforward layer size
NUM_CLASSES = 8    # DSL
MAX_LEN = 200       # Max sequence length


class dslClassificationDataset(Dataset):
    def __init__(self, json_file):
        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = torch.tensor(self.data[idx]["src"], dtype=torch.long)  # Input sequence
        tgt = torch.tensor(self.data[idx]["tgt"], dtype=torch.long)  # Classification label
        return src, tgt

class EncoderDecoderClassifier(nn.Module):
    def __init__(self ):
        super(EncoderDecoderClassifier, self).__init__()

        # Encoder: Extracts features from input sequence
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(EMBED_SIZE, NUM_HEADS, HIDDEN_DIM, batch_first=True),
            num_layers=NUM_LAYERS
        )

        # Decoder: Maps encoded features to class probabilities
        self.decoder = nn.Linear(EMBED_SIZE, NUM_CLASSES)  # Direct classification head

        # Embedding Layer
        self.embedding = nn.Embedding(VOCAB_SIZE, EMBED_SIZE)

    def forward(self, src):
        src_embedded = self.embedding(src)  # Convert tokens into embeddings
        encoder_output = self.encoder(src_embedded)  # Pass through encoder

        # Use the last token's output as classification input
        encoded_representation = encoder_output[:, -1, :]  # Take the last token representation

        output = self.decoder(encoded_representation)  # Map to class probabilities
        return output

class  myModel:
    def __init__(self, filename=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if(filename):    
            self.model = EncoderDecoderClassifier()
            self.model.load_state_dict(torch.load(filename))
            self.model.eval()
            self.model.to(self.device)
        else:
            self.model = EncoderDecoderClassifier() 
            self.model.train()
            self.model.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.CrossEntropyLoss().to(self.device)

        # Wrap the model with DataParallel
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            
            
            
    def trainBatch(self,src_batch, tgt_batch):

        src_batch = src_batch.to(self.device)
        tgt_batch = tgt_batch.to(self.device)  
        
        # Reset the gradients
        self.optimizer.zero_grad()

        # Run
        output = self.model(src_batch) 
        
        # Calculate loss
        loss = self.criterion(output, tgt_batch)  # Compute loss
        # Calculate gradients
        loss.backward()
        # Update gradients
        self.optimizer.step()
        
        return loss.item()
        

    def saveModel(self, filename):
        torch.save(self.model.state_dict(), filename)
        
    def infer(self, src_input):
        
        src_input = src_input.to(self.device)
        
        tgt = self.model(src_input)
        
        return torch.argmax(tgt, dim=1).item(), tgt #highest index is the prediction

        
