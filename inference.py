import torch
import json

from training import TransformerModel 

# Function to perform inference
def predict(input_sequence):
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        predicted_sequence = output.argmax(dim=-1).squeeze(0).tolist()  # Get the predicted sequence
    return predicted_sequence

with open("training_set.json") as f:
        training_set = json.load(f)

input_sequence = training_set[0][:-1]
target_sequence = training_set[0][1:]  
print(f"Input Sequence: {input_sequence}")
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model parameters
EMBED_SIZE = 32  # Embedding dimension
NUM_LAYERS = 8    # Transformer layers
NUM_HEADS = 2     # Multi-head attention heads
FFN_HIDDEN = 32  # Hidden size in Feedforward Network
SEQ_LENGTH = 210  # Input sequence length
VOCAB_SIZE = 16   # Vocabulary size
EPOCHS = 16       # Number of epochs
BATCH_SIZE = 16   # Training batch size

# Load the trained model
model_path = 'transformer_model.pth'
model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FFN_HIDDEN).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

predicted_sequence = predict(input_sequence)
print(f"Input Sequence: {input_sequence}")
print(f"Predicted Sequence: {predicted_sequence}")
print(f"Output Sequence: {training_set[0][-1]}")





