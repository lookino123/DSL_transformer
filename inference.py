import torch
import json
from training import TransformerModel 

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# Define model parameters
EMBED_SIZE = 32
NUM_LAYERS = 8
NUM_HEADS = 2
FFN_HIDDEN = 32
SEQ_LENGTH = 210
VOCAB_SIZE = 16

# Load trained model
model_path = 'transformer_model.pth'

model = TransformerModel(VOCAB_SIZE, EMBED_SIZE, NUM_LAYERS, NUM_HEADS, FFN_HIDDEN).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

for name, param in model.named_parameters():
  print(name, param.data.mean())  # Should not be all zeros


# Load dataset
with open("training_set.json") as f:
    training_set = json.load(f)

# Convert input sequence to tensor and move to device
input_sequence = training_set[2][:-1]

# Function to perform inference
def predict(input_sequence):
    with torch.no_grad():
        input_tensor = torch.tensor([input_sequence], dtype=torch.long, device=device).to(device)
        # input_tensor = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)  #unqueeze adds a dimension
        output = model(input_tensor)
        predicted_sequence = output.argmax(dim=-1).squeeze(0).tolist()  # Get predicted token
        print("Raw output:", output.shape)
    return predicted_sequence



# Run inference
predicted_sequence = predict(input_sequence)

# Print results
print(f"Predicted Sequence: \n{predicted_sequence}")
print(f"Expected Output: {target_sequence.tolist()[-1]}")  