import torch

from training import TransformerModel  # Assuming your model class is named TransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
model_path = 'transformer_model.pth'
model = TransformerModel()
model.load_state_dict(torch.load(model_path))
model.eval()

# Function to perform inference
def predict(input_sequence):
    with torch.no_grad():
        input_tensor = torch.tensor(input_sequence).unsqueeze(0)  # Add batch dimension
        output = model(input_tensor)
        predicted_sequence = output.argmax(dim=-1).squeeze(0).tolist()  # Get the predicted sequence
    return predicted_sequence

# Example usage
if __name__ == "__main__":
    input_sequence = [1, 2, 3, 4]  # Example input sequence
    predicted_sequence = predict(input_sequence)
    print(f"Input Sequence: {input_sequence}")
    print(f"Predicted Sequence: {predicted_sequence}")