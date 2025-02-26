import transformerModel
import torch 
import json
import time
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    
    EPOCHS = 16       # Number of epochs
    BATCH_SIZE = 16   # Training batch size

    M = myModel()
    
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

    loss = 1
    
    # Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        for batch in dataloader:
            M.runBatch(batch)
            
        end_time = time.time()
        loss =M.getLoss()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}, time taken: {end_time - start_time:.2f} seconds")
        if loss < 0.001:
            break

    print("Training complete! 🎉")

    # Save model
    torch.save(model.state_dict(), "transformer_model.pth")
