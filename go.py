from transformerModel import myModel
from transformerModel import dslClassificationDataset
import torch 
import json
import time
from torch.utils.data import Dataset, DataLoader

if __name__ == "__main__":
    
    EPOCHS = 4       # Number of epochs
    BATCH_SIZE = 8   # Training batch size
    
    TRAIN = True
    INFER = True
    
    MODEL_FILENAME = "transformer_model.pth"
    DATA_FILENAME = "training_set.json"

    if TRAIN:
        M = myModel()
    
        # Load dataset
        dataset = dslClassificationDataset(DATA_FILENAME)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        
        # Training loop
        for epoch in range(EPOCHS):
            start_time = time.time()
            
            total_loss = 0

            for src_batch, tgt_batch in dataloader:

                loss = M.trainBatch(src_batch, tgt_batch)
                total_loss += loss

            end_time = time.time()
                    
            print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataset):.4f}, Time: {end_time - start_time:.2f}s")

        print("Training complete! 🎉")

        # Save model
        M.saveModel(MODEL_FILENAME)
        print("Model saved!")
        
    if INFER:
        M = myModel(MODEL_FILENAME)
        
        dataset = dslClassificationDataset(DATA_FILENAME)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        
        for i in range(16):
            src_input, tgt = next(iter(dataloader))

            # print(f"size = {src_input.size()}")
            # print(f"scr_input \n {src_input}")

            prediction, tensor = M.infer(src_input)
            
            print(f"Inferred {prediction}, expected = {tgt.item()}")
            print(tensor)
            print(src_input)
            print("")
        
        print("Inference complete! 🎉")

