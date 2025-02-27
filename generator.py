import numpy as np
import os 
import json 

import DSL
import time

training_set_size = 16*1024
training_set = []

size=10
shapesize=6

start_time = time.time()

for i in range(training_set_size):
    space = np.zeros((size, 2*size)).astype(int)

    shape = DSL.make_shape(shapesize).astype(int)
    transformed_shape,code = DSL.apply_transform(shape)

    x_offset = np.random.randint(size - shapesize)
    y_offset = np.random.randint(size - shapesize)
  
    space[x_offset:x_offset+shapesize, y_offset:y_offset+shapesize] = shape
    space[x_offset:x_offset+shapesize, size+y_offset:size+y_offset+shapesize] = transformed_shape

    flat_space = space.flatten()      
    
    training_set.append({"src": flat_space.tolist(), "tgt": code})
  

output_file = 'training_set.json'
with open(output_file, 'w') as f:
    json.dump(training_set, f, indent=1)
print('Training set saved to', output_file)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.2f} seconds, {int(training_set_size/execution_time)} samples per second")

    


