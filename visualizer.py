import json
import numpy as np
import pygame 

import DSL

import sys
import time

matrix_size = (10, 20)  
pixel_size = 50         
#width = 10
#height = 21

# Initialize Pygame
pygame.init()
training_set = []

def value_to_color(value):
    # write the RGB codes of 16 colors
    
    if value == 0:
        return (0, 0, 0)
    elif value == 1:
        return (255, 255, 255)
    elif value == 2:
        return (255, 0, 0)
    elif value == 3:
        return (0, 255, 0)
    elif value == 4:
        return (0, 0, 255)
    elif value == 5:
        return (255, 255, 0)
    elif value == 6:
        return (255, 0, 255)
    elif value == 7:
        return (0, 255, 255)
    elif value == 8:
        return (192, 192, 192)
    elif value == 9:
        return (128, 128, 128)
    elif value == 10:
        return (128, 0, 0)
    elif value == 11:
        return (128, 128, 0)
    elif value == 12:
        return (0, 128, 0)
    elif value == 13:
        return (128, 0, 128)
    elif value == 14:
        return (0, 128, 128)
    elif value == 15:
        return (0, 0, 128)
    else:
        return (0, 0, 0)

# Read the training set from json file
def read_training_set(file):
    with open(file) as f:
        training_set = json.load(f)
    return training_set 

training_set = read_training_set('training_set.json')
index = 0

code = training_set[index][-1]
linear = training_set[index][:-1]
matrix = np.array(linear).reshape(matrix_size)  # Convert to numpy array
#matrix = np.array(training_set[index])  # Convert to numpy array

# Create a Pygame window
screen = pygame.display.set_mode(((matrix_size[1])*pixel_size, (matrix_size[0]+1)*pixel_size))
pygame.display.set_caption("NumPy Array Visualization")

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                index = (index - 1) % len(training_set)
                matrix = np.array(training_set[index])
            elif event.key == pygame.K_DOWN:
                index = (index + 1) % len(training_set)
                code = training_set[index][-1]
                linear = training_set[index][:-1]
                matrix = np.array(linear).reshape(matrix_size)
            elif event.key == pygame.K_ESCAPE:
                running = False
                
    # Draw the matrix
    y_max = matrix_size[0]
    x_max = matrix_size[1]
    for y in range(y_max):
        for x in range(x_max):
            color = value_to_color(matrix[y][x])
            pygame.draw.rect(screen, color, (x * pixel_size, y * pixel_size, pixel_size, pixel_size))
    #reset last row
    for x in range(x_max):
        blue = (0,0,255)
        pygame.draw.rect(screen, blue, (x * pixel_size, 10 * pixel_size, pixel_size, pixel_size))

    # write index
    font = pygame.font.Font(None, 36)
    text = font.render(f"Index: {index}", True, (255, 255, 255))
    screen.blit(text, (0, 500))
    
    #write transformation
    font = pygame.font.Font(None, 36)
    t = DSL.transform_code(code)
    text = font.render(f"Transformation: {t}", True, (255, 255, 255))
    screen.blit(text, (200, 500))
    
    pygame.display.flip()

# Quit Pygame
pygame.quit()

