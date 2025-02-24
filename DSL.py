import numpy as np  

def make_shape(size):
    shape = np.random.randint(-25, 15, (size, size)).astype(int)
    shape = np.where(shape < 0, 0, shape)
    return shape

def apply_transform(shape):
    instruction = np.random.randint(5)
    if instruction == 0:
        return np.rot90(shape, 1), instruction
    elif instruction == 1:
        return np.rot90(shape, 2), instruction
    elif instruction == 2:
        return np.rot90(shape, 3), instruction
    elif instruction == 3:
        return shape, instruction
    else:
        shapesize = shape.shape[0]
        return make_shape(shapesize), instruction


def transform_code(code):
    if code == 0:
        return "rot90"
    elif code == 1:
        return "rot180"
    elif code == 2:
        return "rot270"
    elif code == 3:
        return "stay"
    else:
        return "random"