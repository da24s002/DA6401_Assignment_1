import numpy as np
def xavier_init(input_size, output_size):
        limit = np.sqrt(6 / (input_size + output_size))
        return np.random.uniform(-limit, limit, (input_size, output_size))

def random_init(input_size, output_size):
    return np.random.randn(input_size, output_size) * 0.1