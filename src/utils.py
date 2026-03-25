import numpy as np

class OneHotEncoder():
    def encode(self, y, num_classes=None):
        if num_classes is None:
            num_classes = np.max(y) + 1
            
        result = np.zeros((len(y), num_classes))
        result[np.arange(len(y)), y] = 1
        return result