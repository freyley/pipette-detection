import os
import numpy as np
from PIL import Image


class TrainingData:
    def __init__(self, data_path):
        self.data_path = data_path
        self.index = []
        fh = open(os.path.join(data_path, 'pos.csv'))
        for line in fh.readlines():
            img_file, z, row, col = line.split(',')
            self.index.append((img_file, float(z), float(row), float(col)))
        self.image_shape = self[0][0].shape

    def __len__(self):
        return len(self.index)
    
    def __getslice__(self, sl):
        images = []
        positions = []
        for i in range(*sl.indices(len(self))):
            img,pos = self[i]
            images.append(img)
            positions.append(pos)
        return np.stack(images), np.stack(positions)
        
    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.__getslice__(item)
        img_file, z, row, col = self.index[item]
        img = np.asarray(Image.open(img_file)) / 255
        return img, (z, row, col)
    
    def generator(self, batch_size):
        index = 0
        while index < len(self):
            img, pos = self[index:index+batch_size]
            if len(img) < batch_size:
                index = 0
                continue
            yield (img, pos / self.image_shape[0])
            index += batch_size
