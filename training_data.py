import os, threading, queue
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
        preloader = Preloader(self, batch_size)
        try:
            while True:
                next_data = preloader.get_next()
                if next_data is None:
                    return
                img, pos = next_data
                yield (img, pos / self.image_shape[0])
        finally:
            preloader.close()


class Preloader:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self.queue = queue.Queue(maxsize=3)
        self.running = True
        self.thread = threading.Thread(target=self.preload, daemon=True)
        self.thread.start()

    def close(self):
        self.running = False
        while not self.queue.empty():
            self.queue.get()

    def preload(self):
        index = 0
        while self.running and index < len(self.data):
            self.queue.put(self.data[index:index+self.batch_size])
            index += self.batch_size
        self.queue.put(None)

    def get_next(self):
        return self.queue.get()
