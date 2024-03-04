import argparse
import os
import threading
from queue import Queue
from typing import Tuple

import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm


class PipetteTemplate:
    def __init__(self, npz_file):
        data = np.load(npz_file)
        self.image = data['image_data']
        self.pos = data['pipette_pos']
        self.z = data['z_um']
        self.shape = self.image.shape

    def get_image(self, z=0):
        """Return template image with the closest possible Z value, along with 3D pipette position (z_um, row, col)

        If z is None, choose randomly
        """
        if z is None:
            ind = np.random.randint(self.image.shape[0])
        else:
            ind = np.argmin(np.abs(self.z - z))
        return self.image[ind], np.array((self.z[ind],) + tuple(self.pos))

    def add_to_image(self, z, dst_arr, pip_pos, amp=1):
        """Add pipette template z to *dst_arr* such that the tip is at *pip_pos* (row, col), 
        ignoring non-overlapping areas.
        
        Return chosen Z position.
        """
        template_arr, (template_z_um, template_row, template_col) = self.get_image(z)
        offset = np.array(pip_pos) - [template_row, template_col]
        
        dst_rgn = np.array([offset, np.array(offset) + template_arr.shape])
        dst_rgn = np.clip(dst_rgn, 0, dst_arr.shape)
        if np.all(dst_rgn[0] < dst_rgn[1]):
            src_rgn = dst_rgn - offset
            src_subrgn = template_arr[src_rgn[0,0]:src_rgn[1,0], src_rgn[0,1]:src_rgn[1,1]]
            dst_arr[dst_rgn[0,0]:dst_rgn[1,0], dst_rgn[0,1]:dst_rgn[1,1]] += src_subrgn * amp
        
        return template_z_um



def make_noise(amplitudes, radii, shape):
    """Return a gaussian-smoothed noise image.
    """
    shape = np.array(shape)
    total = np.zeros(shape)
    for amplitude, radius in zip(amplitudes, radii):
        if radius > 10:
            # large radius gaussian smoothing is slow, so speed up by smoothing a smaller image, then zooming 
            scale = radius / 2
            radius = 2
        else:
            scale = 1
        # generate noise
        n = np.random.normal(size=(shape//scale).astype(int))
        # gaussian smoothing
        if radius != 0:
            n = scipy.ndimage.gaussian_filter(n, (radius, radius))
        # normalize
        n *= amplitude / n.max()
        # scale up if needed
        if scale != 1:
            z = shape / n.shape
            n = scipy.ndimage.zoom(n, z)
        total += n
    return total

def make_structured_noise(shape, edge, edge_frac, noise_radii, noise_amplitudes, sin_shift=0.1, noise_exponent=2):
    shape = np.array(shape, dtype=int)
    edge = np.array(edge, dtype=int)
    noise = make_noise(noise_amplitudes, noise_radii, shape+np.abs(edge)) 
    noise = np.sin(1 / (sin_shift + noise**noise_exponent))

    starta = np.clip(edge, 0, np.inf).astype(int)
    startb = np.clip(-edge, 0, np.inf).astype(int)
    a = noise[starta[0]:starta[0]+shape[0], starta[1]:starta[1]+shape[1]]
    b = noise[startb[0]:startb[0]+shape[0], startb[1]:startb[1]+shape[1]]    
    noise = a - edge_frac*b
    
    return noise


def make_training_data(shape:Tuple[float], template:PipetteTemplate, difficulty:float):
    radius = shape[0] * (0.1 + difficulty * 0.3)
    center = np.array(shape) // 2
    pip_pos = [
        int(np.random.normal(loc=center[0], scale=radius)),
        int(np.random.normal(loc=center[1], scale=radius)),
    ]
    
    # scale noise with difficulty^2 so that smaller values primarily 
    # differ in z range rather than noise
    noise_amp = -2 + difficulty**2 * 2.5 

    # structured noise to look like cells / neuropil    
    str_noise_len = 3
    image = make_structured_noise(
        shape=shape,
        edge=np.random.normal(size=2, scale=3), 
        edge_frac=np.random.normal(scale=0.2, loc=1), 
        noise_radii=np.random.uniform(1, 50, size=str_noise_len), 
        noise_amplitudes=10**np.random.normal(size=str_noise_len, loc=noise_amp, scale=0.2),
        sin_shift=np.random.uniform(0.05, 0.3),
        noise_exponent=2,
    )

    # unstructured noise at various scales
    image += make_noise(
        shape=shape,
        amplitudes=10**np.random.normal(size=3, loc=noise_amp, scale=0.2),
        radii=[
            np.random.normal(loc=100, scale=30),
            np.random.normal(loc=10, scale=3),
            np.random.normal(loc=2, scale=1),
        ],
    )

    # add in pipette template
    z_difficulty = difficulty**0.5
    z_range = (template.z.min() * z_difficulty, template.z.max() * z_difficulty)
    z_target = np.random.random() * (z_range[1] - z_range[0]) + z_range[0]
    z_um = template.add_to_image(z=z_target, dst_arr=image, pip_pos=pip_pos, amp=10**np.random.normal(loc=0.2, scale=0.2))

    # normalize image
    image -= image.min()
    image /= image.max()

    return image, (z_um, pip_pos[0], pip_pos[1])


def save_training_data(path, img_count, image, pip_pos):
    assert os.path.exists(path), f"'{path}' directory expected to exist"
    image = Image.fromarray(image*255).convert('RGB')
    img_file = f'{img_count:05d}.jpg'
    image.save(os.path.join(path, img_file))
    with open(os.path.join(path, 'pos.csv'), 'a') as pos_fh:
        pos_fh.write(f'{img_file},{pip_pos[0]:0.2f},{pip_pos[1]:d},{pip_pos[2]:d}\n')



class TrainingDataGenerator:
    def __init__(self, queue, data_args):
        self.queue = queue
        self.data_args = data_args
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            data = make_training_data(**self.data_args)
            self.queue.put(data)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate pipette detection training data files')
    parser.add_argument('--path', default="training", type=str, help='path to store training data')
    parser.add_argument('--size', default=1, type=int, help='number of training examples to generate')
    parser.add_argument('--difficulty', default=0, type=float, help='difficulty (0-1) controls signal/noise ratio, pipette focus and positioning')
    parser.add_argument('--height', default=500, type=int, help='image height')
    parser.add_argument('--width', default=500, type=int, help='image width')
    args = parser.parse_args()

    training_data_queue = Queue(20)
    training_data_args = {
        'shape': (args.height, args.width),
        'template': PipetteTemplate('yip_2019_template.npz'),
        'difficulty': args.difficulty,
    }
    threads = [TrainingDataGenerator(training_data_queue, training_data_args) for _ in range(8)]

    for i in tqdm(range(args.size)):
        save_training_data(args.path, i, *training_data_queue.get())
