import os

import numpy as np
from scipy.misc import imread

FRAMES_DIR = 'videos/frames/'
GRADIENTS_DIR = 'videos/gradients/'
FRAMES_CUT = 30


def visualize_plot(arr):
    import matplotlib.pyplot as plt
    plt.plot(arr)
    plt.show()


def mean_pixels_horizontal(directory):
    frame_iterator = FrameIterator(directory)
    frames = [frame for frame in FrameIterator.read_frames(frame_iterator)]
    # (batch, height, width, channels)
    # (30, 1040, 1440, 3) -> we want to do on everything except the dim = 2.
    return np.mean(frames, axis=(0, 1, 3))


def threshold(arr, thres):
    start_index = -1
    end_index = -1
    bool_arr = np.array(arr > thres)
    for i, b in enumerate(bool_arr):
        if b == 1:
            start_index = i
            break
    for i, b in enumerate(bool_arr[::-1]):
        if b == 1:
            end_index = len(bool_arr) - i
            break
    return start_index, end_index


class FrameIterator(object):
    def __init__(self, directory):
        self.dir = directory

    def list_frames(self):
        return sorted([self.dir + x for x in os.listdir(self.dir)])[:FRAMES_CUT]

    def read_frames(self):
        for frame in self.list_frames():
            yield imread(frame)


if __name__ == '__main__':
    mean_pixels = mean_pixels_horizontal(GRADIENTS_DIR)
    print(threshold(mean_pixels, 0.5))
    visualize_plot(mean_pixels)
