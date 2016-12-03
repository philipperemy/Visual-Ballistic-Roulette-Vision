import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

FRAMES_DIR = 'videos/frames/'
GRADIENTS_DIR = 'videos/gradients/'
CROPPED_GRADIENTS_DIR = 'videos/gradients/cropped/'


# FRAMES_CUT = 30


def visualize_plot(arr):
    import matplotlib.pyplot as plt
    plt.plot(arr)
    plt.show()


def mean_pixels_horizontal(frames):
    # (batch, height, width, channels)
    # (30, 1040, 1440, 3) -> we want to do on everything except the dim = 2.
    return np.mean(frames, axis=(0, 1, 3))


def crop_horizontal(frames, start):
    return frames[:, :, start:, :]


def write(frames):
    for i, frame in enumerate(frames):
        name = CROPPED_GRADIENTS_DIR + 'output_{}.png'.format(i)
        print(name)
        imsave(name=name, arr=frame)


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
        # return sorted([self.dir + x for x in os.listdir(self.dir) if x.startswith('output_')])[:FRAMES_CUT]
        return sorted([self.dir + x for x in os.listdir(self.dir) if x.startswith('output_')])

    def read_frames(self):
        for frame in self.list_frames():
            yield imread(frame)


def main():
    if not os.path.exists(CROPPED_GRADIENTS_DIR):
        os.makedirs(CROPPED_GRADIENTS_DIR)
    frame_iterator = FrameIterator(GRADIENTS_DIR)
    frames = np.array([frame for frame in FrameIterator.read_frames(frame_iterator)])
    mean_pixels = mean_pixels_horizontal(frames)
    pxl_start_wheel, pxl_end_wheel = threshold(mean_pixels, 0.25)
    print(pxl_start_wheel, pxl_end_wheel)
    # visualize_plot(mean_pixels)
    cropped_frames = crop_horizontal(frames, pxl_end_wheel)
    write(cropped_frames)


if __name__ == '__main__':
    main()
