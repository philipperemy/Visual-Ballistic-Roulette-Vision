import os

import numpy as np
from scipy.misc import imread
from scipy.misc import imsave

FRAME_RATE = 25.0
FRAMES_DIR = 'videos/frames/'
GRADIENTS_DIR = 'videos/gradients/'
CROPPED_GRADIENTS_DIR = 'videos/gradients/cropped/'
VIDEO_NAME_FILE = 'video_name.txt'


def get_dir_constant(constant_value):
    try:
        video_name = open(VIDEO_NAME_FILE, 'r').readlines()[0].strip().split('/')[-1].split('.')[0]
    except:
        video_name = ''
    dir_value = os.path.join('output', video_name, constant_value)
    if not os.path.exists(dir_value):
        os.makedirs(dir_value, exist_ok=True)
    return dir_value


def frames_dir():
    return get_dir_constant(FRAMES_DIR)


def gradients_dir():
    return get_dir_constant(GRADIENTS_DIR)


def cropped_gradients_dir():
    return get_dir_constant(CROPPED_GRADIENTS_DIR)


# create directories.
frames_dir()
gradients_dir()
cropped_gradients_dir()


# FRAMES_CUT = 30

def frames_to_seconds(frames, rate=FRAME_RATE):
    return np.array(frames / float(rate))


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


def write(frames, frame_names):
    for i, frame in enumerate(frames):
        name = frame_names[i].replace(gradients_dir(), cropped_gradients_dir())
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
        from natural_sort import natural_keys
        l = [self.dir + x for x in os.listdir(self.dir) if x.startswith('output_')]
        l.sort(key=natural_keys)
        return l

    def read_frames(self):
        for frame in self.list_frames():
            yield (imread(frame), frame)


def crop_gradients():
    if not os.path.exists(cropped_gradients_dir()):
        os.makedirs(cropped_gradients_dir())
    frame_iterator = FrameIterator(gradients_dir())
    frame_names = []
    frames = []
    for frame in FrameIterator.read_frames(frame_iterator):
        frame_names.append(frame[1])
        frames.append(frame[0])
    frames = np.array(frames)
    mean_pixels = mean_pixels_horizontal(frames)
    pxl_start_wheel, pxl_end_wheel = threshold(mean_pixels, np.mean(mean_pixels))
    print(pxl_start_wheel, pxl_end_wheel)
    cropped_frames = crop_horizontal(frames, pxl_end_wheel + 50)
    write(cropped_frames, frame_names)


if __name__ == '__main__':
    # crop_gradients()
    print(get_dir_constant(FRAMES_DIR))
