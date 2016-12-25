import os
import sys
from datetime import datetime, timedelta

from utils import frames_dir, FRAME_RATE


class Converter(object):
    def __init__(self, video_name, ips=FRAME_RATE):
        self.video_name = video_name
        self.ips = ips
        self.sampling_interval_ms = float(1 / self.ips) * 1000  # 40ms

    def start_sampling(self):
        image_id = 1
        cur_timestamp = datetime(year=2016, month=1, second=0, day=1)
        ts = cur_timestamp.strftime('%H:%M:%S.%f')[:-3]
        while self._sample_image(image_id, timestamp=ts):
            image_id += 1
            cur_timestamp += timedelta(milliseconds=self.sampling_interval_ms)
            ts = cur_timestamp.strftime('%H:%M:%S.%f')[:-3]

    def _sample_image(self, image_id, timestamp='00:03:06.016'):
        # ffmpeg -i 1_10.mov -ss 00:03:06.016 -vframes 1 out.png
        output_name = frames_dir() + 'output_%04d.png' % image_id
        cmd = 'ffmpeg -y -i {} -ss {} -vframes 1 {} > /dev/null 2>&1'.format(self.video_name, timestamp, output_name)
        print('-> {}'.format(cmd))
        os.system(cmd)
        return os.path.isfile(output_name)  # success or failure.

    def get_timestamp(self, image_id):
        return (image_id - 1) / self.ips


if __name__ == '__main__':
    assert len(sys.argv) == 2
    video_name = sys.argv[1]
    converter = Converter(video_name)
    converter.start_sampling()
