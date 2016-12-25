from glob import glob
from subprocess import check_output

from natural_sort import natural_keys


def call_program(v=None):
    if v is None:
        v = 'videos/1_10.mov'
    out = check_output(['./run.sh', v])
    output_filename = v + '.txt'
    with open(output_filename, 'wt', encoding='utf-8') as w:
        print('-> {}'.format(output_filename))
        w.write(out.decode('utf-8'))


if __name__ == '__main__':
    videos = glob('videos/video_dec_24_deutsche_bordeaux/*.mp4')
    videos.sort(key=natural_keys)
    for video in videos:
        call_program(video)
