from subprocess import call


def call_program(video=None):
    if video is None:
        video = 'videos/1_10.mov'
    call(['./run.sh', video])


if __name__ == '__main__':
    call_program()
