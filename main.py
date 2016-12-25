import os

from ball_tracking_from_gradients import start_ball_analysis
from utils import results_dir
from wheel_green_tracking_from_frames import start_wheel_analysis

if __name__ == '__main__':
    print('Python script has started. Please wait.')
    balls = start_ball_analysis()
    wheels = start_wheel_analysis()
    print('\n -- \n')
    print('BALL = {}'.format(balls))
    print('WHEEL = {}'.format(wheels))

    results_filename = os.path.join(results_dir(), 'results.txt')
    with open(results_filename, 'wt', encoding='utf-8') as f:
        f.write(str(list(balls)))
        f.write('\n')
        f.write(str(list(wheels)))
        f.write('\n')
