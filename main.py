from ball_tracking_from_gradients import start_ball_analysis
from wheel_green_tracking_from_frames import start_wheel_analysis

if __name__ == '__main__':
    print('BALL = {}'.format(start_ball_analysis()))
    print('WHEEL = {}'.format(start_wheel_analysis()))
