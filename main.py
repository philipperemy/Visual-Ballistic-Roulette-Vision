from ball_tracking_from_gradients import start_ball_analysis
from wheel_green_tracking_from_frames import start_wheel_analysis

if __name__ == '__main__':
    print('Python script has started. Please wait.')
    balls = start_ball_analysis()
    wheels = start_wheel_analysis()
    print('\n -- \n')
    print('BALL = {}'.format(balls))
    print('WHEEL = {}'.format(wheels))
