import re
from glob import glob

import numpy as np

from natural_sort import natural_keys


def read_experimentation_results():
    output = []
    results = glob('output/**/videos/results/*.txt')
    results.sort(key=natural_keys)
    for result in results:
        with open(result, 'r') as r:
            lines = r.readlines()
            assert len(lines) == 2
            video_id = result.split('/')[1]
            ball_lap_times = [float(v) for v in lines[0].strip().split(',')]
            wheel_lap_times = [float(v) for v in lines[1].strip().split(',')]
            # print('VIDEO ID =', video_id)
            # print('BALL =', ball_lap_times)
            # print('WHEEL =', wheel_lap_times)
            output.append({'video_id': video_id,
                           'ball_lap_times': ball_lap_times,
                           'wheel_lap_times': wheel_lap_times})
    return output


def read_expected_results():
    variable_regex = '^real_BALL_[0-9]'
    output = []
    with open('roulette.R', 'r') as r:
        lines = r.readlines()
        for line in lines:
            if re.match(variable_regex, line):
                video_id = line.split('=')[0].split('_')[-1].strip()
                if '#' in line:
                    line = line[:line.index('#')]
                ball_lap_times = [float(v) for v in line.split('c(')[1].strip()[:-1].split(',')]
                output.append({'video_id': video_id,
                               'ball_lap_times': ball_lap_times
                               })
    return output


def assert_equals(v1, v2):
    assert v1 == v2, 'v1 = {}, v2 = {}'.format(v1, v2)


def loss(predictions, expectations):
    p = np.flip(predictions, axis=0)
    a = np.flip(expectations, axis=0)
    err = np.mean(np.abs(np.subtract(p[:min(len(a), len(p))], a[:min(len(a), len(p))])))
    return err


def print_games(predicted, expected, game_ids, label):
    for p, a in zip(predicted, expected):
        game_id = int(p['video_id'])
        if game_id in game_ids:
            print(game_id, label)
            ball_p = p['ball_lap_times']
            ball_a = a['ball_lap_times']
            print('PRED =', ball_p, 'LEN =', len(ball_p))
            print('ACTU =', ball_a, 'LEN =', len(ball_a))
            print()


def run():
    predicted = read_experimentation_results()
    expected = read_expected_results()

    assert_equals(len(predicted), len(expected))
    losses = []
    for p, a in zip(predicted, expected):
        assert_equals(p['video_id'], a['video_id'])
        ball_p = p['ball_lap_times']
        ball_a = a['ball_lap_times']
        print('PRED =', ball_p, 'LEN = ', len(ball_p))
        print('ACTU =', ball_a, 'LEN = ', len(ball_a))
        losses.append(loss(ball_p, ball_a))
    print(',\n'.join(['{0:.2f}'.format(i) for i in losses]))
    valid_ids = [int(predicted[i]['video_id']) for i in np.where(np.array(losses) < 0.1)[0]]
    invalid_ids = sorted(set(range(1, len(predicted))) - set(valid_ids))

    print('Valid games =', valid_ids)
    print('Invalid games =', invalid_ids)

    print()
    print_games(predicted, expected, valid_ids, 'VALID')
    print_games(predicted, expected, invalid_ids, 'INVALID')

    print('accuracy = {0:.3f}'.format(len(valid_ids) / len(valid_ids + invalid_ids)))


if __name__ == '__main__':
    run()
