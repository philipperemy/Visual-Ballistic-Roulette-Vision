import os
from collections import deque
from glob import glob
from pprint import pprint

import cv2
import dill
import numpy as np

from hyperparameters import *
from utils import FrameIterator, cropped_gradients_dir, frames_to_seconds, crop_gradients, tmp_dir


def bucket_analysis(buckets):
    keep_buckets = []
    print('{} buckets to analyze'.format(len(buckets)))
    for bucket in buckets:
        # (dim1, dim2) = (row, col)
        the_one_the_most_at_the_right_idx = np.argmax([t[0][0] for t in bucket])
        keep_buckets.append(bucket[the_one_the_most_at_the_right_idx])
    return keep_buckets


# because of the B-FRM and P-FRM we have to link the gaps by exactly one.
# actually on some images, the gradients will indicate the motion of the ball and will disappear for one image because
# we swing between B-FRM and P-FRM.
# So the goal is to make NUMBER_OF_CONSECUTIVE_FRAMES_TO_KEEP_A_BALL_BUCKET = 1
# and fill those gaps.


def fill_b_p_frm_gaps(results):
    updated_results = list()
    updated_results.append(results[0])
    for i in range(1, len(results)):
        frame_1 = results[i - 1]
        frame_2 = results[i]
        if frame_1[1] + 2 == frame_2[1]:
            print('B-FRM P-FRM gap between frame {} and {}'.format(frame_1[1], frame_2[1]))
            # check for gaps here. We don't care where is the ball exactly. Therefore, frame_1[0]
            updated_results.append((frame_1[0], frame_1[1] + 1))
        updated_results.append(frame_2)
    return updated_results


def bucket_frames(results):
    results = fill_b_p_frm_gaps(results)
    frames_results = np.array([r[1] for r in results])
    correct_ids = np.where(np.array([0] + list(np.diff(frames_results))) != 1)[0]
    # frames_results[np.where(np.array([0] + list(np.diff(frames_results))) != 1)[0]]
    return [results[i] for i in correct_ids]


def analyze_video():
    # B max 177 min 147
    # G max 195 min 150
    # R max 227 min 172

    # B G R
    white_lower = (10, 10, 10)
    white_upper = (255, 255, 255)
    pts = deque(maxlen=64)
    results = []

    frame_iterator = FrameIterator(cropped_gradients_dir())
    for (frame, name) in frame_iterator.read_frames():
        frame_id = int(name.split('_')[1].split('.')[0])
        mask = cv2.inRange(frame, white_lower, white_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cv2.imshow("Mask", mask)

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        center = None

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use
            # it to compute the minimum enclosing circle and
            # centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            # only proceed if the radius meets a minimum size
            if radius > MINIMUM_PIXELS_BALL_RADIUS:  # 10 seems good.
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                results.append((center, frame_id))
                print(center, frame_id, radius)

        # update the points queue
        pts.appendleft(center)
        # loop over the set of tracked points
        for j in range(1, len(pts)):
            # if either of the tracked points are None, ignore them
            if pts[j - 1] is None or pts[j] is None:
                continue
            # otherwise, compute the thickness of the line and draw the connecting lines
            thickness = int(np.sqrt(64 / float(j + 1)) * 2.5)
            cv2.line(frame, pts[j - 1], pts[j], (0, 0, 255), thickness)

        # show the frame to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the 'q' key is pressed, stop the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    return results


def start_ball_analysis():
    ball_track_file = os.path.join(tmp_dir(), 'b_res.pkl')

    if len(glob(cropped_gradients_dir() + '*.png')) <= 0:
        crop_gradients()

    if os.path.isfile(ball_track_file):
        r = dill.load(open(ball_track_file, 'rb'))
    else:
        r = analyze_video()
        dill.dump(r, open(ball_track_file, 'wb'))
    pprint(r)
    print('\n ---  \n')
    b = bucket_frames(r)
    # a = bucket_analysis(b)
    pprint(b)
    # pprint(a)
    # pprint(len(a))
    frames_seconds = frames_to_seconds(np.array([c[1] for c in b]))
    print(frames_seconds)
    print(np.diff(frames_seconds))
    return frames_seconds


if __name__ == '__main__':
    start_ball_analysis()
