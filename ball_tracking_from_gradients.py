import os
from collections import deque
from pprint import pprint

import cv2
import dill
import numpy as np

from utils import FrameIterator, cropped_gradients_dir, frames_to_seconds, crop_gradients, tmp_dir


def bucket_analysis(buckets):
    keep_buckets = []
    print('{} buckets to analyze'.format(len(buckets)))
    for bucket in buckets:
        # (dim1, dim2) = (row, col)
        the_one_the_most_at_the_right_idx = np.argmax([t[0][0] for t in bucket])
        keep_buckets.append(bucket[the_one_the_most_at_the_right_idx])
    return keep_buckets


def bucket_frames(results):
    # complexity MUST NOT BE QUADRATIC.
    # ball must at least be there for two consecutive frames.
    valid_frames = []
    last_known_result = (None, -1)
    bucket = []
    for result in results:
        cur_frame = result[1]
        if cur_frame <= last_known_result[1] + 3:
            if len(bucket) == 0:
                # push the former element.
                bucket.append(last_known_result)
            bucket.append(result)  # start filling the bucket.
        else:
            if len(bucket) > 0:
                valid_frames.append(bucket.copy())  # push the bucket.
            bucket = []  # reset the bucket.
        last_known_result = result
    if len(bucket) > 0:
        valid_frames.append(bucket.copy())  # push the bucket.
    valid_frames = [f for f in valid_frames if f[0][0] is not None]
    return valid_frames


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
            if radius > 1:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (255, 255, 0), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                results.append((center, frame_id))
                print(center, frame_id)

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
    if os.path.isfile(ball_track_file):
        r = dill.load(open(ball_track_file, 'rb'))
    else:
        crop_gradients()
        r = analyze_video()
        dill.dump(r, open(ball_track_file, 'wb'))
    pprint(r)
    print('\n ---  \n')
    b = bucket_frames(r)
    a = bucket_analysis(b)
    pprint(b)
    pprint(a)
    pprint(len(a))
    frames_seconds = frames_to_seconds(np.array([c[1] for c in a]))
    print(frames_seconds)
    print(np.diff(frames_seconds))
    return frames_seconds


if __name__ == '__main__':
    start_ball_analysis()
