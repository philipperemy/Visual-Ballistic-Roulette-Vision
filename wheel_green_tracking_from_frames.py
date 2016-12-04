# import the necessary packages
import os
from collections import deque

import cv2
import dill
import imutils
import numpy as np

from utils import FRAMES_DIR, FrameIterator, frames_to_seconds


def extract_lap_frames(results):
    x_range = np.array([r[0][0] for r in results])
    indicators = np.array(x_range > np.percentile(x_range, 95), dtype=int)
    buffer = []
    frame_ids = []
    for i, res in enumerate(results):
        frame_id = res[1]
        indicator = indicators[i]
        if indicator == 0:
            if len(buffer) != 0:
                frame_ids.append(int(np.median(buffer)))
            buffer = []  # clear buffer.
        elif indicator == 1:
            buffer.append(frame_id)
    frames_ids_of_interest = np.array(
        [x for x in np.array(np.array([0] + list(np.diff(frame_ids))) > 20, dtype=bool) * frame_ids if x > 0])
    return [r for r in results if r[1] in frames_ids_of_interest]


def analyze_video():
    results = []
    # define the lower and upper boundaries of the "green"
    # ball in the HSV color space, then initialize the
    # list of tracked points
    green_lower = (29, 86, 6)
    green_upper = (64, 255, 255)
    pts = deque(maxlen=64)

    frames_iterator = FrameIterator(FRAMES_DIR)
    for i, (frame, name) in enumerate(frames_iterator.read_frames()):
        print(name)
        # resize the frame, blur it, and convert it to the HSV
        # color space
        frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # construct a mask for the color "green", then perform
        # a series of dilations and erosions to remove any small
        # blobs left in the mask
        mask = cv2.inRange(hsv, green_lower, green_upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # cv2.imshow("Mask", mask)

        # find contours in the mask and initialize the current
        # (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]
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
            if radius > 5:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                print(center, i)
                results.append((center, i))
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


def start_wheel_analysis():
    wheel_tracking_file = 'w_res.pkl'
    if os.path.isfile(wheel_tracking_file):
        r = dill.load(open(wheel_tracking_file, 'rb'))
    else:
        r = analyze_video()
        dill.dump(r, open(wheel_tracking_file, 'wb'))
    a = extract_lap_frames(r)
    frames_seconds = frames_to_seconds(np.array([c[1] for c in a]))
    print(frames_seconds)
    return frames_seconds


if __name__ == '__main__':
    start_wheel_analysis()
