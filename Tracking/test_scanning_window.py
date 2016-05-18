import cv2
from detection import *
from structure import *
from time import time
import numpy as np

def get_fps(time1, time2):
    fps = int(100 / (time2 - time1)) / 100.0
    if fps < 25:
        color = (0,0,255)
    else:
        color = (0,255,0)
    return fps, color

def get_frame(path, area=320*240, dsize = None, fx_size = None, fy_size = None):
    if path is not None:
        frame = cv2.imread(path)
        if area is not None and area != 0:
            f_size = np.sqrt(1.0*area/(frame.shape[0]*frame.shape[1]))
            frame = cv2.resize(frame, None, fx=f_size, fy=f_size)
        elif dsize is not None:
            frame = cv2.resize(frame, dsize)
        elif fx_size is not None and fy_size is not None:
            frame = cv2.resize(frame, None, fx=fx_size, fy=fy_size)
        frame = cv2.flip(frame, 1)
        return frame
    else:
        return None

frame = get_frame('test.jpg')
cv2.imshow("w",frame)
print "Shape:", frame.shape

# scales_step = 1.15
# slip_step = 0.01
# minimal_bounding_box_size = 40
# min_step = 2
# max_step = 10
# for size in [minimal_bounding_box_size*i for i in xrange(1,min(frame.shape[0],frame.shape[1])/minimal_bounding_box_size)]:
#     print "Square bounding box size:", size
#     position = Position(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 0, size, size)
#
#     windows = []
#     for current_position in scanning_window(position, scales_step, slip_step, minimal_bounding_box_size, min_step, max_step):
#         windows.append((current_position.x, current_position.y, current_position.width, current_position.height))
#     print "Windows count", len(windows)
#
#     positions = []
#     for current_position in scanning_window(position, scales_step, slip_step, minimal_bounding_box_size):
#         pos = copy(current_position)
#         pos.frame = None
#         positions.append(pos)
#
#     print "Scanning window"
#     start = time()
#     for current_position in scanning_window(position, scales_step, slip_step, minimal_bounding_box_size):
#         pass
#     print "Time:", time()-start
#     print "Fps:", get_fps(start,time())[0]
#     print
#
#     print "For windows"
#     start = time()
#     for window in windows:
#         current_position = Position(frame,*window)
#     print "Time:", time()-start
#     print "Fps:", get_fps(start,time())[0]
#     print
#
#     print "For positions"
#     start = time()
#     for current_position in positions:
#         current_position.update(frame=frame)
#     print "Time:", time()-start
#     print "Fps:", get_fps(start,time())[0]
#     print "------------------------------"

minimal_bounding_box_size = 40
min_step = 2
max_step = 10
size = 100
print " & ".join([str(elem) for elem in [0.02,0.04,0.06,0.08,0.1]]), "\\\\\hline"
for scales_step in [1.10,1.20,1.30,1.40,1.50]:
    counts = []
    times = []
    for slip_step in [0.02,0.04,0.06,0.08,0.1]:
        position = Position(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 0, 0, size, size)
        positions = []
        for current_position in scanning_window(position, scales_step, slip_step, minimal_bounding_box_size):
            pos = copy(current_position)
            pos.frame = None
            positions.append(pos)

        start = time()
        for current_position in positions:
            current_position.update(frame=frame)
        counts.append(len(positions))
        times.append(time()-start)

    print "\\textbf{%0.1f} &" % scales_step," & ".join(["%0.4f"% elem for elem in times]), "\\\\\hline"
