import pyzed.sl as sl
import cv2
import subprocess
import time

THOR_IP = "192.168.50.20"

# ZED X Mini
PORT_LEFT = 50001
PORT_RIGHT = 50002
PORT_DEPTH = 50003

# # ZED One GS
PORT_LEFT_WRIST = 50004
PORT_RIGHT_WRIST = 50005

WIDTH = 640
HEIGHT = 480
FPS = 30


class GstSender:

    def __init__(self, port):

        gst_cmd = (
            f'gst-launch-1.0 fdsrc ! '
            f'rawvideoparse width={WIDTH} height={HEIGHT} format=bgra framerate={FPS}/1 ! '
            f'videoconvert ! '
            f'x264enc tune=zerolatency bitrate=4000 speed-preset=ultrafast key-int-max=30 ! '
            f'rtph264pay pt=96 config-interval=1 ! '
            f'udpsink host={THOR_IP} port={port} sync=false async=false'
        )

        print(gst_cmd)

        self.proc = subprocess.Popen(
            gst_cmd,
            shell=True,
            stdin=subprocess.PIPE
        )

    def send(self, frame):

        try:
            self.proc.stdin.write(frame.tobytes())
        except:
            pass


############################################################
# ZED X MINI
############################################################

zed = sl.Camera()

init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD1200
init.camera_fps = FPS
init.depth_mode = sl.DEPTH_MODE.NEURAL

status = zed.open(init)

if status != sl.ERROR_CODE.SUCCESS:
    print("ZED X MINI OPEN FAILED")
    print(status)
    exit(1)

runtime = sl.RuntimeParameters()

left_mat = sl.Mat()
right_mat = sl.Mat()
depth_mat = sl.Mat()

sender_left = GstSender(PORT_LEFT)
sender_right = GstSender(PORT_RIGHT)
sender_depth = GstSender(PORT_DEPTH)

print("ZED X MINI STREAMING STARTED")


############################################################
# METRIC DEPTH (uint16 mm, TCP) for Thor seg_overlay
# The VIEW.DEPTH stream above is a display rendering whose scale shifts
# with the scene; this channel carries retrieve_measure(DEPTH) values.
# Frame: struct ">HHI" (w, h, nbytes) + little-endian uint16 mm, 0 = invalid.
############################################################
import socket
import struct
import threading
import numpy as np

METRIC_DEPTH_PORT = 55703
METRIC_DEPTH_EVERY = 3  # every Nth grab (~10 Hz at 30 fps)
measure_mat = sl.Mat()
_unit_to_mm = {sl.UNIT.MILLIMETER: 1.0, sl.UNIT.CENTIMETER: 10.0,
               sl.UNIT.METER: 1000.0}.get(init.coordinate_units, 1.0)
_latest_depth = {"buf": None}
_depth_cv = threading.Condition()


def _depth_client(conn):
    # One thread per client: a slow client never stalls the grab loop.
    with conn:
        last = None
        while True:
            with _depth_cv:
                _depth_cv.wait_for(lambda: _latest_depth["buf"] is not last, timeout=5)
                buf = _latest_depth["buf"]
            if buf is None or buf is last:
                continue
            last = buf
            try:
                conn.sendall(buf)
            except OSError:
                return


def _depth_server():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", METRIC_DEPTH_PORT))
    srv.listen(4)
    while True:
        conn, _ = srv.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        threading.Thread(target=_depth_client, args=(conn,), daemon=True).start()


threading.Thread(target=_depth_server, daemon=True).start()
print(f"METRIC DEPTH TCP SERVER :{METRIC_DEPTH_PORT} (unit->mm x{_unit_to_mm})")
_frame_i = 0


# ############################################################
# # ZED ONE GS #1
# ############################################################

# zed_one_left = sl.CameraOne()

# init_left = sl.InitParametersOne()
# init_left.camera_resolution = sl.RESOLUTION.HD1200
# init_left.camera_fps = FPS

# # 추가
# init_left.async_grab_camera_recovery = False

# status = zed_one_left.open(init_left)

# # if status != sl.ERROR_CODE.SUCCESS:
# #     print("LEFT WRIST OPEN FAILED")
# #     print(status)
# #     exit(1)

# left_wrist_mat = sl.Mat()

# sender_left_wrist = GstSender(PORT_LEFT_WRIST)

# print("LEFT WRIST STREAMING STARTED")


# ############################################################
# # ZED ONE GS #2
# ############################################################

# zed_one_right = sl.CameraOne()

# init_right = sl.InitParametersOne()
# init_right.camera_resolution = sl.RESOLUTION.HD1200
# init_right.camera_fps = FPS

# # 추가
# init_right.async_grab_camera_recovery = False

# # 두 번째 카메라 선택
# devices = sl.CameraOne.get_device_list()

# if len(devices) >= 2:
#     init_right.set_from_serial_number(devices[1].serial_number)

# status = zed_one_right.open(init_right)

# # if status != sl.ERROR_CODE.SUCCESS:
# #     print("RIGHT WRIST OPEN FAILED")
# #     print(status)
# #     exit(1)

# right_wrist_mat = sl.Mat()

# sender_right_wrist = GstSender(PORT_RIGHT_WRIST)

# print("RIGHT WRIST STREAMING STARTED")


############################################################
# MAIN LOOP
############################################################

while True:

    ########################################################
    # ZED X MINI
    ########################################################

    if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:

        zed.retrieve_image(left_mat, sl.VIEW.LEFT)
        zed.retrieve_image(right_mat, sl.VIEW.RIGHT)
        zed.retrieve_image(depth_mat, sl.VIEW.DEPTH)

        left_frame = left_mat.get_data()
        right_frame = right_mat.get_data()
        depth_frame = depth_mat.get_data()

        left_frame = cv2.resize(left_frame, (WIDTH, HEIGHT))
        right_frame = cv2.resize(right_frame, (WIDTH, HEIGHT))
        depth_frame = cv2.resize(depth_frame, (WIDTH, HEIGHT))

        sender_left.send(left_frame)
        sender_right.send(right_frame)
        sender_depth.send(depth_frame)

        _frame_i += 1
        if _frame_i % METRIC_DEPTH_EVERY == 0:
            zed.retrieve_measure(measure_mat, sl.MEASURE.DEPTH)
            # NEAREST: never blend depth across object edges
            d = cv2.resize(measure_mat.get_data(), (WIDTH, HEIGHT),
                           interpolation=cv2.INTER_NEAREST)
            d = np.nan_to_num(d * _unit_to_mm, nan=0.0, posinf=0.0, neginf=0.0)
            mm = np.clip(d, 0, 65535).astype("<u2")
            payload = struct.pack(">HHI", WIDTH, HEIGHT, mm.nbytes) + mm.tobytes()
            with _depth_cv:
                _latest_depth["buf"] = payload
                _depth_cv.notify_all()

    # ########################################################
    # # LEFT WRIST
    # ########################################################

    # if zed_one_left.grab() == sl.ERROR_CODE.SUCCESS:

    #    status = zed_one_left.retrieve_image(
    #        left_wrist_mat,
    #        sl.VIEW.LEFT_UNRECTIFIED
    #    )

    #    if status == sl.ERROR_CODE.SUCCESS:

    #        left_wrist_frame = left_wrist_mat.get_data()

    #        if left_wrist_frame is not None and left_wrist_frame.size != 0:

    #            left_wrist_frame = cv2.resize(
    #                left_wrist_frame,
    #                (WIDTH, HEIGHT)
    #            )

    #            sender_left_wrist.send(left_wrist_frame)

    #        else:
    #            print("Empty frame received")

    #    else:
    #        print(f"retrieve_image failed: {status}")

    # ########################################################
    # # RIGHT WRIST
    # ########################################################

    # if zed_one_right.grab() == sl.ERROR_CODE.SUCCESS:

    #     status = zed_one_right.retrieve_image(
    #         right_wrist_mat,
    #         sl.VIEW.LEFT_UNRECTIFIED
    #     )

    #     if status == sl.ERROR_CODE.SUCCESS:

    #         right_wrist_frame = right_wrist_mat.get_data()

    #         if right_wrist_frame is not None and right_wrist_frame.size != 0:

    #             right_wrist_frame = cv2.resize(
    #                 right_wrist_frame,
    #                 (WIDTH, HEIGHT)
    #             )

    #             sender_right_wrist.send(right_wrist_frame)

    #         else:
    #             print("Empty RIGHT frame received")

    #     else:
    #         print(f"RIGHT retrieve_image failed: {status}")