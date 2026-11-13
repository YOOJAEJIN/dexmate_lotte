"""KORAS RS485 링크 실시간 테스트 — 배선 만지면서 응답률을 눈으로 확인.

초당 ~4회 상태 질의를 보내고 성공(O)/실패(·)를 한 글자씩 출력,
최근 40회 성공률과 finger 위치를 계속 갱신한다. Ctrl+C로 종료.

사용 (로봇 켜고, 컨버터/웹UI는 끈 상태에서):
  source ~/physicalworks_edge/venv/bin/activate && source /opt/ros/jazzy/setup.bash
  export ROS_DOMAIN_ID=10 ROBOT_NAME=dm/<시리얼> ROBOT_CONFIG=vega_1p
  export ZENOH_CONFIG=$HOME/.dexmate/comm/zenoh/<파일>.dzcfg
  python3 ~/physicalworks_edge/src/ee/gripper/koras/koras_link_test.py --side right

판정 기준: 95% 이상이면 INIT/텔레옵 안정. 60~80%에서 오르내리면 접촉 불량
(접점을 살짝 흔들어 보며 %가 출렁이는 지점 = 범인).
"""

from pathlib import Path as _ModulePath
import sys as _module_sys
_module_sys.path.insert(0, str(_ModulePath(__file__).resolve().parents[3]))

import argparse
import os
import struct
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_ee_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from koras_gripper import _crc16  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--side", choices=["left", "right", "both"], default="right")
    ap.add_argument("--rate", type=float, default=4.0, help="질의 Hz (기본 4)")
    a = ap.parse_args()

    os.environ.setdefault("DEXCONTROL_DISABLE_HAND_QUERY", "1")
    os.environ.setdefault("DEXCONTROL_DISABLE_HEARTBEAT", "1")
    from common.dex_connection import robot_without_head

    q = struct.pack(">BBHH", 1, 3, 0x000A, 5)
    frame = q + _crc16(q)
    sides = ["left", "right"] if a.side == "both" else [a.side]
    period = 1.0 / a.rate

    with robot_without_head() as bot:
        arms = {s: getattr(bot, f"{s}_arm") for s in sides}
        last_ts = {s: 0 for s in sides}
        hist = {s: [] for s in sides}       # 최근 성공(1)/실패(0)
        total = {s: [0, 0] for s in sides}  # [성공, 시도]
        pos = {s: "?" for s in sides}
        print(f">>> {'/'.join(sides)} 링크 테스트 — O=응답 ·=유실, Ctrl+C 종료", flush=True)
        try:
            n = 0
            while True:
                t0 = time.time()
                mark = {}
                for s, arm in arms.items():
                    arm.send_ee_pass_through_message(frame)
                for s, arm in arms.items():
                    got = False
                    while time.time() - t0 < period * 0.9:
                        r = arm.get_ee_pass_through_response()
                        if r and r["timestamp_ns"] > last_ts[s]:
                            last_ts[s] = r["timestamp_ns"]
                            data = bytes(r["data"])
                            if len(data) >= 13 and data[1] == 3:
                                pos[s] = struct.unpack(">H", data[11:13])[0]
                                got = True
                                break
                        time.sleep(0.005)
                    hist[s].append(1 if got else 0)
                    hist[s] = hist[s][-40:]
                    total[s][0] += got
                    total[s][1] += 1
                    mark[s] = "O" if got else "·"
                n += 1
                line = "".join(mark[s] for s in sides)
                if n % 20 == 0:
                    stats = " | ".join(
                        f"{s}: 최근 {sum(hist[s])*100//max(1,len(hist[s]))}%"
                        f" (누적 {total[s][0]}/{total[s][1]}) pos={pos[s]}"
                        for s in sides)
                    print(f" {line}   {stats}", flush=True)
                else:
                    print(line, end="", flush=True)
                dt = period - (time.time() - t0)
                if dt > 0:
                    time.sleep(dt)
        except KeyboardInterrupt:
            print()
            for s in sides:
                r = total[s]
                print(f">>> {s}: 총 {r[0]}/{r[1]} ({r[0]*100//max(1,r[1])}%)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
