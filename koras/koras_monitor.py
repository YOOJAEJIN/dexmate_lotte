"""KORAS 그리퍼 상태 모니터 (read-only) — 위치/상태/부하 실시간 출력.

  python3 koras_monitor.py [--port /dev/ttyUSB0] [--sec 30]
"""
import argparse, struct, sys, time
import serial

# USB-RS485 포트 준비(권한·FTDI latency 자동 보정) — src/ee/shared/rs485_port.py 공용
try:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "shared"))
    from rs485_port import prepare_serial_port
except Exception:                       # 헬퍼 없으면 그냥 연다
    def prepare_serial_port(port, *a, **k):
        return ""

def crc16(d):
    crc = 0xFFFF
    for b in d:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return struct.pack("<H", crc)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--sec", type=float, default=30.0)
    ap.add_argument("--slave", type=int, default=1)
    a = ap.parse_args()
    prepare_serial_port(a.port)
    ser = serial.Serial(a.port, 115200, timeout=0.05)
    t0 = time.time()
    while time.time() - t0 < a.sec:
        q = struct.pack(">BBHH", a.slave, 3, 0x000A, 5); q += crc16(q)
        ser.reset_input_buffer(); ser.write(q); time.sleep(0.02)
        r = ser.read(64)
        if len(r) >= 5 and r[1] == 3:
            v = [struct.unpack(">H", r[3+2*i:5+2*i])[0] for i in range(r[2]//2)]
            st, vel, load, _, pos = v
            vel = vel - 65536 if vel > 32767 else vel
            sys.stdout.write(f"\r상태={st} 위치={pos:4d}/1000 (개폐도 {pos/1000:.2f}) "
                             f"속도={vel:+5d} 부하={load:4d}   ")
            sys.stdout.flush()
        time.sleep(0.1)
    print()
    return 0

if __name__ == "__main__":
    sys.exit(main())
