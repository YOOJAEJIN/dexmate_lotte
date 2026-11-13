"""KORAS(KR-GCS) 그리퍼 — Modbus RTU 드라이버 + 텔레옵 + 단독 테스트.

프로토콜 (실기 검증 2026-08-31 + Dexmate 참고 스크립트):
  RS485 Modbus RTU, 115200 8N1, slave 1
  명령: FC16 → 0x0000에 [명령코드 u16, 값 u16]
    101 = INIT(캘리브레이션, 최초 1회 필수) / 102 = 완전 열기 / 103 = 완전 닫기
    104 = 위치 이동 (값 0~1000, 0=닫힘 1000=열림) — 전 구간 ±1 정확 (실측)
    108/109 = 임피던스 제어 켬/끔 (다음 INIT 부터 적용) / 110 = [타입, 강성 1~10] (INIT 이후에 써야 유지)
             — Dexmate Koras_Impedance_manager.py(KR-RTS 매뉴얼 4.3·5.1.2) 기준, 실기 미검증
  상태: FC03 0x000A ×5 →
    [상태비트(b0 enable, b1 init중, b5 여는중, b6 닫는중, b9 fault),
     모터각도, 전류 mA, RPM(i16), finger_position 0~1000]

전송 경로 2종:
  KorasGripper   — USB-RS485 어댑터 직결 (CH340으로 충분 — 115200 저속)
  KorasEEGripper — dexcontrol 팔 EE 패스스루 (조인트7 뒤 손목 RS485 신호선.
                   XHand A안과 동일 경로 — 실제 로봇 장착 시 이쪽.
                   EE 타입이 UNKNOWN이어야 패스스루 활성)

사용 (단독 테스트):
  python3 koras_gripper.py --port /dev/ttyUSB0 --action demo          # USB 직결
  python3 koras_gripper.py --transport ee --side right --action demo  # 로봇 EE 장착
  python3 koras_gripper.py --port /dev/ttyUSB0 --action imp_on --stiffness 5   # 임피던스 지원 확인 (ACK/Modbus 예외)
텔레옵 통합: 컨버터 --ee right=koras (EE 패스스루) / --ee right=koras:/dev/ttyUSB0 (USB)
            좌우 다른 EE 혼용: --ee left=koras,right=xhand  (레거시 --koras right 도 동작)
"""

from pathlib import Path as _ModulePath
import sys as _module_sys
_module_sys.path.insert(0, str(_ModulePath(__file__).resolve().parents[3]))

import argparse
import os
import struct
import sys
import threading
import time

# USB-RS485 포트 준비(권한·FTDI latency 자동 보정) — src/ee/shared/rs485_port.py 공용
try:
    import os as _os, sys as _sys
    _sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "shared"))
    from rs485_port import prepare_serial_port
except Exception:                       # 헬퍼 없으면 그냥 연다
    def prepare_serial_port(port, *a, **k):
        return ""

try:
    import serial
except ImportError:          # EE 패스스루만 쓰는 환경이면 pyserial 없어도 됨
    serial = None

CMD_INIT, CMD_OPEN, CMD_CLOSE, CMD_GOTO = 101, 102, 103, 104
# 임피던스 제어 — 상태 레지스터에 켬/끔 비트가 없어 확인은 동작(순응 파지)으로만 가능
CMD_IMP_ON, CMD_IMP_OFF, CMD_IMP_PARAMS = 108, 109, 110
IMP_TYPES = {1: "2FPA-M-03-035/080", 2: "2FPA-M-03-035/080-CO", 3: "2FST-M-04-100-HS"}  # 매뉴얼 Table 4.5 — 틀리면 물체 손상 가능
_MODBUS_EXC = {1: "illegal function", 2: "illegal data address", 3: "illegal data value", 4: "slave device failure"}
POS_MAX = 1000
IMP_MARGIN = 30      # 임피던스 모드 목표 여유(카운트): 하드스톱 0/1000 을 목표로 주면 조가 닿지 못해 재보정 스윕 반복 (2026-09-28 실기)
BAUD = 115200


def _crc16(d: bytes) -> bytes:
    crc = 0xFFFF
    for b in d:
        crc ^= b
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return struct.pack("<H", crc)


class _KorasBase:
    """프로토콜 공통부 — 전송은 하위 클래스의 _xfer(요청, 대기초)→응답bytes."""

    slave = 1
    CMD_WAIT = 0.03
    STATUS_WAIT = 0.02
    last_error = ""          # 마지막 _cmd 실패 원인 ("응답 없음" / "Modbus 예외 N (...)") — 진단 출력용

    def _xfer(self, query: bytes, wait: float) -> bytes:
        raise NotImplementedError

    def _cmd(self, code: int, *values: int, retries: int = 1) -> bool:
        """FC16 → HR0.. 에 [code, 값1, 값2..] (값 없으면 [code, 0]). retries: RS485 유실 대비 (실측 28% 유실 개체)."""
        regs = (code, *(values or (0,)))
        q = struct.pack(">BBHHB", self.slave, 0x10, 0x0000, len(regs), len(regs) * 2)
        q += b"".join(struct.pack(">H", r) for r in regs)
        frame = q + _crc16(q)
        for _ in range(max(1, retries)):
            r = self._xfer(frame, self.CMD_WAIT)
            if len(r) >= 5 and r[1] == 0x10:
                return True
            if len(r) >= 5 and r[1] == 0x90:      # Modbus 예외 = 명령 거부 (미지원 코드 등) — 재시도 무의미
                self.last_error = f"Modbus 예외 {r[2]} ({_MODBUS_EXC.get(r[2], '?')})"
                return False
        self.last_error = "응답 없음"
        return False

    def status(self) -> dict | None:
        q = struct.pack(">BBHH", self.slave, 3, 0x000A, 5)
        r = self._xfer(q + _crc16(q), self.STATUS_WAIT)
        if len(r) < 5 or r[1] != 3:
            return None
        v = [struct.unpack(">H", r[3 + 2 * i:5 + 2 * i])[0] for i in range(r[2] // 2)]
        if len(v) < 5:
            return None
        vel = v[3] - 65536 if v[3] > 32767 else v[3]
        return {"enabled": bool(v[0] & 1), "initializing": bool(v[0] & 2),
                "opening": bool(v[0] & 32), "closing": bool(v[0] & 64),
                "fault": bool(v[0] & 512),
                "current_ma": v[2], "velocity_rpm": vel,
                "pos": v[4],                      # 0=닫힘 ~ 1000=열림
                "squeeze": 1.0 - v[4] / POS_MAX}  # 0=열림 ~ 1=닫힘

    def init(self, timeout: float = 8.0) -> bool:
        """캘리브레이션 (전원 인가 후 최초 1회). 조 사이 비울 것.

        유실 배선 대비: ①INIT 명령 3회 재시도 ②2차 시도 ③그래도 진행 관측이
        안 되면 실제 움직임 검증(goto 900)으로 최종 판정 — 이미 초기화된 개체가
        진행 플래그를 안 띄우는 경우도 커버."""
        for _ in range(2):
            self._cmd(CMD_INIT, retries=3)
            t0 = time.time()
            seen = False
            while time.time() - t0 < timeout / 2:
                s = self.status()
                if s and s["initializing"]:
                    seen = True
                elif s and seen and not s["initializing"]:
                    return not s["fault"]
                time.sleep(0.1)
        return self._verify_motion()

    def _verify_motion(self) -> bool:
        """goto 900이 실제로 반영되는지로 동작 가능 여부 판정 (끝나면 열어둠)."""
        self._cmd(CMD_GOTO, 900, retries=3)
        time.sleep(1.2)
        s = self.status() or self.status() or self.status()
        if s and abs(s["pos"] - 900) <= 60 and not s["fault"]:
            self._cmd(CMD_OPEN, retries=3)
            return True
        return False

    def open(self):
        self._cmd(CMD_OPEN)

    def close(self):
        self._cmd(CMD_CLOSE)

    def goto(self, pos: int):
        """위치 이동 (0=닫힘 ~ 1000=열림)."""
        self._cmd(CMD_GOTO, max(0, min(POS_MAX, int(pos))))

    def set_squeeze(self, squeeze: float):
        """텔레옵 계약: 0=열림 ~ 1=닫힘."""
        self.goto(round(POS_MAX * (1.0 - max(0.0, min(1.0, squeeze)))))

    def set_impedance(self, on: bool) -> bool:
        """108(켬)/109(끔) — 다음 INIT 부터 적용되므로 init() 전에 보낸다."""
        return self._cmd(CMD_IMP_ON if on else CMD_IMP_OFF, retries=3)

    def set_impedance_params(self, gtype: int, stiffness: int) -> bool:
        """110 [그리퍼 타입, 강성 1~10] — INIT 이후에 써야 스윕에 지워지지 않는다."""
        if gtype not in IMP_TYPES or not 1 <= stiffness <= 10:
            raise ValueError(f"임피던스 타입 {gtype} (1~3) / 강성 {stiffness} (1~10) 범위 밖")
        return self._cmd(CMD_IMP_PARAMS, gtype, stiffness, retries=3)

    def shutdown(self):
        pass


class KorasGripper(_KorasBase):
    """USB-RS485 직결 (노트북/thor USB에 어댑터)."""

    def __init__(self, port: str = "/dev/ttyUSB0", slave: int = 1):
        if serial is None:
            raise RuntimeError("pyserial 미설치 — USB 직결 불가 (EE 패스스루는 가능)")
        self.slave = slave
        # exclusive: 다른 프로세스(컨버터 텔레옵 ↔ 수동 CLI)가 같은 포트를 동시에 열지 못하게
        # (Linux TIOCEXCL) — 두 프로세스가 한 그리퍼를 두드려 응답이 뒤섞이는 사고 방지
        if "://" in port:                                # socket://thor:700x — thor rs485_tcp_bridge 경유
            self._ser = serial.serial_for_url(port, baudrate=BAUD, timeout=0.05)
        else:
            prepare_serial_port(port)                    # 권한/latency 매번 자동 점검·보정
            self._ser = serial.Serial(port, BAUD, timeout=0.05, exclusive=True)

    def _xfer(self, query: bytes, wait: float) -> bytes:
        self._ser.reset_input_buffer()
        self._ser.write(query)
        time.sleep(wait)
        return self._ser.read(64)

    def shutdown(self):
        try:
            self._ser.close()
        except Exception:
            pass


class KorasEEGripper(_KorasBase):
    """dexcontrol 팔 EE 패스스루 (손목 RS485) — XHand A안과 같은 경로.

    Modbus 프레임이 작아(≤64B) 분할 불필요. 단 브리지 전달 틱이 ~10ms라
    대기를 시리얼보다 길게 잡는다. 보드레이트 115200을 EE 포트에 설정."""

    CMD_WAIT = 0.08
    STATUS_WAIT = 0.06

    def __init__(self, arm, side: str, slave: int = 1):
        self.slave = slave
        self.side = side
        self._arm = arm
        self._last_ts = 0
        if not getattr(arm, "enable_ee_pass_through", False):
            raise RuntimeError(
                f"[{side}] EE 패스스루 비활성 — EE 타입이 UNKNOWN인지 확인"
                " (dexcontrol이 EE를 다른 타입으로 잡으면 패스스루가 닫힘)")
        cur = arm.get_ee_baud_rate()
        if not (cur.get("success") and cur.get("baud_rate") == BAUD):
            res = arm.set_ee_baud_rate(BAUD)
            if not res.get("success"):
                raise RuntimeError(
                    f"[{side}] EE 보드레이트 {BAUD} 설정 실패: {res.get('message')}")
            time.sleep(0.2)

    def _drain(self) -> bytes:
        buf = b""
        for _ in range(8):
            r = self._arm.get_ee_pass_through_response()
            if not r or r["timestamp_ns"] <= self._last_ts:
                break
            self._last_ts = r["timestamp_ns"]
            buf += bytes(r["data"])
        return buf

    def _xfer(self, query: bytes, wait: float) -> bytes:
        self._drain()                                    # 잔여 응답 버림
        self._arm.send_ee_pass_through_message(query)
        deadline = time.time() + max(wait, 0.05) + 0.15  # 브리지 틱 여유
        buf = b""
        while time.time() < deadline:
            time.sleep(0.02)
            buf += self._drain()
            if len(buf) >= 5:                            # 최소 응답 도착
                break
        return buf


def _say(msg: str):
    print(time.strftime("%H:%M:%S"), msg, flush=True)      # 브리지/컨버터 로그에서 INIT 시각을 볼 수 있게


class KorasTeleop:
    """스퀴즈(0~1) → KORAS 위치 비례 추종 스레드 (컨버터 --koras 통합).

    robotiq GripperTeleop과 같은 계약: start()/stop(),
    get_squeeze(side)→0~1, paused()→bool,
    on_command(side, 개폐도 0~1) / on_state(side, 개폐도 0~1) — 19차원 기록 형식 유지.
    RS485 왕복이 느려 12Hz 루프 + 변화 없으면(데드밴드) 명령 생략.
    임피던스 모드에서는 목표를 하드스톱(0/1000) 안쪽으로 IMP_MARGIN 만큼 제한한다 — 2026-09-28 실기: 임피던스 켬이면 조가 열림 1000 에
    닿지 못하고 980 에 머물다(실측 0.02) 약 20초 뒤 재보정 스윕(실측 0.99)이 반복됐고, 끔이면 1000 도달·정상. 도달 못 하는 목표를 밀지 않게 한다.

    grippers: {side: 그리퍼객체(_KorasBase) 또는 시리얼포트문자열}
    impedance: {side: (타입, 강성)} = 켬(108 → INIT → 110) / {side: None} = 끔(109 → INIT) / 키 없음 = 건드리지 않음"""

    def __init__(self, grippers: dict, get_squeeze, paused=lambda: False,
                 rate: float = 12.0, deadband: float = 0.01,
                 on_command=None, on_state=None, state_period: float = 0.3,
                 impedance: dict | None = None, release_on_stop: bool = True):
        self._spec = dict(grippers)
        self._impedance = dict(impedance or {})
        self._release_on_stop = release_on_stop      # False = 종료 시 마지막 상태 유지 (zenoh 브리지: 잡은 물체를 안 놓침)
        self._get_squeeze = get_squeeze
        self._paused = paused
        self._period = 1.0 / rate
        self._deadband = deadband
        self._on_command = on_command
        self._on_state = on_state
        self._state_period = state_period
        self._grippers = {}
        self._threads = []
        self._stop_ev = threading.Event()

    def start(self) -> bool:
        for side, spec in self._spec.items():
            try:
                g = KorasGripper(spec) if isinstance(spec, str) else spec
                st = g.status()
                if st is None:
                    _say(f"[koras:{side}] 응답 없음 — 이 쪽은 제외")
                    continue
                imp = self._impedance.get(side, False)   # False=건드리지 않음 / None=끔 / (타입, 강성)=켬
                if imp is not False:
                    ok = g.set_impedance(imp is not None)  # 108/109 — 다음 INIT 에서 적용
                    _say(f"[koras:{side}] 임피던스 {'켬(108)' if imp else '끔(109)'}: "
                          f"{'ACK' if ok else g.last_error + ' — 펌웨어 미지원?'}")
                # 전원 인가 후 INIT(캘리브레이션) 필수 — 조 사이 비어있어야 함
                _say(f"[koras:{side}] INIT(캘리브레이션) 시작 — 조 사이를 비워두세요")
                if not g.init():
                    _say(f"[koras:{side}] INIT 실패/타임아웃 — 제외")
                    continue
                if imp:
                    ok = g.set_impedance_params(*imp)      # 110 — INIT 이후에만 유지됨
                    _say(f"[koras:{side}] 임피던스 타입 {imp[0]}({IMP_TYPES[imp[0]]}) 강성 {imp[1]}/10: "
                          f"{'ACK' if ok else g.last_error}")
                if imp:
                    g._cmd(CMD_GOTO, self.target_pos(side, 0.0), retries=3)   # 102(하드스톱 열림) 대신 여유 열림
                else:
                    g._cmd(CMD_OPEN, retries=3)
                self._grippers[side] = g
            except Exception as e:
                _say(f"[koras:{side}] 초기화 실패: {e}")
        for side, g in self._grippers.items():
            t = threading.Thread(target=self._loop, args=(side, g), daemon=True)
            t.start()
            self._threads.append(t)
        return bool(self._grippers)

    def target_pos(self, side: str, squeeze: float) -> int:
        """스퀴즈(0=열림~1=닫힘) → 위치 카운트. 임피던스 켬인 쪽은 하드스톱 안쪽 [IMP_MARGIN, POS_MAX-IMP_MARGIN] 으로 제한."""
        pos = round(POS_MAX * (1.0 - max(0.0, min(1.0, squeeze))))
        if self._impedance.get(side):
            pos = max(IMP_MARGIN, min(POS_MAX - IMP_MARGIN, pos))
        return pos

    def _loop(self, side: str, g: _KorasBase):
        last_sq = -1.0
        last_state = 0.0
        last_cmd_t = 0.0
        while not self._stop_ev.is_set():
            t0 = time.time()
            try:
                if not self._paused():
                    sq = max(0.0, min(1.0, float(self._get_squeeze(side))))
                    # 변화가 있거나, 0.4초마다 목표 재전송 — RS485 유실(실측: 우측 28%)
                    # 시 명령이 사라져도 다음 재전송이 따라잡음
                    if (abs(sq - last_sq) > self._deadband
                            or time.time() - last_cmd_t > 0.4):
                        g.goto(self.target_pos(side, sq))
                        last_sq = sq
                        last_cmd_t = time.time()
                        if self._on_command:
                            self._on_command(side, sq)
                if time.time() - last_state >= self._state_period:
                    st = g.status()
                    if st is not None and self._on_state:
                        self._on_state(side, st["squeeze"])
                    last_state = time.time()
            except Exception:
                pass   # 일시 통신 오류는 다음 틱에 재시도
            dt = self._period - (time.time() - t0)
            if dt > 0:
                self._stop_ev.wait(dt)

    def stop(self):
        """스레드 정지 + 종료. release_on_stop=True(텔레옵 기본)면 그리퍼를 열고 끝냄, False 면 마지막 상태 유지."""
        self._stop_ev.set()
        for t in self._threads:
            t.join(timeout=2.0)
        for side, g in self._grippers.items():
            try:
                if self._release_on_stop:
                    g.open()
                g.shutdown()
            except Exception:
                pass


def _fmt_status(s: dict | None) -> str:
    """상태 dict → 한 줄 (웹UI/터미널 공용). None = 무응답."""
    if not s:
        return "응답 없음 — 전원/RS485 배선(A·B)/포트/slave ID 확인"
    flags = [k for k in ("enabled", "initializing", "opening", "closing", "fault") if s.get(k)]
    return (f"위치 {s['pos']}/1000 (개폐도 {s['squeeze']:.2f}) · 전류 {s['current_ma']}mA"
            f" · {s['velocity_rpm']}rpm · {'/'.join(flags) or 'idle'}")


def _run_actions(g: _KorasBase, a) -> int:
    """단독 동작 실행. 반환값 = 종료코드 (0 성공, 1 무응답/실패 — 웹UI가 ✓/✗ 판정에 사용)."""
    if a.action == "status":
        s = g.status()
        print(_fmt_status(s))
        return 0 if s else 1
    if a.action == "init":
        ok = g.init()
        print("INIT:", "완료" if ok else "실패/타임아웃 — 조 사이가 비었는지, 응답이 오는지 확인")
        print(_fmt_status(g.status()))
        return 0 if ok else 1
    if a.action in ("open", "close", "goto"):
        if a.action == "open":
            g.open()
        elif a.action == "close":
            g.close()
        else:
            g.goto(a.value)
        time.sleep(1.2)
        s = g.status()
        label = {"open": "열기", "close": "닫기", "goto": f"이동 {a.value}"}[a.action]
        print(f"{label} → {_fmt_status(s)}")
        return 0 if s else 1
    if a.action in ("imp_on", "imp_off"):
        # 임피던스 지원 확인: 108/109 ACK 여부 → INIT(적용) → 켬이면 110 [타입, 강성]. Modbus 예외 = 펌웨어 미지원
        on = a.action == "imp_on"
        ok = g.set_impedance(on)
        print(f"임피던스 {'켬(108)' if on else '끔(109)'}:", "ACK" if ok else g.last_error)
        print("INIT(적용):", "완료" if g.init() else "실패/타임아웃")
        if on:
            ok2 = g.set_impedance_params(a.gripper_type, a.stiffness)
            print(f"파라미터 110 [타입 {a.gripper_type}, 강성 {a.stiffness}]:", "ACK" if ok2 else g.last_error)
            ok = ok and ok2
        print(_fmt_status(g.status()))
        return 0 if ok else 1
    # demo: init → 사분위 왕복
    print("INIT:", g.init())
    for p in (0, 500, 1000, 250, 750, 1000):
        g.goto(p)
        time.sleep(1.2)
        s = g.status()
        if s:
            print(f"목표 {p:4d} → 실측 {s['pos']:4d} (전류 {s['current_ma']}mA)")
        else:
            print(f"목표 {p:4d} → 상태 응답 없음")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--transport", choices=["usb", "ee"], default="usb",
                    help="usb=USB-RS485 직결 / ee=팔 EE 패스스루 (로봇 장착 시)")
    ap.add_argument("--port", default="/dev/ttyUSB0", help="usb 전송용")
    ap.add_argument("--side", choices=["left", "right"], default="right",
                    help="ee 전송용 — 그리퍼 달린 팔")
    ap.add_argument("--action", default="demo",
                    choices=["init", "open", "close", "goto", "status", "demo", "imp_on", "imp_off"])
    ap.add_argument("--value", type=int, default=500, help="goto 목표 (0~1000)")
    ap.add_argument("--stiffness", type=int, default=5, help="imp_on 강성 1~10")
    ap.add_argument("--gripper-type", type=int, default=1, choices=sorted(IMP_TYPES),
                    help="imp_on 그리퍼 타입 — 매뉴얼 Table 4.5: "
                         + ", ".join(f"{k}={v}" for k, v in IMP_TYPES.items()) + " (틀리면 물체 손상 가능)")
    a = ap.parse_args()

    if a.transport == "usb":
        try:
            g = KorasGripper(a.port)
        except Exception as e:      # 포트 없음/권한(dialout)/사용 중 등
            print(f"포트 열기 실패 {a.port}: {e}")
            return 1
        try:
            return _run_actions(g, a)
        finally:
            g.shutdown()

    # EE 패스스루 — dexcontrol Robot 필요 (env: ROBOT_NAME/ROBOT_CONFIG/ZENOH_CONFIG)
    os.environ.setdefault("DEXCONTROL_DISABLE_HAND_QUERY", "1")
    os.environ.setdefault("DEXCONTROL_DISABLE_HEARTBEAT", "1")
    _ee_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    from common.dex_connection import robot_without_head

    with robot_without_head() as bot:   # 헤드를 안 건드림
        arm = getattr(bot, f"{a.side}_arm")
        return _run_actions(KorasEEGripper(arm, a.side), a)


if __name__ == "__main__":
    sys.exit(main())
