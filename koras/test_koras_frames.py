"""_cmd 프레임이 Dexmate Koras_Impedance_manager.py 의 koras_command_message 와 바이트 단위로 같은지 + Modbus 예외 진단.
실행: venv/bin/python src/ee/gripper/koras/test_koras_frames.py (하드웨어 불필요)"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from koras_gripper import CMD_GOTO, CMD_IMP_OFF, CMD_IMP_ON, CMD_IMP_PARAMS, IMP_MARGIN, _KorasBase, KorasTeleop


class _Fake(_KorasBase):
    def __init__(self, reply=b""):
        self.sent, self.reply = [], reply

    def _xfer(self, q, wait):
        self.sent.append(q.hex(" "))
        return self.reply


def frame(*a, **k):
    g = _Fake(); g._cmd(*a, **k); return g.sent[0]


def test_frames():
    assert frame(CMD_IMP_ON) == "01 10 00 00 00 02 04 00 6c 00 00 33 b2"
    assert frame(CMD_IMP_OFF) == "01 10 00 00 00 02 04 00 6d 00 00 62 72"
    assert frame(CMD_IMP_PARAMS, 1, 5) == "01 10 00 00 00 03 06 00 6e 00 01 00 05 9e 8a"
    assert frame(CMD_GOTO, 500) == "01 10 00 00 00 02 04 00 68 01 f4 72 64"      # 기존 2레지스터 형식 유지


def test_errors():
    g = _Fake(bytes.fromhex("01 90 01 8d c0"))                 # Modbus 예외 illegal function → 재시도 없이 False
    assert g.set_impedance(True) is False and "illegal function" in g.last_error and len(g.sent) == 1
    g = _Fake()
    assert g.set_impedance_params(2, 10) is False and g.last_error == "응답 없음" and len(g.sent) == 3
    for bad in ((4, 5), (1, 0), (1, 11)):
        try:
            g.set_impedance_params(*bad); raise AssertionError(bad)
        except ValueError:
            pass


def test_teleop_order():
    """켬: 108 → INIT → 110 → 여유 열림 GOTO(1000-IMP_MARGIN, 하드스톱 회피). 끔: 109 → INIT → OPEN. 미지정: INIT → OPEN."""
    calls = []
    class _G(_KorasBase):
        def _cmd(self, code, *v, retries=1): calls.append((code, *v)); return True
        def status(self): return {"pos": 1000, "squeeze": 0.0}
        def init(self): calls.append("init"); return True
    KorasTeleop({"left": _G(), "right": _G(), "x": _G()}, get_squeeze=lambda s: 0.0,
                impedance={"left": (2, 7), "right": None}).start()
    assert calls[:9] == [(108,), "init", (110, 2, 7), (104, 1000 - IMP_MARGIN), (109,), "init", (102,), "init", (102,)], calls   # 이후는 추종 루프 goto


if __name__ == "__main__":
    test_frames(); test_errors(); test_teleop_order(); print("ok")
