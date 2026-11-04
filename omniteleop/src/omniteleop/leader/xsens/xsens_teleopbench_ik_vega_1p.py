#!/usr/bin/env python3
"""
Xsens TF -> vega_1p  TeleOpBench식 다중 프레임 IK 리타게팅 (pink 직접 사용)

    ※ vega_1p 전용 파일. (참고: vega_1 과 vega_1p 의 팔 kinematics 는 URDF FK 비교로
      완전히 동일함을 확인 — 어깨 앵커/링크길이 0.3396·0.4548 m/관절축·리밋 일치.
      로봇별로 따로 관리하기 위해 파일을 분리했다.)

TeleOpBench(https://gorgeous2002.github.io/TeleOpBench/)의 xsens teleop 구조를
Vega-1p 에 이식한 것. 핵심은 "관절각 복사"가 아니라:

    사람 세그먼트 pose(TF) → 팔당 2개 목표 프레임(손 + 팔꿈치) → pink 미분 IK → 관절각

  * TeleOpBench 는 팔당 3프레임(손/팔꿈치/위팔)을 추종하지만, Vega 는 torso 를
    잠그면 어깨(L_arm_l1)가 고정이라 위팔 프레임 태스크는 생략해도 등가다.
  * 팔꿈치까지 추종하므로 7-DOF redundancy 가 사람 팔꿈치를 따라간다
    → EE-only IK(retarget_tf_to_vega.py) 대비 훨씬 사람같은 모션.
  * dexmotion 의 mm.ik() 는 EE 외 프레임 target 을 거부하는 것을 확인했기 때문에
    (Unexpected target poses ... L_arm_l4) pink 를 직접 쓴다.

[convert_ros_to.py 와의 방향 정합성]
  - IK 해는 URDF(vega_1p.urdf) 관절 규약으로 나온다. convert_ros_to.py 가 잘 돌던
    것이 곧 "robot/safe_commands 의 pos = URDF 규약" 이라는 증거다
    (예: 좌 J1=+1.57/우 J1=-1.57 = 팔 내림 — URDF FK 로 확인, convert 오프셋과 일치).
  - 즉 모터 회전 방향은 URDF 를 통해 자동으로 convert_ros_to 와 같은 규약이 된다.
  - 사람 움직임 방향이 로봇과 어긋나면 관절 부호가 아니라 ROBOT_FROM_HUMAN
    (축 정렬)을 고치는 것이 맞다. DEBUG_AXIS=True 로 확인 후 튜닝.

[검증된 수치 (이 파일 작성 시 실측)]
  - 로봇 위팔(L_arm_l1→L_arm_l4) 0.3396 m, 아래팔(L_arm_l4→L_ee) 0.4548 m
  - 다중 프레임 IK 수렴: hand 오차 0.3 mm, 틱당 ~0.3 ms (50Hz 여유)

실행 (dex_env python + ROS jazzy source; Xsens ROS 드라이버가 TF 발행 중이어야 함):
    source /opt/ros/jazzy/setup.bash
    /home/jude/dex_env/bin/python3 xsens_teleopbench_ik_vega_1p.py
  → 시작 후 CALIB_SECONDS 동안 팔이 보이게 서 있으면 사람 팔길이를 자동 측정해
    로봇 팔길이에 맞는 스케일을 잡는다 (TeleOpBench 의 autofit 과 동일).
"""

import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from tf2_ros import Buffer, TransformListener

import pinocchio as pin
import pink
import qpsolvers
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask

from dexcomm import Node as dexNode
from dexcomm.codecs import DictDataCodec
from dexmate_urdf import robots

# ============================================================================
#  설정
# ============================================================================
ROBOT_TYPE = "vega_1p"
ROBOT_COMMANDS_TOPIC = "robot/safe_commands"
RATE_HZ = 20.0          # command_processor 와 동일 발행 주기
IK_SUBSTEPS = 3         # 틱당 IK 적분 횟수 (수렴성 ↑, 비용 미미)

# --- TF 프레임 이름 (None = 시작 시 TF 트리에서 자동 탐지) ---
PELVIS_FRAME = None            # 예: "skeleton_pelvis"
FRAMES = {
    "left":  {"upper_arm": None, "forearm": None, "hand": None},
    "right": {"upper_arm": None, "forearm": None, "hand": None},
}

# --- 축 정렬: 사람(pelvis 프레임) 축 -> 로봇 base 축 (로봇: X앞 Y왼 Z위) ---
#  스켈레톤이 REP-103 이면 단위 매핑. 방향이 어긋나면 DEBUG_AXIS=True 로 확인 후 수정.
ROBOT_FROM_HUMAN = {
    "x": ("x", +1),
    "y": ("y", +1),
    "z": ("z", +1),
}
DEBUG_AXIS = False      # True: 로봇에 발행 안 하고 사람 벡터만 출력 (축 캘리브레이션용)

# --- 손 방향(orientation) 추종. 위치 추종이 안정된 뒤에 켜는 것 권장 ---
HAND_ORI_COST = 0.0     # 0.0 = 위치 전용. TeleOpBench 는 0.6 사용
R_HAND_OFFSET = {"left": np.eye(3), "right": np.eye(3)}   # 켤 때 손목 축 오프셋 튜닝

# --- IK 태스크 비용 (TeleOpBench 준용: hand pos 3.0, forearm pos 낮게) ---
HAND_POS_COST = 3.0
ELBOW_POS_COST = 1.5
POSTURE_COST = 0.05
POSTURE_GAIN = 0.2

# --- 잠글 조인트 기준값 ---
#  torso 는 omniteleop configs/vega_1.yaml·vega_1p.yaml 의 init_pos.torso 와 동일하게.
#  (0 으로 잠그면 어깨 앵커가 실제보다 0.38m 낮고 장착부 pitch ~9° 어긋남 — FK 확인)
LOCKED_Q = {"torso_j1": 0.52, "torso_j2": 1.05, "torso_j3": 0.37,
            "head_j1": 0.0, "head_j2": 0.0, "head_j3": 0.0}

# --- home 자세: 팔 내림. convert_ros_to.py 의 ±1.57 오프셋과 동일 규약 ---
HOME_Q = {"L_arm_j1": +1.57, "R_arm_j1": -1.57}

# --- 안전 ---
AUTO_START_DELAY = 2.0   # 시작 후 이 시간 동안 발행 안 함
CALIB_SECONDS = 2.0      # 사람 팔길이 자동 측정 시간 (autofit)
MAX_JOINT_STEP = 0.15    # 틱당 관절 변화 상한(rad)
LIMIT_MARGIN = 0.95      # URDF 리밋의 95% 클램프 (TeleOpBench 동일)
TF_TIMEOUT = 0.05

# --- 로봇 상태 기반 소프트 스타트 (J1 미동작/점프 방지) ---
#  로봇은 home(팔 앞으로, init_pos=0)에서 시작하는데 IK 첫 해는 팔내림(j1=±1.57)이라
#  J1 명령-실위치 격차가 커서 추종이 안 걸리는 문제 → 실제 로봇 자세로 IK 를 seed 하고
#  스텝 상한을 서서히 올려 부드럽게 물린다.
ROBOT_JOINTS_TOPIC = "robot/joints"   # robot_controller 의 joint feedback
SOFT_START_SECONDS = 3.0   # 추종 시작 후 스텝 상한을 서서히 올리는 시간
SOFT_START_STEP = 0.02     # 소프트 스타트 시작 시 틱당 스텝 상한(rad)
ROBOT_STATE_TIMEOUT = 8.0  # robot/joints 미수신 허용 시간(초과 시 home 에서 시작+경고)

# --- 모터 방향 보정 (basic control 전관절 실측 대조, 2026-07-02) ---
#  사용자 실측(기준자세=전관절 0=팔 앞으로)과 URDF FK 분류 대조표.
#  로봇좌표: X앞/Y왼/Z위. "바깥" = 몸 중심선에서 멀어지는 쪽.
#   관절| 실측 (오른팔 / 왼팔)            | URDF + 방향 (FK 확인)         |일치
#   j1 | +위로 / -위로 (0=앞, R-90·L+90=내림)| R+위(dz+0.23) / L+아래       | ✓
#   j2 | -바깥 / +바깥                    | 둘다 +Y(왼): R안·L바깥        | ✓
#   j3 | +바깥회전 / -바깥회전            | 둘다 -Y로 기움: R바깥·L안     | ✓
#   j4 | -굽힘(안으로) 양쪽               | -위로컬(굽힘), 리밋 -3.07~0.24| ✓
#   j5 | +바깥회전 / -바깥회전            | j3 과 동일 패턴               | ✓
#   j6 | +안굽힘 / -안굽힘                | R+위 / L+아래 ("안"=위 해석)  | ✓주의
#   j7 | -바깥 / +바깥                    | 둘다 +Y: R안·L바깥            | ✓
#  → 전 관절에서 URDF = 하드웨어 방향 일치. 보정 불필요(빈 dict).
#  ※ 이전의 L_arm_j3=-1 반전은 잘못된 추론이어서 제거함 (실측이 우선).
#  ※ j6 만 "안으로"의 상하 해석이 모호 — 실기에서 손목 상하 굽힘이 반대로 보이면
#     좌우 동시에 {"L_arm_j6": -1.0, "R_arm_j6": -1.0} 추가.
#  발행(URDF→모터)·피드백(모터→URDF) 양쪽에 곱한다(부호반전은 자기역원).
MOTOR_SIGN = {}

L_JOINTS = [f"L_arm_j{i}" for i in range(1, 8)]
R_JOINTS = [f"R_arm_j{i}" for i in range(1, 8)]
ARM_JOINTS = L_JOINTS + R_JOINTS
EE = {"left": "L_ee", "right": "R_ee"}
# ★ l4 프레임 원점은 팔꿈치 축을 따라 +y 로 치우쳐 있고 그 부호가 양팔 동일(미러 아님)
#   → 어깨 대비 왼팔 +10.3cm/오른팔 -1.7cm 로 좌우 비대칭 마커가 됨. 이걸 그대로
#   추종하면 대칭 입력에도 양팔이 같은 월드방향으로 쏠린다(실기에서 확인된 증상:
#   오른팔 j2바깥/j3안, 왼팔 j2안/j3바깥). 그래서 j5 원점 y(-0.0434, 정확히 상쇄)로
#   당긴 '대칭 팔꿈치 중심' OP 프레임을 런타임에 추가해 그걸 추종한다(__init__ 참고).
ELBOW = {"left": "L_elbow_center", "right": "R_elbow_center"}
SHOULDER = {"left": "L_arm_l1", "right": "R_arm_l1"}     # l1 원점 = 어깨(j1)


def _build_align(spec):
    e = {"x": np.array([1., 0, 0]), "y": np.array([0, 1., 0]), "z": np.array([0, 0, 1.])}
    return np.array([spec[r][1] * e[spec[r][0]] for r in ("x", "y", "z")])


R_ALIGN = _build_align(ROBOT_FROM_HUMAN)


# ============================================================================
#  Vega 축소 모델 + pink 다중 프레임 IK  (TeleOpBench Xsense_IK 상당)
# ============================================================================
class VegaMultiFrameIK:
    def __init__(self, robot_type=ROBOT_TYPE):
        # dexmate_urdf 패키지 경로: robots.humanoid.<robot>.<robot>.urdf
        pkg = getattr(robots.humanoid, robot_type)
        urdf_path = str(getattr(pkg, robot_type).urdf)
        full = pin.buildModelFromUrdf(urdf_path)

        # 팔 14관절만 남기고 잠금 (torso/head 는 LOCKED_Q, 나머지는 0)
        qref = pin.neutral(full)
        for name, val in LOCKED_Q.items():
            qref[full.idx_qs[full.getJointId(name)]] = val
        lock_ids = [i for i in range(1, full.njoints) if full.names[i] not in ARM_JOINTS]
        self.model = pin.buildReducedModel(full, lock_ids, qref)

        # 좌우 대칭 '팔꿈치 중심' 프레임 추가 (ELBOW 상수 주석 참고).
        # 오프셋은 하드코딩 대신 URDF 의 j5 배치에서 읽는다 (양팔 동일: y=-0.0434).
        for S in ("L", "R"):
            j4 = self.model.getJointId(f"{S}_arm_j4")
            j5 = self.model.getJointId(f"{S}_arm_j5")
            y_off = float(self.model.jointPlacements[j5].translation[1])
            fid_l4 = self.model.getFrameId(f"{S}_arm_l4")
            placement = self.model.frames[fid_l4].placement * pin.SE3(
                np.eye(3), np.array([0.0, y_off, 0.0]))
            self.model.addFrame(pin.Frame(
                f"{S}_elbow_center", j4, fid_l4, placement, pin.FrameType.OP_FRAME))

        self.data = self.model.createData()

        # home: 팔 내림 (convert_ros_to 규약 ±1.57)
        self.q_home = pin.neutral(self.model)
        for name, val in HOME_Q.items():
            self.q_home[self.model.idx_qs[self.model.getJointId(name)]] = val
        self.q_idx = {n: self.model.idx_qs[self.model.getJointId(n)] for n in ARM_JOINTS}

        self.config = pink.Configuration(self.model, self.data, self.q_home.copy())

        # 관절 리밋 95% 클램프 준비
        lo, hi = self.model.lowerPositionLimit, self.model.upperPositionLimit
        mid, half = (lo + hi) / 2, (hi - lo) / 2 * LIMIT_MARGIN
        self.q_min, self.q_max = mid - half, mid + half

        # 태스크: 팔당 손(pos+선택적 ori) + 팔꿈치(pos)  — TeleOpBench 준용
        self.tasks = {}
        for side in ("left", "right"):
            self.tasks[f"{side}_hand"] = FrameTask(
                EE[side], position_cost=HAND_POS_COST,
                orientation_cost=HAND_ORI_COST, lm_damping=1.0,
            )
            self.tasks[f"{side}_elbow"] = FrameTask(
                ELBOW[side], position_cost=ELBOW_POS_COST,
                orientation_cost=0.0, lm_damping=1.0,
            )
        self.posture = PostureTask(cost=POSTURE_COST, lm_damping=1.0, gain=POSTURE_GAIN)
        self.posture.set_target(self.q_home)

        self.solver = "quadprog" if "quadprog" in qpsolvers.available_solvers \
            else qpsolvers.available_solvers[0]

        # 로봇 앵커/링크 길이 (home FK 기준; 실측: upper≈0.340, fore≈0.455)
        self.shoulder_anchor, self.robot_len = {}, {}
        for side in ("left", "right"):
            sh = self.frame_pos(SHOULDER[side])
            el = self.frame_pos(ELBOW[side])
            ee = self.frame_pos(EE[side])
            self.shoulder_anchor[side] = sh
            self.robot_len[side] = {"upper": np.linalg.norm(el - sh),
                                    "fore": np.linalg.norm(ee - el)}

    def frame_pos(self, frame):
        return np.asarray(self.config.get_transform_frame_to_world(frame).translation)

    @staticmethod
    def _se3(pos, rot=None):
        T = pin.SE3.Identity()
        T.translation = np.asarray(pos, dtype=float)
        if rot is not None:
            T.rotation = np.asarray(rot, dtype=float)
        return T

    def set_targets(self, side, hand_pos, elbow_pos, hand_rot=None):
        # 방향을 안 쓰는 태스크에도 현재 프레임 방향을 넣어 target 을 유효하게 유지
        cur_h = self.config.get_transform_frame_to_world(EE[side]).rotation
        cur_e = self.config.get_transform_frame_to_world(ELBOW[side]).rotation
        rot = hand_rot if (hand_rot is not None and HAND_ORI_COST > 0) else cur_h
        self.tasks[f"{side}_hand"].set_target(self._se3(hand_pos, rot))
        self.tasks[f"{side}_elbow"].set_target(self._se3(elbow_pos, cur_e))

    def solve(self, dt):
        """미분 IK 1틱 (substep 적분) -> 관절각 dict."""
        sub_dt = dt / IK_SUBSTEPS
        for _ in range(IK_SUBSTEPS):
            v = solve_ik(self.config,
                         list(self.tasks.values()) + [self.posture],
                         sub_dt, solver=self.solver)
            self.config.integrate_inplace(v, sub_dt)
        q = np.clip(self.config.q, self.q_min, self.q_max)   # 리밋 95% 클램프
        self.config.update(q)
        return {n: float(q[i]) for n, i in self.q_idx.items()}

    def seed(self, joint_dict):
        """IK 내부 상태를 실제 로봇 관절각(URDF 규약)으로 초기화."""
        q = self.config.q.copy()
        for n, v in joint_dict.items():
            if n in self.q_idx:
                q[self.q_idx[n]] = v
        q = np.clip(q, self.q_min, self.q_max)
        self.config.update(q)


# ============================================================================
#  ROS 노드: TF 수신 → 리타게팅 → IK → 발행
# ============================================================================
class XsensTeleopBenchIKVega(Node):
    def __init__(self, robot_type=ROBOT_TYPE):
        super().__init__("xsens_teleopbench_ik_vega_1p")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"{robot_type} 다중 프레임 IK 초기화 중 (pink/pinocchio)...")
        self.ik = VegaMultiFrameIK(robot_type)
        for side in ("left", "right"):
            rl = self.ik.robot_len[side]
            self.get_logger().info(
                f"[robot] {side} shoulder={np.round(self.ik.shoulder_anchor[side],3)} "
                f"upper={rl['upper']:.3f}m fore={rl['fore']:.3f}m"
            )

        self.dex_node = dexNode(name="dex_node")
        self.dexPub = self.dex_node.create_publisher(
            ROBOT_COMMANDS_TOPIC, encoder=DictDataCodec.encode
        )
        # robot_controller joint feedback 구독 (소프트 스타트 seed 용)
        self._state_lock = threading.Lock()
        self._robot_joints = None
        self.dexSub = self.dex_node.create_subscriber(
            ROBOT_JOINTS_TOPIC, callback=self._on_robot_joints,
            decoder=DictDataCodec.decode,
        )
        self.tracking_started = False
        self.track_start_time = None
        self.track_wait_start = None

        self.pelvis = PELVIS_FRAME
        self.frames = {s: dict(FRAMES[s]) for s in FRAMES}
        self.frames_resolved = False

        # 사람 팔길이 캘리브레이션 (autofit)
        self.calib = {"left": {"upper": [], "fore": []}, "right": {"upper": [], "fore": []}}
        self.scale = None      # {"left": {"upper": s, "fore": s}, ...}

        self.prev_pub = None   # 직전 발행 관절 dict (스텝 제한용)
        self.start_time = time.monotonic()
        self.log_counter = 0
        self.published_count = 0

        self.timer = self.create_timer(1.0 / RATE_HZ, self.on_timer)
        self.get_logger().info(
            f"시작. {AUTO_START_DELAY:.0f}s 대기 + {CALIB_SECONDS:.0f}s 팔길이 캘리브레이션 후 발행."
        )

    # ---- TF 트리에서 프레임 자동 탐지 ----
    def resolve_frames(self):
        import re
        names = re.findall(r"Frame (\S+) exists", self.tf_buffer.all_frames_as_string())
        if not names:
            return False

        def pick(words, side=None):
            cands = [n for n in names if (side is None or side in n.lower())]
            for w in words:
                for n in cands:
                    if w in n.lower():
                        return n
            return None

        if not self.pelvis:
            self.pelvis = pick(["pelvis", "hip", "t8", "spine", "root"])
        want = {"hand": ["hand", "wrist"],
                "forearm": ["forearm", "lower_arm", "lowerarm", "elbow"],
                "upper_arm": ["upper_arm", "upperarm", "shoulder"]}
        for side in ("left", "right"):
            for key, words in want.items():
                if not self.frames[side][key]:
                    self.frames[side][key] = pick(words, side)

        self.get_logger().info(f"[TF] pelvis={self.pelvis} frames={self.frames}")
        missing = [f"{s}/{k}" for s in self.frames for k in self.frames[s]
                   if self.frames[s][k] is None]
        if not self.pelvis or missing:
            self.get_logger().warning(
                f"프레임 자동탐지 실패({missing}). 아래에서 골라 상단 상수에 지정:\n  "
                + "\n  ".join(sorted(names))
            )
            return False
        return True

    def lookup_pos(self, frame):
        """pelvis 기준 frame 위치. 실패 시 None."""
        try:
            tf = self.tf_buffer.lookup_transform(
                self.pelvis, frame, Time(), timeout=Duration(seconds=TF_TIMEOUT))
        except Exception:
            return None
        t = tf.transform.translation
        return np.array([t.x, t.y, t.z])

    def lookup_rot(self, frame):
        try:
            tf = self.tf_buffer.lookup_transform(
                self.pelvis, frame, Time(), timeout=Duration(seconds=TF_TIMEOUT))
        except Exception:
            return None
        q = tf.transform.rotation
        return Rot.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    # ---- 사람 팔 벡터 읽기: (어깨→팔꿈치, 팔꿈치→손) ----
    def read_human_arm(self, side):
        p_sh = self.lookup_pos(self.frames[side]["upper_arm"])
        p_el = self.lookup_pos(self.frames[side]["forearm"])
        p_ha = self.lookup_pos(self.frames[side]["hand"])
        if p_sh is None or p_el is None or p_ha is None:
            return None
        return p_el - p_sh, p_ha - p_el

    # ---- autofit: 사람 팔길이 측정 → 로봇 링크 길이 대비 스케일 ----
    def try_calibrate(self):
        for side in ("left", "right"):
            arm = self.read_human_arm(side)
            if arm is not None:
                self.calib[side]["upper"].append(np.linalg.norm(arm[0]))
                self.calib[side]["fore"].append(np.linalg.norm(arm[1]))
        elapsed = time.monotonic() - self.start_time - AUTO_START_DELAY
        if elapsed < CALIB_SECONDS:
            return
        self.scale = {}
        for side in ("left", "right"):
            sc = {}
            for part in ("upper", "fore"):
                human = float(np.mean(self.calib[side][part])) if self.calib[side][part] else 0.0
                if human < 0.05:   # TF 미수신/비정상 → 스케일 1.0 폴백
                    self.get_logger().warning(
                        f"[autofit] {side}/{part} 사람 길이 비정상({human:.3f}m) → scale=1.0")
                    sc[part] = 1.0
                else:
                    sc[part] = self.ik.robot_len[side][part] / human
            self.scale[side] = sc
        self.get_logger().info(f"[autofit] 완료: {self.scale} → 추종 시작")

    # ---- 로봇 상태 피드백 ----
    def _on_robot_joints(self, data):
        joints = data.get("joints", {}) if isinstance(data, dict) else {}
        if "left_arm" in joints and "right_arm" in joints:
            with self._state_lock:
                self._robot_joints = {"left_arm": list(joints["left_arm"]),
                                      "right_arm": list(joints["right_arm"])}

    def _robot_state_urdf(self):
        """robot/joints 피드백 → URDF 규약 관절 dict (MOTOR_SIGN 적용)."""
        with self._state_lock:
            rj = self._robot_joints
        if rj is None:
            return None
        out = {}
        for comp, names in (("left_arm", L_JOINTS), ("right_arm", R_JOINTS)):
            for name, v in zip(names, rj[comp]):
                out[name] = MOTOR_SIGN.get(name, 1.0) * float(v)
        return out

    # ---- 사람 벡터 → 로봇 목표 (체인 누적 스케일링: TeleOpBench 방식) ----
    def retarget(self, side, v_se, v_eh):
        sc = self.scale[side]
        elbow = self.ik.shoulder_anchor[side] + sc["upper"] * (R_ALIGN @ v_se)
        hand = elbow + sc["fore"] * (R_ALIGN @ v_eh)
        return hand, elbow

    def on_timer(self):
        now = time.monotonic()
        if not self.frames_resolved:
            if now - self.start_time < 1.5:
                return
            if not self.resolve_frames():
                return
            self.frames_resolved = True

        if now - self.start_time < AUTO_START_DELAY:
            return

        # 캘리브레이션 단계
        if self.scale is None:
            self.try_calibrate()
            return

        # 추종 시작 전: 실제 로봇 자세로 IK seed (J1 명령-실위치 격차 제거)
        if not self.tracking_started:
            if self.track_wait_start is None:
                self.track_wait_start = now
            state = self._robot_state_urdf()
            if state is None:
                if now - self.track_wait_start < ROBOT_STATE_TIMEOUT:
                    if int((now - self.track_wait_start) * RATE_HZ) % int(RATE_HZ) == 0:
                        self.get_logger().info("robot/joints 수신 대기 중... (robot_controller 실행 확인)")
                    return
                self.get_logger().warning(
                    "robot/joints 미수신 → home 자세에서 시작 (J1 이 안 걸릴 수 있음)")
                state = {n: float(self.ik.q_home[self.ik.q_idx[n]]) for n in ARM_JOINTS}
            self.ik.seed(state)
            self.prev_pub = dict(state)
            self.tracking_started = True
            self.track_start_time = now
            self.get_logger().info(
                f"[soft-start] 로봇 현재 자세에서 추종 시작 "
                f"(L_j1={state['L_arm_j1']:.2f}, R_j1={state['R_arm_j1']:.2f}, "
                f"{SOFT_START_SECONDS:.0f}s 램프)")

        # 사람 팔 읽기 (실패한 쪽은 직전 목표 유지 = 태스크 target 미갱신)
        ok = {}
        for side in ("left", "right"):
            arm = self.read_human_arm(side)
            ok[side] = arm is not None
            if arm is None:
                continue
            hand, elbow = self.retarget(side, *arm)
            hand_rot = None
            if HAND_ORI_COST > 0:
                hr = self.lookup_rot(self.frames[side]["hand"])
                if hr is not None:
                    hand_rot = R_ALIGN @ hr @ R_HAND_OFFSET[side]
            if DEBUG_AXIS:
                continue
            self.ik.set_targets(side, hand, elbow, hand_rot)

        self.log_counter += 1
        do_log = self.log_counter % int(RATE_HZ) == 0

        # 축 캘리브레이션 모드: 발행 없이 사람 벡터만 출력
        if DEBUG_AXIS:
            if self.log_counter % max(1, int(RATE_HZ // 4)) == 0:
                la = self.read_human_arm("left")
                if la is not None:
                    self.get_logger().info(
                        f"[DEBUG축] L 어깨→손={np.round(R_ALIGN @ (la[0]+la[1]), 3)} "
                        "(앞/위/바깥으로 움직여 +축 확인)")
            return

        if not (ok["left"] or ok["right"]):
            if do_log:
                self.get_logger().warning("양팔 TF 모두 실패 → 직전 목표 유지 중")

        # --- IK 1틱 ---
        try:
            sol = self.ik.solve(1.0 / RATE_HZ)
        except Exception as e:
            self.get_logger().warning(f"IK 실패: {e}")
            return

        # 틱당 스텝 제한 — 소프트 스타트 동안은 작게 시작해 서서히 완화
        ramp = min(1.0, (now - self.track_start_time) / SOFT_START_SECONDS)
        step_max = SOFT_START_STEP + ramp * (MAX_JOINT_STEP - SOFT_START_STEP)
        if self.prev_pub is not None:
            for k, v in sol.items():
                d = v - self.prev_pub[k]
                if abs(d) > step_max:
                    sol[k] = self.prev_pub[k] + np.clip(d, -step_max, step_max)
        self.prev_pub = dict(sol)

        left = [float(MOTOR_SIGN.get(j, 1.0) * sol[j]) for j in L_JOINTS]
        right = [float(MOTOR_SIGN.get(j, 1.0) * sol[j]) for j in R_JOINTS]
        combined_command = {
            "timestamp_ns": time.time_ns(),
            "components": {
                "left_arm": {"pos": left, "mode": "absolute"},
                "right_arm": {"pos": right, "mode": "absolute"},
                "safety_flags": {"emergency_stop": False, "exit_requested": False},
            },
        }
        self.dexPub.publish(combined_command)
        self.published_count += 1
        if do_log:
            self.get_logger().info(
                f"[IK] TF L={'OK' if ok['left'] else 'X'} R={'OK' if ok['right'] else 'X'} "
                f"L팔={[round(x,2) for x in left]} 발행={self.published_count}")


def main(args=None):
    rclpy.init(args=args)
    node = XsensTeleopBenchIKVega()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# ============================================================================
#  캘리브레이션 / 사용 절차
# ============================================================================
#  0) 처음엔 시뮬/브레이크 상태에서. Xsens ROS 드라이버(TF 발행) 먼저 실행.
#  1) 축 정렬: DEBUG_AXIS=True 로 실행 → 손을 (1)앞 (2)위 (3)바깥으로 움직여
#     출력 벡터의 +축이 로봇 규약(X앞/Y왼/Z위)과 일치하는지 확인.
#     어긋나면 ROBOT_FROM_HUMAN 의 (축,부호)를 수정. 끝나면 False 로.
#  2) 실행 직후 AUTO_START_DELAY+CALIB_SECONDS 동안 팔이 TF 에 보이게 서 있기
#     (팔길이 autofit). 이후 추종 시작.
#  3) 위치 추종이 자연스러우면 HAND_ORI_COST=0.6 으로 손 방향 추종을 켜고,
#     손을 비틀 때 로봇 손목이 같은 축으로 돌도록 R_HAND_OFFSET 을 튜닝.
#  4) 움직임이 둔하면 POSTURE_COST↓ 또는 HAND_POS_COST↑,
#     팔꿈치가 사람과 다르게 서면 ELBOW_POS_COST↑.
# ============================================================================
