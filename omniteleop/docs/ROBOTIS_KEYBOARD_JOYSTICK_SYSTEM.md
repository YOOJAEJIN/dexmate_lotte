# Robotis 조이스틱 + 키보드 입력 시스템

## 개요

Robotis 조이스틱과 키보드를 조합하여 로봇을 제어하는 시스템입니다.

**입력 장치:**
- Robotis 조이스틱: X, Y 축 + Stick 버튼 (조이스틱 누르기)
- Keyboard: 모드 전환, 응급 제어
- 그리퍼: 모터 피드백 읽기

## 시스템 구조

```
조이스틱 (X, Y, 버튼)                    
    ↓                                      
robotis_exo_arm_reader.py                 
    ├─ 팔 모터 위치 읽기 → exo/joints     
    └─ 조이스틱 데이터 + 그리퍼 → exo/joycon
    
    ↓
    
RobotisExoKeyboardHandler (keyboard 입력)
    ↓
    └─ on_keyboard_event() 콜백 호출
    
    ↓
    
RobotisExoController (메인 제어기)
    ├─ process(joycon_data) - 조이스틱 처리
    ├─ on_keyboard_event(command) - 키보드 처리
    └─ get_gripper_commands() - 그리퍼 명령
    
    ↓
    
component_processors (각 모드별 처리)
    ├─ BaseController (팔 움직임)
    ├─ HeadController (머리)
    ├─ TorsoController (허리)
    └─ HandController (손/그리퍼)
```

## 파일 설명

### 1. robotis_exo_keyboard_handler.py (새로 생성)

**용도:** 키보드 입력을 명령으로 변환

**기본 키 매핑:**
```python
{
    "h": "head_toggle",        # 머리 제어 토글
    "t": "torso_toggle",       # 허리 제어 토글
    "b": "base_toggle",        # 팔 제어 토글
    "f": "fine_toggle",        # 세밀 조정 토글
    "r": "recording_toggle",   # 녹음 시작/중지
    "q": "estop_toggle",       # 긴급 정지
    "esc": "exit_request",     # 종료 요청
}
```

**사용 예제:**
```python
from robotis_exo_keyboard_handler import RobotisExoKeyboardHandler, create_pynput_listener

# 핸들러 생성
handler = RobotisExoKeyboardHandler()

# 콜백 등록 (선택사항)
handler.register_callback("head_toggle", lambda: print("머리 모드!"))

# pynput으로 키 입력 수신
listener = create_pynput_listener(handler)
listener.start()

# 또는 수동으로 키 상태 업데이트
handler.update_key_state("h", pressed=True)
handler.update_key_state("h", pressed=False)
```

**메서드:**
- `update_key_state(key, pressed)` - 키 상태 업데이트
- `register_callback(command, callback)` - 콜백 등록
- `set_key_mapping(key, command)` - 키 매핑 변경
- `get_key_map()` - 현재 매핑 확인

### 2. robotis_exo_controller.py (수정됨)

**용도:** 조이스틱 + 키보드 입력을 로봇 명령으로 변환

**주요 변경:**
- JoyCon 전용 버튼 콤보 제거 (원래 JoyCon 매핑: L+R, +/-, ZL/ZR 등)
- 간단한 stick 버튼 처리만 유지
- 키보드 이벤트 처리 메서드 추가
- 그리퍼 제어 메서드 추가

**제어 모드 우선순위:**
```
exit > estop > base > head > torso > gripper
```

**핵심 메서드:**

1. **on_keyboard_event(command)**
   ```python
   # 키보드에서 받은 명령 처리
   controller.on_keyboard_event("head_toggle")
   controller.on_keyboard_event("gripper_open")
   controller.on_keyboard_event("estop_toggle")
   ```

2. **process(joycon_data)**
   ```python
   # 조이스틱 데이터 처리 (40Hz로 계속 호출)
   commands = controller.process(joycon_data)
   # joycon_data 구조:
   # {
   #     "left": {
   #         "buttons": {"stick": bool},
   #         "stick": {"x": float, "y": float},
   #         "gripper": {"position": float}
   #     },
   #     "right": { ... }
   # }
   ```

3. **Gripper 제어**
   ```python
   # 목표 위치 설정 (0.0 = 닫음, 1.0 = 완전 열림)
   controller.set_gripper_target("left", 0.5)
   controller.set_gripper_target("right", 0.5)
   
   # 피드백 업데이트 (robotis_exo_arm_reader에서 읽은 값)
   controller.update_gripper_feedback(
       left_pos=motor_data["left_gripper"],
       right_pos=motor_data["right_gripper"]
   )
   
   # 그리퍼 명령 조회
   commands = controller.get_gripper_commands()
   # {"left": 0.5, "right": 0.5}
   ```

### 3. robotis_exo_arm_reader.py (수정됨)

**변경 사항:**

JoyConData에 그리퍼 정보 추가:
```python
JoyConData(
    timestamp_ns=...,
    left={
        "buttons": {"stick": bool},
        "stick": {"x": float, "y": float},
        "gripper": {"position": float}  # ← 새로 추가
    },
    right={...}
)
```

## 통합 예제

```python
from omniteleop.follower.input_handlers.control.joycon.robotis_exo_keyboard_handler import (
    RobotisExoKeyboardHandler,
    create_pynput_listener
)
from omniteleop.follower.input_handlers.control.joycon.robotis_exo_controller import (
    RobotisExoController
)

# 초기화
keyboard_handler = RobotisExoKeyboardHandler()
controller = RobotisExoController(config={...})

# 키보드 리스너 시작 (별도 스레드)
listener = create_pynput_listener(keyboard_handler)
listener.start()

# 키보드 → 컨트롤러 연결
def on_keyboard_command(command):
    controller.on_keyboard_event(command)

for command in keyboard_handler.callbacks.keys():
    keyboard_handler.register_callback(command, 
        lambda c=command: on_keyboard_command(c))

# 메인 루프 (40Hz)
while running:
    # 1. 조이스틱 + 그리퍼 데이터 수신 (exo/joycon 토픽)
    joycon_data = receive_joycon_data()
    
    # 2. 그리퍼 피드백 업데이트
    controller.update_gripper_feedback(
        left_pos=joycon_data["left"]["gripper"]["position"],
        right_pos=joycon_data["right"]["gripper"]["position"]
    )
    
    # 3. 제어 명령 생성
    commands = controller.process(joycon_data)
    
    # 4. 로봇에 명령 전송
    send_to_robot(commands)
    
    # 키보드는 별도 스레드에서 처리
    # keyboard_handler.update_key_state() → on_keyboard_event()
```

## 커스터마이징

### 키 매핑 변경

```python
handler = RobotisExoKeyboardHandler()

# 개별 키 매핑 변경
handler.set_key_mapping("w", "head_toggle")  # W 키로 머리 제어
handler.set_key_mapping("a", "torso_toggle")  # A 키로 허리 제어

# 또는 처음부터 새로운 매핑으로 초기화
custom_map = {
    "w": "head_up",
    "s": "head_down",
    "a": "head_left",
    "d": "head_right",
}
handler = RobotisExoKeyboardHandler(key_map=custom_map)
```

### Stick 버튼 활용

현재 stick 버튼은 특정 기능이 없습니다. 원하는 대로 할당할 수 있습니다:

**옵션 1: 그리퍼 제어**
```python
# robotis_exo_controller.py의 process() 메서드에 추가
if joycon_data["left"]["buttons"]["stick"]:
    # stick 누르면 left 그리퍼 토글
    self.gripper_left_toggled = not self.gripper_left_toggled
```

**옵션 2: 속도 조절**
```python
# stick 누르면 slow mode
slow_mode = joycon_data["left"]["buttons"]["stick"]
base_velocity = 0.2 if slow_mode else 0.5
```

## 데이터 흐름 다시 정리

1. **robotis_exo_arm_reader.py** (리더)
   - 팔 모터: 40Hz로 `exo/joints` 토픽에 발행
   - 조이스틱 + 그리퍼: 40Hz로 `exo/joycon` 토픽에 발행

2. **RobotisExoKeyboardHandler** (입력 수신)
   - 키보드 입력을 감지하여 명령으로 변환
   - pynput 사용 (또는 다른 방식)

3. **RobotisExoController** (제어 로직)
   - `process(joycon_data)`: 조이스틱 데이터 처리
   - `on_keyboard_event(command)`: 키보드 명령 처리
   - 우선순위에 따라 component controller 선택

4. **component_processors** (모드별 처리)
   - BaseController: 팔 이동
   - HeadController: 머리 제어
   - TorsoController: 허리 제어
   - HandController: 그리퍼 제어

## 주요 특징

✅ **유연한 입력:** 조이스틱 + 키보드 조합  
✅ **그리퍼 피드백:** 모터 위치 읽기 및 제어  
✅ **안전성:** 긴급 정지 (estop) 기능  
✅ **우선순위:** 명확한 제어 모드 우선순위  
✅ **커스터마이징:** 키 매핑 쉽게 변경 가능  
✅ **확장성:** 새로운 모드/명령 추가 용이

## 다음 단계

1. **키보드 리스너 통합:** ExoJoyconHandler에서 keyboard_handler 초기화
2. **그리퍼 제어 로직:** gripper 명령을 실제 모터 제어로 변환
3. **테스트:** 각 모드 별로 입력-출력 검증
4. **튜닝:** 감도, 속도 등 파라미터 조정
