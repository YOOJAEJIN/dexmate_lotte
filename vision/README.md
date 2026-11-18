# vision — Targeting System (구 HEAD SEG): 상자 더미 세그멘테이션 · 3D 좌표 · 파지점 / place 타깃 엣지

헤드 ZED X Mini 영상에서 골판지 상자 더미만 분할하고, 실측 depth와 헤드 IMU의 중력 방향으로
상자 앞면의 꼭짓점·크기·층 경계선(6.1 cm = 20장)·파지점을 계산해
omniteleop GUI의 `HEAD SEG` 타일(WebRTC 60006)에 실시간으로 띄운다.
상자 크기와 층 경계선은 카메라와 IMU만으로 계산하므로 토르소·헤드 자세가 바뀌어도 같은 값이 나온다.
**베이스 좌표계** 값은 여기에 로봇 FK(실시간 관절값)를 더해 구한다.

```
nano  gstsender.py ──H.264 RTP──▶ Thor webrtc_launch.py 60001 ──zmq 55601──┐
      (ZED X Mini)  └─TCP 55703 (metric depth, uint16 mm)─────────────────┤
      └─zenoh sensors/head_camera/imu (중력 방향, ~140 Hz)───────────────┤
Thor  robot_controller ─zenoh robot/joints (100 Hz, 1초 넘게 끊기면 미사용)─┤
                                                                         seg_overlay.py
                                                                           │ FastSAM-s + 중력 + FK
      GUI HEAD SEG ◀── webrtc_launch.py 60006 ◀──H.264 RTP udp 50006 ─────┘
```

## 파일

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `seg_overlay.py` | Thor (GPU) | 세그멘테이션 · 3D/베이스 좌표 · 오버레이 송출 |
| `webrtc_launch.py` | Thor | UDP H.264 → WebRTC 릴레이. `SEG_TAP_PORT` 설정 시 디코드 프레임을 zmq로 공개 |
| `face_plane.py` | (seg_overlay가 import) | 상자 앞면 = 수직 평면 위 직사각형 (자세·윗면·그리퍼 가림에 무관한 꼭짓점) |
| `box_hold.py` | (seg_overlay가 import) | 그리퍼 가림 시 마지막 전체 측정 유지 |
| `nano/gstsender.py` | nano (`~/gstsender/`) | ZED 송출. 기존 3 스트림 + **실측 depth TCP 55703** + 헤드 IMU(zenoh) |

## 실행

GUI(`omniteleop/src/omniteleop/app/launch.sh`)를 띄운 뒤:

```bash
vision/start_head_seg.sh
```

nano 송출기 → Thor 릴레이 6개 → seg_overlay 순서로 띄운다. 이미 떠 있는 부분은 건너뛰므로 여러 번 실행해도 안전하고,
부팅 직후 nano가 늦게 올라와도 최대 3분 기다린다. 로그는 `vision/logs/`.

부팅 시 자동 실행 (`crontab -e`에 한 줄):

```
@reboot <repo>/vision/start_head_seg.sh >> <repo>/vision/logs/autostart.log 2>&1
```

경로 기본값은 vega-1p 로봇 기준이며 `NANO`, `NANO_PY`, `NANO_SENDER`, `RELAY_PY`, `SEG_PY`, `ROBOT_NAME`, `ZENOH_CONFIG` 환경변수로 바꾼다.
nano 쪽 `gstsender.py`는 `vision/nano/gstsender.py`를 nano의 `~/gstsender/`에 둔다.

수동으로 다룰 때 주의:
- nano 송출기는 SIGINT를 무시하므로 종료는 `kill <pid>`(SIGTERM). `pgrep -f gstsender`는 ssh로 띄울 때 생긴 `bash -c` 래퍼도 잡으니, 실제 python pid인지 확인한다.
- 릴레이의 `SEG_TAP_PORT`는 60001에만 준다. 모든 포트에 같은 값을 주면 bind 충돌로 탭이 동작하지 않는다.
- 릴레이 인증서: 기본 `/home/dexmate/certs/{cert,key}.pem`, 또는 `--cert/--key`, `XR_TELEOP_CERT/XR_TELEOP_KEY`.
- GUI를 릴레이와 다른 PC에서 띄우면 `CAM_HOST=<로봇 IP>` (vite가 `/cam/<port>/`로 중계하므로 브라우저 인증서 수락 불필요).
- seg_overlay는 torch(CUDA) · ultralytics · opencv · pyzmq · pinocchio(`pip install pin`) · dexcomm(≥0.6, IMU·관절 구독) 필요. `FastSAM-s.pt`는 없으면 자동 다운로드.
- 화면 왼쪽 위 `grav:` 표시가 중력 출처다 — `IMU`(정상), `FK`(IMU 없음, 관절값으로 대신), `-`(둘 다 없음: 자세에 따라 값이 변하는 예전 방식으로 계산).
- 좌표는 `robot/joints`가 실시간으로 들어올 때만 베이스(B), 아니면 카메라(C) 기준이다. 관절값은 teleop 스택의 `robot_controller`가 publish한다.
  GUI가 Start 상태여도 `robot_controller`가 죽어 있으면(예: HEARTBEAT TIMEOUT) C로 나온다 — `app/logs/robot_controller.log` 확인 후 Stop → Start.

결과는 화면 오버레이와 `vision/logs/box_coords.json`(2초마다: 상자별 bbox · corners · size_m(폭, 높이) · layers · grasp)로 나온다.

## 처리 방법

1. **세그멘테이션 — FastSAM-s**: SAM의 everything 출력을 흉내 내도록 학습된 YOLOv8-seg 계열 CNN. Thor 640×480 기준 13 ms.
   (yolo11n-seg 7 ms는 COCO 클래스라 골판지 없음, SAM2.1-tiny everything 6100 ms라 실시간 불가.)
   2초마다 전체 화면 스캔(좌표 계산), 매 프레임 줌 화면 재분할(마스크 표시).
2. **박스 판별**: 면적 0.5~35% · 화면 85% 이상 가로지르지 않음 · 테두리 2변 이상 접촉 안 함 · depth 0.3~1.6 m ·
   **채도 기반 갈색 게이트**(H 5~30, S≥80 픽셀 50% 이상 — Hue는 상자/크레이트 모두 ~16이라 채도로 구분. 상자 S는 조명에 따라 89~130, 회색 크레이트 ~67) ·
   가장 큰 후보에서 인접한 것만 묶는 클러스터링.
3. **포커스 줌**: 더미 끝점 → 4% 여백 → 4:3 맞춤 → EMA(0.7/0.3) → bicubic 확대.
4. **Depth**: ZED SDK 스테레오(NEURAL)의 `retrieve_measure(DEPTH)`를 mm로 받아 9×9 중앙값.
   60003의 `VIEW.DEPTH`는 표시용 렌더라 장면에 따라 스케일이 바뀌어(같은 거리에서 픽셀 193→58) 쓰지 않는다.
5. **카메라 모델**: 핀홀, ZED 공장 캘리브(SN50691325, FHD1200)를 640×480으로 환산 — fx 245.1, fy 294.1, cx 330.9, cy 247.6.
   1920×1200→640×480 리사이즈가 비균일(가로 1/3, 세로 0.4)이라 fx≠fy.
6. **꼭짓점 — 중력 기준 수직 평면** (`face_plane.py`): 세워 쌓은 더미의 앞면은 수직 평면이다.
   - 중력 방향: 헤드 IMU 가속도 평균(카메라에 붙어 있어 관절값·URDF와 무관). IMU가 없으면 FK.
   - 마스크 안 depth 점을 위에서 내려다본 평면에 놓고 RANSAC 직선(허용 1.5 cm) → 앞면 평면.
   - 윗변: 평면 위 depth 점의 밀도가 시작되는 높이. 위에서 내려다보면 윗면(맨 위 장)이 마스크에 같이 들어오는데, 윗면 점은 수직 평면에서 벗어나므로 높이에 더해지지 않는다.
   - 좌우·아랫변: 마스크 픽셀의 광선을 평면과 교차시킨 위치의 밀도 경계(전형 밀도의 35 %). 실루엣에서는 스테레오 depth가 몇 픽셀 번져서(왼쪽 변 2 cm 손실) depth 대신 마스크 경계를 쓴다.
   - 그리퍼가 모서리 앞에 있어도 그 점들은 평면 밖이고, 변은 가려지지 않은 행·열에서 정해지므로 꼭짓점이 움직이지 않는다(변의 65 %까지 가려져도 됨).
   - 중력을 못 얻으면 예전 방식(자유 평면 + 마스크 볼록껍질의 대각선 극점)으로 계산한다. 이 방식은 보는 높이에 따라 값이 달라진다 — 같은 45.5 cm 더미가 위에서 보면 50 cm로 나왔다.
   자체 검사: `python face_plane.py` (카메라 높이·기울기·yaw를 바꾼 다섯 자세에서 같은 치수).
7. **층 경계선 / 파지점**: 좌우 세로 모서리를 3D에서 위부터 `SEG_PITCH`(0.061 m)씩 따라 내려가 재투영. GRASP = 윗변~L1 중간(맨 위 20장 묶음 중심).
8. **베이스 좌표**: pinocchio로 `vega_1p_gripper.urdf`의 `zed_left_camera`까지 FK, 관절값은 zenoh `robot/joints`(수신 후 1초 이내 것만).
   백엔드 `GET /robots/joints`는 마지막 값을 시각 없이 계속 돌려줘서 쓰지 않는다 — `robot_controller`가 죽은 뒤의 캐시로 카메라 기울기가 22° 틀렸다(2026-11-12).
   실시간 관절값일 때 FK와 IMU의 중력 방향 차이는 0.3°.
   장착 보정 0 — reach 설정의 `camera_mount_rpy_corr`(Isaac Lab용)는 실기에서 수직 모서리를 20° 기울인다.
   베이스 원점은 바닥 위 5.48 cm(구동휠 중심 3.25 cm + 반지름 8.73 cm).
9. **가림 유지** (`box_hold.py`): 그리퍼가 파지 대상 상자를 가리면 가리기 전 마지막 측정을 유지하고 화면에 `GRASP HELD`로 표시한다.
   가림 판정은 3D 윗변·아랫변 폭으로 한다 — 짧은 변이 긴 변의 0.7배 미만이거나, 긴 변이 직전 측정의 짧은 변보다 0.85배 미만일 때.
   높이 변화(맨 위 20장 묶음을 들어냄)는 가림으로 보지 않는다. 유지 중인 점은 현재 헤드 자세로 재투영한다. 자체 검사: `python box_hold.py`.

## 정확도 (실측 대비)

| 항목 | 낮은 자세, 정면 (11-10, 예전 방식) | 높은 자세, 31° 내려다봄 (11-12, 수직 평면) | 실측 |
|---|---|---|---|
| 폭 | 41.1 / 40.5 cm (윗변/아랫변) | 41.4 cm | 40.5 cm |
| 높이 | 44.5~44.8 cm | 46.5 cm | 45.5 cm |
| 바닥→윗면 | 136.2 cm | 134.9 cm | 137.5 cm |

640×480에서 1.2 m 거리의 1픽셀이 약 0.5 cm라, 변 하나당 ±1픽셀이 곧 ±1 cm다.
높은 자세에서 예전 방식은 높이 50~67 cm가 나왔다(윗면 포함, 그리퍼 depth 혼입).

## 환경변수 (seg_overlay.py)

`SEG_TAP`(tcp://127.0.0.1:55601) · `SEG_DEPTH_SRC`(192.168.50.22:55703) · `SEG_DEST`/`SEG_PORT`(127.0.0.1/50006) ·
`SEG_MODEL`(FastSAM-s.pt) · `SEG_CONF`(0.4) · `SEG_SCAN_S`(2.0) · `SEG_PITCH`(0.061) ·
`SEG_BROWN_H0`/`H1`/`S`/`V`/`MIN`(5/30/80/50/0.25, MIN=0이면 색 필터 끔 — 위치가 주 기준이라 색은 벽·팔·그리퍼만 거름) · `SEG_BOX_MIN_W`/`MIN_H`/`MAX`(0.05/0.03/1.2 m, 실측 앞면 크기 상하한 — 점 노이즈·벽 크기 마스크 제외, MIN_W=0이면 끔) · `SEG_MIN_AREA`(0.002, 후보 bbox 최소 화면비) · `SEG_CENTER_FRAC`(0.5, 후보 중심이 화면 가로 가운데 50% 안이어야 함 — 박스는 항상 정면, 0이면 끔) · `SEG_HOLD_SELF`/`SEG_HOLD_RATIO`(0.7/0.85) · `SEG_IMU_TOPIC`/`SEG_JOINTS_TOPIC` · `SEG_COORDS_JSON` · `SEG_URDF` · `ROBOT_NAME`/`ZENOH_CONFIG`(zenoh 접속)

## Targeting System 모드 (2026-11-18)

- **SEG** (기본): 위의 상자 세그멘테이션 그대로.
- **EDGE**: 왼손 그리퍼가 박스를 쥐었고(KORAS 모터 전류 > 500 mA, 모터 각도 500~5500° — 실측 쥠 815 mA/3414°, 빈손 닫힘 58 mA/6135°, 열림 279 mA/284°) 그리퍼 x축이 수직 ±15° 안(place 자세 0.4°)으로 0.5 s 유지되면 전환. FastSAM 정지, 1.5 s마다 nano HD1200 스냅샷(TCP 55704) → `place_edges/edges.py`로 H5(띠 윗변)·V2(띠 오른쪽)·D(꼭짓점→뒤 윗모서리)·V3(뒤 끝)·K(D×V3 교점) → 화면 + `logs/place_target.json`(최근 5개 중 K 중앙값에 가장 가까운 결과). 검색 영역은 기준 꼭짓점(base 좌표)을 현재 카메라로 투영해 따라감. V2는 LSD가 조각만 주는 경우가 많아 조각에서 위아래로 띠→면 경계를 따라가 전체 길이로 늘린 뒤 H5를 찾고, 기울기(lean)는 이 V2로 계산(사용자: V2가 더 정확, 실측 11.4~11.8°). **박스 중심점** `center`(base, m): B1(V3 발치 오른쪽 바닥선) → 바닥 높이(밝은 바닥 띠 depth 중앙값 — 최근접값은 옆 흰 팔을 집음) → Vb = V3×B1을 바닥 평면에 투영 → 옆면 대각선 C..Vb 중점 + H5 두께/2 만큼 면 안쪽, 최근 5개 중앙값(실측 (1.289, 0.370, 0.864) ±2 mm). B1이 팔에 가리면 `center: null` + `center_why`. 그리퍼가 열리거나 기울면 SEG로 복귀. 관절 전류는 정지 마찰 때문에 같은 자세·빈손에서도 −4.6/−10.8로 갈려 쥠 판정에서 제외.
- 쥠 판정에는 robot_controller가 robot/joints에 싣는 `hands` 상태가 필요(omniteleop `robot_controller.py`, teleop Start 시 반영). 오른쪽 KORAS는 응답이 없어 `TGT_SIDE=left` 고정.
- 환경변수: `TGT_SIDE`(left) · `TGT_HOLD_MA`(500) · `TGT_POS_MIN`/`MAX`(500/5500) · `TGT_ALIGN_DEG`(15) · `TGT_SWITCH_S`(0.5) · `TGT_EDGE_S`(1.5) · `TGT_HIRES_SRC`(192.168.50.22:55704) · `TGT_ANCHOR`(1.124,0.409,1.055) · `TGT_PLACE_JSON`
- 자체 점검: `python place_edges/targeting.py`, `python place_edges/edges.py <hires.png·depth_hi.npy 있는 폴더>`

## 한계

- 색 게이트 수치는 현재 조명에서 측정한 값 — 조명이 바뀌면 `SEG_BROWN_*` 재조정.
- 수직 평면 방식은 더미가 똑바로 서 있다고 가정한다. 로봇이 들어서 기울인 묶음에는 맞지 않는다.
- 로봇이 움직이는 동안은 IMU에 가속이 섞여 중력 방향이 잠깐 틀어진다(0.5초 평균). 멈춘 뒤 값을 쓴다.
- 바닥 기준 절대 높이는 FK(관절값·URDF) 오차를 그대로 받는다: 1.3~2.6 cm 낮게 나온다.
- 좌표 갱신 2초 주기, 실측 depth는 약 6 Hz.
- nano 송출기는 1개만: 중복 실행 시 두 번째가 카메라를 뺏고 첫 번째는 고장 상태로 옛 depth를 계속 준다(2026-11-12). 송출기에 포트 락·grab 15초 실패 시 종료를 넣었고, seg_overlay는 접속 첫 depth 프레임을 버린다.
- 백엔드 reach(`/control/reach`)는 아직 표시용 depth · 설정 내참 · Isaac Lab 장착 보정을 그대로 쓴다.
