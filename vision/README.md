# vision — 상자 더미 세그멘테이션 · 3D 좌표 · 파지점 (HEAD SEG)

헤드 ZED X Mini 영상에서 골판지 상자 더미만 분할하고, 실측 depth와 로봇 FK로
**베이스 좌표계** 기준 꼭짓점·층 경계선(6.1 cm = 20장)·파지점을 계산해
omniteleop GUI의 `HEAD SEG` 타일(WebRTC 60006)에 실시간으로 띄운다.

```
nano  gstsender.py ──H.264 RTP──▶ Thor webrtc_launch.py 60001 ──zmq 55601──┐
      (ZED X Mini)  └─TCP 55703 (metric depth, uint16 mm)─────────────────┤
Thor  omniteleop backend :5006  GET /robots/joints ────────────────────────┤
                                                                         seg_overlay.py
                                                                           │ FastSAM-s + FK
      GUI HEAD SEG ◀── webrtc_launch.py 60006 ◀──H.264 RTP udp 50006 ─────┘
```

## 파일

| 파일 | 실행 위치 | 역할 |
|---|---|---|
| `seg_overlay.py` | Thor (GPU) | 세그멘테이션 · 3D/베이스 좌표 · 오버레이 송출 |
| `webrtc_launch.py` | Thor | UDP H.264 → WebRTC 릴레이. `SEG_TAP_PORT` 설정 시 디코드 프레임을 zmq로 공개 |
| `nano/gstsender.py` | nano (`~/gstsender/`) | ZED 송출. 기존 3 스트림 + **실측 depth TCP 55703** 추가 |

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

경로 기본값은 vega-1p 로봇 기준이며 `NANO`, `NANO_PY`, `NANO_SENDER`, `RELAY_PY`, `SEG_PY` 환경변수로 바꾼다.
nano 쪽 `gstsender.py`는 `vision/nano/gstsender.py`를 nano의 `~/gstsender/`에 둔다.

수동으로 다룰 때 주의:
- nano 송출기는 SIGINT를 무시하므로 종료는 `kill <pid>`(SIGTERM). `pgrep -f gstsender`는 ssh로 띄울 때 생긴 `bash -c` 래퍼도 잡으니, 실제 python pid인지 확인한다.
- 릴레이의 `SEG_TAP_PORT`는 60001에만 준다. 모든 포트에 같은 값을 주면 bind 충돌로 탭이 동작하지 않는다.
- 릴레이 인증서: 기본 `/home/dexmate/certs/{cert,key}.pem`, 또는 `--cert/--key`, `XR_TELEOP_CERT/XR_TELEOP_KEY`.
- GUI를 릴레이와 다른 PC에서 띄우면 `CAM_HOST=<로봇 IP>` (vite가 `/cam/<port>/`로 중계하므로 브라우저 인증서 수락 불필요).
- seg_overlay는 torch(CUDA) · ultralytics · opencv · pyzmq · pinocchio(`pip install pin`) 필요. `FastSAM-s.pt`는 없으면 자동 다운로드.
- GUI에서 teleop **Start** 중이면 좌표가 베이스(B), 아니면 카메라(C) 기준 — 관절값을 teleop 스택의 `robot_controller`가 publish하기 때문.

결과는 화면 오버레이와 `vision/logs/box_coords.json`(2초마다: 상자별 bbox · corners · layers · grasp)로 나온다.

## 처리 방법

1. **세그멘테이션 — FastSAM-s**: SAM의 everything 출력을 흉내 내도록 학습된 YOLOv8-seg 계열 CNN. Thor 640×480 기준 13 ms.
   (yolo11n-seg 7 ms는 COCO 클래스라 골판지 없음, SAM2.1-tiny everything 6100 ms라 실시간 불가.)
   2초마다 전체 화면 스캔(좌표 계산), 매 프레임 줌 화면 재분할(마스크 표시).
2. **박스 판별**: 면적 0.5~35% · 화면 85% 이상 가로지르지 않음 · 테두리 2변 이상 접촉 안 함 · depth 0.3~1.6 m ·
   **채도 기반 갈색 게이트**(H 5~30, S≥95 픽셀 40% 이상 — 이 조명에서 Hue는 상자/크레이트 모두 ~16이라 채도로 구분) ·
   가장 큰 후보에서 인접한 것만 묶는 클러스터링.
3. **포커스 줌**: 더미 끝점 → 4% 여백 → 4:3 맞춤 → EMA(0.7/0.3) → bicubic 확대.
4. **Depth**: ZED SDK 스테레오(NEURAL)의 `retrieve_measure(DEPTH)`를 mm로 받아 9×9 중앙값.
   60003의 `VIEW.DEPTH`는 표시용 렌더라 장면에 따라 스케일이 바뀌어(같은 거리에서 픽셀 193→58) 쓰지 않는다.
5. **카메라 모델**: 핀홀, ZED 공장 캘리브(SN50691325, FHD1200)를 640×480으로 환산 — fx 245.1, fy 294.1, cx 330.9, cy 247.6.
   1920×1200→640×480 리사이즈가 비균일(가로 1/3, 세로 0.4)이라 fx≠fy.
6. **꼭짓점**: 마스크 볼록껍질의 대각선 극점 4개. `minAreaRect`는 원근 사다리꼴을 직사각형으로 펴서 먼 아랫변을 과대 계산한다.
7. **층 경계선 / 파지점**: 좌우 세로 모서리를 3D에서 위부터 `SEG_PITCH`(0.061 m)씩 따라 내려가 재투영. GRASP = 윗변~L1 중간(맨 위 20장 묶음 중심).
8. **베이스 좌표**: pinocchio로 `vega_1p_gripper.urdf`의 `zed_left_camera`까지 FK, 관절값은 `GET /robots/joints`.
   장착 보정 0 — reach 설정의 `camera_mount_rpy_corr`(Isaac Lab용)는 실기에서 수직 모서리를 20° 기울인다.
   베이스 원점은 바닥 위 5.48 cm(구동휠 중심 3.25 cm + 반지름 8.73 cm).

## 정확도 (실측 대비, 거리 약 1.2 m)

| 항목 | 계산 | 실측 |
|---|---|---|
| 폭 윗변/아랫변 | 41.1 / 40.5 cm | 40.5 / 40.5 cm |
| 높이 | 44.5~44.8 cm | 45.5 cm |
| 바닥→윗면 | 136.2 cm | 137.5 cm |
| 모서리 수직도 | 1~2° | 0° |

## 환경변수 (seg_overlay.py)

`SEG_TAP`(tcp://127.0.0.1:55601) · `SEG_DEPTH_SRC`(192.168.50.22:55703) · `SEG_DEST`/`SEG_PORT`(127.0.0.1/50006) ·
`SEG_MODEL`(FastSAM-s.pt) · `SEG_CONF`(0.4) · `SEG_SCAN_S`(2.0) · `SEG_PITCH`(0.061) ·
`SEG_BROWN_H0`/`H1`/`S`/`V`/`MIN`(5/30/95/50/0.4, MIN=0이면 색 필터 끔) · `SEG_JOINTS_URL` · `SEG_COORDS_JSON` · `SEG_URDF`

## 한계

- 색 게이트 수치는 현재 조명에서 측정한 값 — 조명이 바뀌면 `SEG_BROWN_*` 재조정.
- 층 격자는 검출된 윗변에 고정 — 윗면이 보일 만큼 내려다보면 전체가 함께 밀린다.
- 좌표 갱신 2초 주기, 실측 depth는 약 6 Hz.
- 백엔드 reach(`/control/reach`)는 아직 표시용 depth · 설정 내참 · Isaac Lab 장착 보정을 그대로 쓴다.
