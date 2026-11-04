##### unitree_sim_lab 다운로드 및 가상환경 설치
# git clone
ubuntu@eastflag:~$ git clone https://github.com/unitreerobotics/unitree_sim_isaaclab.git
ubuntu@eastflag:~$ cd unitree_sim_isaaclab
ubuntu@eastflag:~/unitree_sim_isaaclab$ git submodule update --init --depth 1

# asset download
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ sudo apt install git-lfs
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ . fetch_assets.sh

# uv 가상환경을 만들고 activate

# pyproject.toml이 없으므로 uv sync를 할수 없고 빈공간만 만든다.
ubuntu@eastflag:~/unitree_sim_isaaclab$ uv venv --python=3.11
Using CPython 3.11.15
Creating virtual environment at: .venv
Activate with: source .venv/bin/activate

ubuntu@eastflag:~/unitree_sim_isaaclab$ source .venv/bin/activate
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$

# unitree_sdk2_python 설치.
(unitree_sim_isaaclab) ubuntu@eastflag:~/IsaacLab2$ cd ~/unitree/unitree_sdk2_python/
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree/unitree_sdk2_python$ uv pip install -e .
# 만일 Could not locate cyclonedds. 에러가 발생한다면 아래 페이지 FAQ를 참고해서 설치
# https://github.com/unitreerobotics/unitree_sdk2_python

##### isaac sim 설치: pip로 가상환경에 설치해도 무방하지만
# 여기서는 컴파일해서 설치했다고 가정하고 skip. 나중에 conda 환경만 source 하면 됨

##### Isaaclab 설치
# IsaacLab main 브랜치가 현재 3.0이므로 uv 가 지원되는 2.3 버전으로 헤더를 이동해서 설치한다.
# timeout 에러가 거의 나므로 미리 timeout 을 60으로 설정해둔다.
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ cd ~/IsaacLab/
(unitree_sim_isaaclab) ubuntu@eastflag:~/IsaacLab$ git reset --hard e10c302cd247eb98201067952fcd38979074fe3d
(unitree_sim_isaaclab) ubuntu@eastflag:~/IsaacLab$ uv pip install --upgrade pip
(unitree_sim_isaaclab) ubuntu@eastflag:~/IsaacLab$ export UV_HTTP_TIMEOUT=60
(unitree_sim_isaaclab) ubuntu@eastflag:~/IsaacLab$ ./isaaclab.sh -i


##### unitree_sim_isaaclab 모듈 설치
# requirements.txt  화일은 cns 브랜치에서 변경사항이 있으므로 반드시 cns 브랜치로 체크아웃해서 실행해야 함.
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ uv pip install -r requirements.txt

# git submodule인 teleimager 설치
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ cd teleimager/
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab/teleimager$ uv pip install -e .
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab/teleimager$ uv pip install -e ".[server]"

# dexcontrol, dextop 설치
(unitree_sim_isaaclab) ubuntu@eastflag-desktop:~/unitree_sim_isaaclab$ uv pip install dexcontrol
(unitree_sim_isaaclab) ubuntu@eastflag-desktop:~/unitree_sim_isaaclab$ uv pip install dextop

##### 설치 정상 유무 테스트
# isaac sim을 컴파일하여 설치 및 alias 등록을 가정
# alias loadsim="source ~/isaacsim/_build/linux-x86_64/release/setup_conda_env.sh; echo \"isaacsim conda loaded!!\""
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ loadsim
isaacsim conda loaded!!
(unitree_sim_isaaclab) ubuntu@eastflag:~/unitree_sim_isaaclab$ python sim_main.py --device cpu  --enable_cameras  --task Isaac-PickPlace-Cylinder-G129-Inspire-Joint    --enable_inspire_dds --robot_type g129


## 카메라 프로세스
각 task mdp > observations > get_camera_image
camera_state > get_camera_image
  InteractiveSceneCfg 을 상속한 config class에서 camera 생성. 예> CameraPresets.vega1p_front_camera()
  > 생성된 변수를 camera  key에 할당.  예> front_camera, left_wrist_camera, right_wrist_camera
  > camera 변수명을 찾아서 매핑
head_depth는 data_types에 distance_to_image_plane 추가
shared_memory_utils > 각 camera key 에서 매핑한 스트링으로 ["head", "head_right", "head_depth", "wrist_left", "wrist_right"] 로 데이터 write
image_server > cam_config_server.yaml > head_depth 키 추가
  각 camera key로 데이터 read 후 webrtc 생성

## S3 usd에 접근이 안될때
인터넷이 안되는 공간 혹은 회사 방화벽 정책으로 S3 usd에 접근이 안되는 경우 처리
먼저 ~/isaacsim_assets 폴더에 로컬 애셋 설정이 완료되어야 함.
/home/ubuntu/.local/share/ov/data/Kit/Isaac-Sim Full/5.1/user.config.json 에 cloud 1줄 추가
  "asset_root": {
    "timeout": 5.0,
    "default": "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.1",
+   "cloud": "/home/ubuntu/isaacsim_assets/Assets/Isaac/5.1"
  }
동작 원리: Isaac Lab이 /persistent/isaac/asset_root/cloud를 읽어 ISAAC_NUCLEUS_DIR를 만드는데,
이 키가 없어서 S3 URL 폴백이 일어났고, 로컬 에셋 경로를 넣어 해결한 것입니다.
