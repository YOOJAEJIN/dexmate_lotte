* submodule 설치: dexmate-urdf 를 external 폴더에 설치한다.
git submodule update --init --recursive

* 설치
처음 설치시 or pyproject.toml에 변경 사항이 있다면, ()는 optional 사항
uv sync
uv sync --extra dev

만일 가상환경을 완전히 지우고 새로 생성하고 싶다면, (--python 3.12) 은 특정한 파이썬 버전을 설치시 사용
uv venv (--python 3.12) --clear
uv sync
uv sync --extra dev

특정 버전의 모듈을 설치하고자 한다면
uv lock --upgrade-package dexcontrol==0.4.9
uv sync
uv sync --extra dev

* 프런트엔드 설정
- 심볼릭 링크 추가 (생략해도 무방. 프런트엔드에 로봇 3d 화면이 안나옴)
$ ln -s /home/ubuntu/dex/omniteleop/external/dexmate-urdf/robots/humanoid/vega_1 /home/ubuntu/dex/omniteleop/src/omniteleop/app/frontend/public/assets/vega_1
$ ln -s /home/ubuntu/dex/omniteleop/external/dexmate-urdf/robots/humanoid/vega_1p /home/ubuntu/dex/omniteleop/src/omniteleop/app/frontend/public/assets/vega_1p
- frontend 폴더로 이동후  node_modules 폴더 삭제 후 npm install (node 버전 v20 이상 설치가 되어있어야 한다.)
$ cd ./src/omniteleop/app/frontend
$ rm -rf node_modules
$ npm install

* 실행: 백엔드와 프런트엔드를 동시에 실행하는 스크립트 실행
$ source .venv/bin/activate
$ export ROBOT_NAME=dm/vg1030101482-1p
$ export ROBOT_CONFIG=vega_1p
$ export ZENOH_CONFIG=/home/ubuntu/.dexmate/comm/zenoh/zenoh_config.json5
$ .src/omniteleop/app/launch.sh