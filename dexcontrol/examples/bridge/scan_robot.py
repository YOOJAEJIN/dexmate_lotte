import os
import zenoh
import time
from loguru import logger


def main():
    # peer 모드에서는 특정 endpoint로 "직접 connect"를 고정하기보다,
    # ZENOH_CONFIG(예: {"mode":"peer"}) 기반으로 피어 디스커버리를 타는 게 일반적입니다.
    zenoh_config_path = os.environ.get("ZENOH_CONFIG")

    if zenoh_config_path:
        logger.info(f"ZENOH_CONFIG 사용: {zenoh_config_path}")
        conf = zenoh.Config.from_file(zenoh_config_path)
    else:
        logger.warning("ZENOH_CONFIG가 설정되지 않았습니다. 기본 peer 설정으로 진행합니다.")
        conf = zenoh.Config()
        conf.insert_json5("mode", '"peer"')

    session = None
    try:
        logger.info("Zenoh(peer) 세션 오픈 시도 중...")
        # logger.info(f"config: {conf}")
        session = zenoh.open(conf)

        zid = session.info.zid()
        logger.info(f"연결 성공! Local Zenoh ID: {zid}")

        print("\n" + "=" * 60)
        print("실시간 데이터 감시 중 (10초간 대기)...")
        print("네트워크 상에서 발행되는 토픽(키)이 있다면 즉시 추출합니다.")
        print("=" * 60)

        found_keys = set()

        def listener(sample):
            key = str(sample.key_expr)
            found_keys.add(key)

        # 모든 토픽(**)에 대해 구독(Subscriber) 설정
        sub = session.declare_subscriber("**", listener)

        # 10초 동안 데이터를 수집합니다.
        for i in range(10):
            print(f"감시 중... ({10 - i}초 남음) | 발견된 토픽 수: {len(found_keys)}", end="\r")
            time.sleep(1)

        print("\n" + "=" * 60)
        if found_keys:
            print("발견된 토픽 목록:")
            for k in sorted(found_keys):
                print(f"- {k}")
        else:
            print("10초 동안 아무 토픽도 감지되지 않았습니다.")
            print("\n체크 포인트:")
            print("1) 동일 L2 네트워크에서 피어 디스커버리가 가능한지(멀티캐스트/방화벽) 확인")
            print("2) 상대 피어가 실제로 publish 중인지 확인 (publish가 없으면 subscribe로는 '목록'이 안 보입니다)")
        print("=" * 60)

        _ = sub  # lint 방지용(필요 없으면 제거 가능)

    except Exception as e:
        logger.error(f"오류 발생: {e}")
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    main()