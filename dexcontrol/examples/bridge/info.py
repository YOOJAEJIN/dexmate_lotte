from dexbot_utils import RobotInfo

cfg = RobotInfo().config
print("robot_model:", RobotInfo().robot_model)
print("robot_name:", RobotInfo().robot_name)
print("config:", RobotInfo().config)
print("querables keys:", list(cfg.querables.keys()))
print("hand_info service_name:", cfg.querables["hand_info"])