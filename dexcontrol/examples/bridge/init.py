# Copyright (C) 2025 Dexmate Inc.
#
# This software is dual-licensed:
#
# 1. GNU Affero General Public License v3.0 (AGPL-3.0)
#    See LICENSE-AGPL for details
#
# 2. Commercial License
#    For commercial licensing terms, contact: contact@dexmate.ai

"""Example script to control robot head movements.

This script demonstrates basic head control by moving to zero position and then
moving each joint individually.
"""
from loguru import logger

from dexcontrol.robot import Robot

def main(
        step_size: float = 0.5,
) -> None:

    robot = None
    try:
        robot = Robot()


    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        pass

    logger.info(f"robot: {robot}")

if __name__ == "__main__":
    main()
