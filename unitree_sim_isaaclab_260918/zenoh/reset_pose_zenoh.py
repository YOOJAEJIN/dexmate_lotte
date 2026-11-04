# Copyright (c) 2025, Unitree Robotics Co., Ltd. All Rights Reserved.
# License: Apache License, Version 2.0  
"""
Reset pose Zenoh communication class
Specialized in receiving the reset pose command via Zenoh
"""

import threading
from typing import Any, Dict, Optional
import dexcomm
from dexcomm.codecs import DictDataCodec
from dds.dds_base import DDSObject


class ResetPoseCmdZenoh(DDSObject):
    """Reset pose command Zenoh node (singleton pattern)"""
    
    def __init__(self, node_name: str = "reset_pose_zenoh"):
        """Initialize the reset pose Zenoh node"""
        # avoid duplicate initialization
        if hasattr(self, '_initialized'):
            return
        super().__init__()
        
        self._initialized = True
        self.node_name = node_name
        self.zenoh_node = None
        self.subscriber = None
        
        # setup the shared memory
        self.setup_shared_memory(
            output_shm_name="isaac_reset_pose_cmd", 
            output_size=512, 
            outputshm_flag=True,
            inputshm_flag=False,
        )
        print(f"[{self.node_name}] Reset pose Zenoh node initialized")

    def setup_publisher(self) -> bool:
        """Setup the reset pose command publisher (optional)"""
        pass
    
    def setup_subscriber(self) -> bool:
        """Setup the reset pose command subscriber"""
        try:
            if self.zenoh_node is None:
                self.zenoh_node = dexcomm.Node(name=self.node_name)
                
            self.subscriber = self.zenoh_node.create_subscriber(
                "reset_pose/cmd", 
                self.zenoh_subscriber_callback, 
                decoder=DictDataCodec.decode
            )
            
            print(f"[{self.node_name}] Reset pose command Zenoh subscriber initialized on topic: reset_pose/cmd")
            return True
        except Exception as e:
            print(f"reset_pose_zenoh [{self.node_name}] Failed to initialize the reset pose command subscriber: {e}")
            return False
    
    def dds_publisher(self) -> Any:
        """Process the publish data (optional)"""
        pass
    
    def dds_subscriber(self, msg: Any, datatype: str = None) -> None:
        """Placeholder for abstract method from DDSObject"""
        pass

    def zenoh_subscriber_callback(self, msg: Any):
        """Process the subscribe data from Zenoh"""
        try:
            # Assuming msg is decoded as a dict like {"reset_category": 1}
            # or directly the value if the publisher sent it differently
            reset_category = None
            
            if isinstance(msg, dict):
                reset_category = msg.get("reset_category")
            else:
                reset_category = msg
                
            if reset_category is not None:
                cmd_data = {
                    "reset_category": reset_category
                }
                if self.output_shm:
                    self.output_shm.write_data(cmd_data)
        except Exception as e:
            print(f"reset_pose_zenoh [{self.node_name}] Failed to process the subscribe data: {e}")

    def get_reset_pose_command(self) -> Optional[Dict[str, Any]]:
        """Get the reset pose command
        
        Returns:
            Dict: the reset pose command, if no command return None
        """
        if self.output_shm:
            return self.output_shm.read_data()
        return None
    
    def write_reset_pose_command(self, flag_category):
        """Write the reset pose command to the shared memory
        
        Args:
            flag_category: the reset pose command
        """
        try:
            # prepare the reset pose data
            cmd_data = {
                "reset_category": flag_category
            }
            
            # write the reset pose data to the shared memory
            if self.output_shm:
                self.output_shm.write_data(cmd_data)
                
        except Exception as e:
            print(f"reset_pose_zenoh [{self.node_name}] Failed to write the reset pose command: {e}")

    def stop_communication(self):
        super().stop_communication()
        if self.zenoh_node:
            try:
                if hasattr(self.zenoh_node, 'close'):
                    self.zenoh_node.close()
                elif hasattr(self.zenoh_node, 'destroy'):
                    self.zenoh_node.destroy()
            except Exception:
                pass
