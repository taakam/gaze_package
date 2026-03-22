#!/usr/bin/env python3
from __future__ import annotations

import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped, PoseStamped

from moveit_commander import MoveGroupCommander, RobotCommander, PlanningSceneInterface


def quat_from_rpy(roll: float, pitch: float, yaw: float):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


class MoveItTargetListener(Node):
    def __init__(self):
        super().__init__("moveit_target_listener")

        self.robot = RobotCommander()
        self.scene = PlanningSceneInterface()
        self.group = MoveGroupCommander("ur_manipulator")

        # Tuning
        self.group.set_pose_reference_frame("base_link")
        self.group.set_end_effector_link("tool0")
        self.group.set_max_velocity_scaling_factor(0.2)
        self.group.set_max_acceleration_scaling_factor(0.2)
        self.group.set_goal_position_tolerance(0.01)
        self.group.set_goal_orientation_tolerance(0.05)
        self.group.set_planning_time(5.0)
        self.group.set_num_planning_attempts(5)

        self.sub = self.create_subscription(
            PointStamped,
            "/gaze_target_point",
            self.cb,
            10,
        )

        self.get_logger().info("Listening on /gaze_target_point")

    def cb(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y
        z = msg.point.z

        self.get_logger().info(
            f"Received target in {msg.header.frame_id}: "
            f"[{x:+.3f}, {y:+.3f}, {z:+.3f}]"
        )

        # Fixed downward-facing tool orientation
        # Adjust later if needed.
        qx, qy, qz, qw = quat_from_rpy(math.pi, 0.0, 0.0)

        pose = PoseStamped()
        pose.header.frame_id = "base_link"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.x = qx
        pose.pose.orientation.y = qy
        pose.pose.orientation.z = qz
        pose.pose.orientation.w = qw

        try:
            self.group.set_pose_target(pose, end_effector_link="tool0")
            success = self.group.go(wait=True)
            self.group.stop()
            self.group.clear_pose_targets()

            if success:
                self.get_logger().info("MoveIt execution succeeded.")
            else:
                self.get_logger().warn("MoveIt planning/execution failed.")
        except Exception as e:
            self.group.stop()
            self.group.clear_pose_targets()
            self.get_logger().error(f"MoveIt exception: {e}")


def main():
    rclpy.init()
    node = MoveItTargetListener()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()