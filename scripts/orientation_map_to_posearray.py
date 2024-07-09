#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose
import numpy as np
import cv2
from cv_bridge import CvBridge

class HairStrandOrientationNode:

    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('hair_strand_orientation_node')

        # Create a subscriber to the orientation image topic
        self.subscriber = rospy.Subscriber('orientation_image', Image, self.orientation_image_callback)

        # Create a publisher for the PoseArray
        self.publisher = rospy.Publisher('pose_array', PoseArray, queue_size=10)

        # CV Bridge for converting ROS Image messages to OpenCV images
        self.bridge = CvBridge()

    def orientation_image_callback(self, msg):
        # Convert ROS Image message to OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Assuming the orientation image encodes 3D direction vectors in RGB channels
        directions = self.extract_directions(cv_image)

        # Create PoseArray from direction vectors
        pose_array = self.create_pose_array(directions)

        # Publish PoseArray
        self.publisher.publish(pose_array)

    def extract_directions(self, image):
        # Normalize the image to get direction vectors
        directions = image.astype(np.float32) / 255.0
        directions = (directions - 0.5) * 2.0  # Map to range [-1, 1]
        return directions

    def create_pose_array(self, directions):
        pose_array = PoseArray()
        pose_array.header.stamp = rospy.Time.now()
        pose_array.header.frame_id = 'camera_color_optical_frame'

        for i in range(0, directions.shape[0], 10):  # Iterate with interval
            for j in range(0, directions.shape[1], 10):
                direction = directions[i, j]

                # Normalize the direction vector
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction = direction / norm

                # Create Pose
                pose = Pose()
                pose.position.x = i
                pose.position.y = j
                pose.position.z = 0.0
                pose.orientation.x = direction[0]
                pose.orientation.y = direction[1]
                pose.orientation.z = direction[2]
                pose.orientation.w = 1.0  # Assuming no rotation, only direction

                pose_array.poses.append(pose)

        return pose_array

if __name__ == '__main__':
    try:
        node = HairStrandOrientationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
