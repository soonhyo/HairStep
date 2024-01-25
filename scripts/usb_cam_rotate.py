#!/usr/bin/env python
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

class ImageRotator:
    def __init__(self):
        self.node_name = "image_rotator"

        rospy.init_node(self.node_name)

        # Initialize a CvBridge to convert ROS image messages to OpenCV images
        self.bridge = CvBridge()

        # Subscribe to the usb_cam image topic
        self.image_sub = rospy.Subscriber("/camera/color/image_rect_color", Image, self.callback)

        # Publisher for the rotated image
        self.image_pub = rospy.Publisher("/camera/color/image_rect_color/rotated_image", Image, queue_size=10)

    def callback(self, data):
        try:
            # Convert the ROS image message to an OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        # Rotate the image 90 degrees counterclockwise
        rotated_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
        rotated_image = cv2.flip(rotated_image, 1)

        try:
            # Convert the OpenCV image back to a ROS image message and publish it
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(rotated_image, "bgr8"))
        except CvBridgeError as e:
            rospy.logerr(e)

if __name__ == '__main__':
    ImageRotator()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
