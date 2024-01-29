import cv2
import numpy as np
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
from mediapipe.tasks import python

class MP(object):
    def __init__(self):
        base_options = python.BaseOptions(model_asset_path='face_landmarker_v2_with_blendshapes.task')
        options = vision.FaceLandmarkerOptions(base_options=base_options,
                                               output_face_blendshapes=True,
                                               output_facial_transformation_matrixes=True,
                                               num_faces=1)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.face_landmarks_list = None
        self.face_landmarks_np = []

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        self.face_landmarks_list = detection_result.face_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected faces to visualize.
        for idx in range(len(self.face_landmarks_list)):
            face_landmarks = self.face_landmarks_list[idx]

            # Draw the face landmarks.
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_iris_connections_style())
        return annotated_image

    def run(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        out = self.detector.detect(mp_image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if out is None:
            return [0]
        else:
            return out

    def get_landmarks(self):
        if len(self.face_landmarks_list) > 0:
            self.face_landmarks_np = np.asarray([[landmark.x, landmark.y] for landmark in self.face_landmarks_list[0]])
        return self.face_landmarks_list

def image_callback(data):
    try:
        # 이미지 메시지를 OpenCV 이미지로 변환
        bridge = CvBridge()
        cv_image = bridge.imgmsg_to_cv2(data, "bgr8")

        # 얼굴 감지 및 랜드마크 그리기
        result_image = face_detector.run(cv_image)

        # 이미지 퍼블리시
        pub.publish(bridge.cv2_to_imgmsg(result_image, "bgr8"))
    except CvBridgeError as e:
        print(e)

if __name__=='__main__':
    rospy.init_node('face_landmark_node')
    face_detector = MP()

    # 이미지 토픽을 구독하고 콜백 함수 등록
    image_topic = "/segmented_image"  # 여기에 실제 이미지 토픽을 입력하세요
    rospy.Subscriber(image_topic, Image, image_callback)

    # 이미지 퍼블리셔 설정
    pub = rospy.Publisher("/output/image/topic", Image, queue_size=1)  # 출력 이미지 토픽 설정

    rospy.spin()
