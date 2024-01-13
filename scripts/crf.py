import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

class CRFSegmentationRefiner:
    def __init__(self, gt_prob=0.6, scale_factor=1.0):
        """
        CRF를 사용하여 세그멘테이션 결과를 보정하는 클래스를 초기화합니다.

        :param original_image: 원본 이미지 (높이, 너비, 채널)
        :param mask: 초기 세그멘테이션 마스크 (높이, 너비), 값은 0 또는 1
        :param gt_prob: 마스크의 Ground Truth 확률
        :param scale_factor: 이미지와 마스크의 크기 조정 비율
        """
        self.gt_prob = gt_prob
        self.scale_factor = scale_factor

    def refine(self, original_image, mask):
        """
        세그멘테이션 결과를 CRF를 사용하여 보정합니다.

        :return: 보정된 세그멘테이션 마스크 (높이, 너비)
        """
        # 이미지와 마스크의 크기를 조정
        if self.scale_factor != 1.0:
            resized_image = cv2.resize(original_image, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_LINEAR)
            resized_mask = cv2.resize(mask, None, fx=self.scale_factor, fy=self.scale_factor, interpolation=cv2.INTER_NEAREST)
        else:
            resized_image = original_image
            resized_mask = mask/255

        # CRF 모델 초기화
        h, w = resized_image.shape[:2]
        crf = dcrf.DenseCRF2D(w, h, 2)  # 두 레이블: 배경과 객체

        # Unary potential 설정
        unary = unary_from_labels(resized_mask.astype(np.uint8), 2, gt_prob=self.gt_prob, zero_unsure=False)
        crf.setUnaryEnergy(unary)

        # Pairwise potentials 추가
        crf.addPairwiseGaussian(sxy=(3, 3), compat=3)
        crf.addPairwiseBilateral(sxy=(5, 5), srgb=(13, 13, 13), rgbim=resized_image, compat=10)

        # CRF 최적화
        Q = crf.inference(5)

        # 결과 추출
        refined_mask = np.argmax(Q, axis=0).reshape((h, w)).astype(np.uint8) * 255

        return refined_mask

