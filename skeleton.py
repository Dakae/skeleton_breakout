# 여기에 미디어파이프를 활용한 스켈레톤 코드가 작성이 됩니다.
# 손 랜드마크를 따로 그룹화, 포즈 랜드마크에서는 팔 랜드마크, 몸 랜드마크, 다리 랜드마크를 따로 그룹화합니다.(팔, 다리, 몸의 경우 연결되는 어깨와 같은 부분은 양쪽 그룹에 중복되어 존재할 수 있습니다. )
# 각 그룸별로 다른 색의 선으로 연결합니다.
# 이 코드의 카메라 출력과 사용자의 포즈 랜드마크는 다른 소스와 합쳐서 운용될 수 있습니다 이 점 고려하여 작성해주세요요

import cv2
import numpy as np
import os
import urllib.request
from typing import Optional, Tuple, List
from dataclasses import dataclass

# MediaPipe Tasks API (최신 버전)
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================== 모델 다운로드 ====================
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# 모델 다운로드 URL (Google 공식)
# 포즈 모델: lite < full < heavy (정확도 순)
MODEL_URLS = {
    "pose_landmarker.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
}


def download_model(model_name: str) -> str:
    """모델 파일 다운로드 (없으면 자동 다운로드)"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, model_name)
    
    if not os.path.exists(model_path):
        print(f"모델 다운로드 중: {model_name}...")
        url = MODEL_URLS.get(model_name)
        if url:
            urllib.request.urlretrieve(url, model_path)
            print(f"다운로드 완료: {model_path}")
        else:
            raise FileNotFoundError(f"모델 URL을 찾을 수 없습니다: {model_name}")
    
    return model_path


# ==================== 색상 정의 ====================
# BGR 형식 (OpenCV 기본)
class Colors:
    """각 랜드마크 그룹별 색상 정의"""
    # 포즈 랜드마크 색상
    LEFT_ARM = (255, 0, 0)       # 파란색 - 왼팔
    RIGHT_ARM = (0, 0, 255)      # 빨간색 - 오른팔
    BODY = (0, 255, 0)           # 초록색 - 몸통
    LEFT_LEG = (255, 255, 0)     # 청록색 - 왼다리
    RIGHT_LEG = (0, 255, 255)    # 노란색 - 오른다리
    
    # 손 랜드마크 색상
    LEFT_HAND = (255, 128, 0)    # 하늘색 - 왼손
    RIGHT_HAND = (0, 128, 255)   # 주황색 - 오른손
    
    # 랜드마크 포인트 색상
    LANDMARK_POINT = (255, 255, 255)  # 흰색


# ==================== 랜드마크 그룹 정의 ====================
class PoseLandmarkGroups:
    """포즈 랜드마크 그룹 정의 (MediaPipe Pose 인덱스 기반)"""
    
    # 왼팔: 어깨 - 팔꿈치 - 손목 - 손가락들
    LEFT_ARM_INDICES = [11, 13, 15, 17, 19, 21]
    LEFT_ARM_CONNECTIONS = [(11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19)]
    
    # 오른팔: 어깨 - 팔꿈치 - 손목 - 손가락들
    RIGHT_ARM_INDICES = [12, 14, 16, 18, 20, 22]
    RIGHT_ARM_CONNECTIONS = [(12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20)]
    
    # 몸통: 양 어깨 - 양 엉덩이 연결
    BODY_INDICES = [11, 12, 23, 24]
    BODY_CONNECTIONS = [(11, 12), (11, 23), (12, 24), (23, 24)]
    
    # 왼다리: 엉덩이 - 무릎 - 발목 - 발
    LEFT_LEG_INDICES = [23, 25, 27, 29, 31]
    LEFT_LEG_CONNECTIONS = [(23, 25), (25, 27), (27, 29), (27, 31), (29, 31)]
    
    # 오른다리: 엉덩이 - 무릎 - 발목 - 발
    RIGHT_LEG_INDICES = [24, 26, 28, 30, 32]
    RIGHT_LEG_CONNECTIONS = [(24, 26), (26, 28), (28, 30), (28, 32), (30, 32)]


class HandLandmarkGroups:
    """손 랜드마크 그룹 정의 (MediaPipe Hands 인덱스 기반)"""
    
    # 손바닥
    PALM_CONNECTIONS = [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)]
    
    # 엄지
    THUMB_CONNECTIONS = [(1, 2), (2, 3), (3, 4)]
    
    # 검지
    INDEX_FINGER_CONNECTIONS = [(5, 6), (6, 7), (7, 8)]
    
    # 중지
    MIDDLE_FINGER_CONNECTIONS = [(9, 10), (10, 11), (11, 12)]
    
    # 약지
    RING_FINGER_CONNECTIONS = [(13, 14), (14, 15), (15, 16)]
    
    # 새끼
    PINKY_CONNECTIONS = [(17, 18), (18, 19), (19, 20)]
    
    # 모든 손 연결
    ALL_CONNECTIONS = (PALM_CONNECTIONS + THUMB_CONNECTIONS + INDEX_FINGER_CONNECTIONS + 
                       MIDDLE_FINGER_CONNECTIONS + RING_FINGER_CONNECTIONS + PINKY_CONNECTIONS)


# ==================== 데이터 클래스 ====================
@dataclass
class SkeletonData:
    """스켈레톤 데이터를 저장하는 클래스 (다른 소스와 합칠 때 사용)"""
    # 포즈 랜드마크 (정규화된 좌표 0~1)
    pose_landmarks: Optional[List[Tuple[float, float, float]]] = None
    # 왼손 랜드마크
    left_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    # 오른손 랜드마크
    right_hand_landmarks: Optional[List[Tuple[float, float, float]]] = None
    # 프레임 크기
    frame_width: int = 0
    frame_height: int = 0
    
    def get_pixel_coords(self, landmarks: List[Tuple[float, float, float]], 
                         index: int) -> Optional[Tuple[int, int]]:
        """정규화된 좌표를 픽셀 좌표로 변환"""
        if landmarks is None or index >= len(landmarks):
            return None
        x, y, _ = landmarks[index]
        return (int(x * self.frame_width), int(y * self.frame_height))


# ==================== 스켈레톤 트래커 클래스 ====================
class SkeletonTracker:
    """MediaPipe Tasks API를 활용한 스켈레톤 추적 클래스"""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 enable_pose: bool = True,
                 enable_hands: bool = True):
        """
        Args:
            min_detection_confidence: 감지 신뢰도 임계값
            min_tracking_confidence: 추적 신뢰도 임계값
            enable_pose: 포즈 추적 활성화 여부
            enable_hands: 손 추적 활성화 여부
        """
        self.enable_pose = enable_pose
        self.enable_hands = enable_hands
        
        # 포즈 감지기 (Tasks API)
        if enable_pose:
            pose_model_path = download_model("pose_landmarker.task")
            pose_options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=pose_model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                num_poses=1
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
        else:
            self.pose_landmarker = None
            
        # 손 감지기 (Tasks API)
        if enable_hands:
            hand_model_path = download_model("hand_landmarker.task")
            hand_options = vision.HandLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_path=hand_model_path),
                running_mode=vision.RunningMode.IMAGE,
                min_hand_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
                num_hands=2
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
        else:
            self.hand_landmarker = None
        
        # 선 두께
        self.line_thickness = 2
        self.point_radius = 4
        
    def process_frame(self, frame: np.ndarray) -> SkeletonData:
        """프레임을 처리하여 스켈레톤 데이터 추출
        
        Args:
            frame: BGR 형식의 OpenCV 프레임
            
        Returns:
            SkeletonData: 추출된 스켈레톤 데이터
        """
        height, width = frame.shape[:2]
        skeleton_data = SkeletonData(frame_width=width, frame_height=height)
        
        # BGR -> RGB 변환 후 MediaPipe Image로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # 포즈 처리
        if self.pose_landmarker is not None:
            pose_result = self.pose_landmarker.detect(mp_image)
            if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0:
                skeleton_data.pose_landmarks = [
                    (lm.x, lm.y, lm.z) for lm in pose_result.pose_landmarks[0]
                ]
        
        # 손 처리
        if self.hand_landmarker is not None:
            hand_result = self.hand_landmarker.detect(mp_image)
            if hand_result.hand_landmarks and hand_result.handedness:
                for hand_landmarks, handedness in zip(
                    hand_result.hand_landmarks, 
                    hand_result.handedness
                ):
                    landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks]
                    # 손 좌우 판별 (카메라 미러링 고려)
                    label = handedness[0].category_name
                    if label == "Left":
                        skeleton_data.right_hand_landmarks = landmarks  # 미러링
                    else:
                        skeleton_data.left_hand_landmarks = landmarks   # 미러링
        
        return skeleton_data
    
    def draw_skeleton(self, frame: np.ndarray, skeleton_data: SkeletonData, 
                      draw_pose: bool = True, draw_hands: bool = True) -> np.ndarray:
        """스켈레톤을 프레임에 그리기
        
        Args:
            frame: 그릴 프레임
            skeleton_data: 스켈레톤 데이터
            draw_pose: 포즈 그리기 여부
            draw_hands: 손 그리기 여부
            
        Returns:
            스켈레톤이 그려진 프레임
        """
        output = frame.copy()
        
        # 포즈 그리기
        if draw_pose and skeleton_data.pose_landmarks:
            self._draw_pose(output, skeleton_data)
        
        # 손 그리기
        if draw_hands:
            if skeleton_data.left_hand_landmarks:
                self._draw_hand(output, skeleton_data, skeleton_data.left_hand_landmarks, 
                               Colors.LEFT_HAND)
            if skeleton_data.right_hand_landmarks:
                self._draw_hand(output, skeleton_data, skeleton_data.right_hand_landmarks, 
                               Colors.RIGHT_HAND)
        
        return output
    
    def _draw_pose(self, frame: np.ndarray, skeleton_data: SkeletonData):
        """포즈 랜드마크 그리기 (그룹별 다른 색상)"""
        landmarks = skeleton_data.pose_landmarks
        
        # 각 그룹별로 연결선 그리기
        groups = [
            (PoseLandmarkGroups.LEFT_ARM_CONNECTIONS, Colors.LEFT_ARM),
            (PoseLandmarkGroups.RIGHT_ARM_CONNECTIONS, Colors.RIGHT_ARM),
            (PoseLandmarkGroups.BODY_CONNECTIONS, Colors.BODY),
            (PoseLandmarkGroups.LEFT_LEG_CONNECTIONS, Colors.LEFT_LEG),
            (PoseLandmarkGroups.RIGHT_LEG_CONNECTIONS, Colors.RIGHT_LEG),
        ]
        
        for connections, color in groups:
            for start_idx, end_idx in connections:
                start_point = skeleton_data.get_pixel_coords(landmarks, start_idx)
                end_point = skeleton_data.get_pixel_coords(landmarks, end_idx)
                
                if start_point and end_point:
                    cv2.line(frame, start_point, end_point, color, self.line_thickness)
        
        # 랜드마크 포인트 그리기 (모든 포즈 포인트)
        all_indices = set(
            PoseLandmarkGroups.LEFT_ARM_INDICES + 
            PoseLandmarkGroups.RIGHT_ARM_INDICES + 
            PoseLandmarkGroups.BODY_INDICES + 
            PoseLandmarkGroups.LEFT_LEG_INDICES + 
            PoseLandmarkGroups.RIGHT_LEG_INDICES
        )
        
        for idx in all_indices:
            point = skeleton_data.get_pixel_coords(landmarks, idx)
            if point:
                cv2.circle(frame, point, self.point_radius, Colors.LANDMARK_POINT, -1)
    
    def _draw_hand(self, frame: np.ndarray, skeleton_data: SkeletonData,
                   hand_landmarks: List[Tuple[float, float, float]], color: Tuple[int, int, int]):
        """손 랜드마크 그리기"""
        # 연결선 그리기
        for start_idx, end_idx in HandLandmarkGroups.ALL_CONNECTIONS:
            start = skeleton_data.get_pixel_coords(hand_landmarks, start_idx)
            end = skeleton_data.get_pixel_coords(hand_landmarks, end_idx)
            
            if start and end:
                cv2.line(frame, start, end, color, self.line_thickness)
        
        # 포인트 그리기
        for idx in range(21):
            point = skeleton_data.get_pixel_coords(hand_landmarks, idx)
            if point:
                cv2.circle(frame, point, self.point_radius - 1, Colors.LANDMARK_POINT, -1)
    
    def release(self):
        """리소스 해제"""
        if self.pose_landmarker:
            self.pose_landmarker.close()
        if self.hand_landmarker:
            self.hand_landmarker.close()


# ==================== 카메라 관리 클래스 ====================
class CameraManager:
    """카메라 입력 관리 클래스"""
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Args:
            camera_id: 카메라 장치 ID
            width: 프레임 너비
            height: 프레임 높이
        """
        self.camera_id = camera_id
        self.cap = None
        self.width = width
        self.height = height
        
    def open(self) -> bool:
        """카메라 열기"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """프레임 읽기"""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def release(self):
        """카메라 해제"""
        if self.cap:
            self.cap.release()


# ==================== 메인 애플리케이션 ====================
class SkeletonApp:
    """스켈레톤 추적 애플리케이션"""
    
    def __init__(self, camera_id: int = 0, mirror: bool = True):
        """
        Args:
            camera_id: 카메라 장치 ID
            mirror: 좌우 반전 여부
        """
        self.camera = CameraManager(camera_id)
        self.tracker = SkeletonTracker()
        self.mirror = mirror
        self.running = False
        
    def run(self):
        """애플리케이션 실행"""
        if not self.camera.open():
            print("카메라를 열 수 없습니다.")
            return
        
        self.running = True
        print("스켈레톤 추적 시작... (종료: 'q' 또는 ESC)")
        print("색상 가이드:")
        print("  - 왼팔: 파란색 / 오른팔: 빨간색")
        print("  - 몸통: 초록색")
        print("  - 왼다리: 청록색 / 오른다리: 노란색")
        print("  - 왼손: 하늘색 / 오른손: 주황색")
        
        while self.running:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            # 좌우 반전 (미러 모드)
            if self.mirror:
                frame = cv2.flip(frame, 1)
            
            # 스켈레톤 추적
            skeleton_data = self.tracker.process_frame(frame)
            
            # 스켈레톤 그리기
            output_frame = self.tracker.draw_skeleton(frame, skeleton_data)
            
            # 화면 표시
            cv2.imshow('Skeleton Tracking', output_frame)
            
            # 키 입력 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 또는 ESC
                self.running = False
        
        self.cleanup()
    
    def cleanup(self):
        """리소스 정리"""
        self.camera.release()
        self.tracker.release()
        cv2.destroyAllWindows()


# ==================== 외부 연동용 함수 ====================
def create_tracker(enable_pose: bool = True, enable_hands: bool = True) -> SkeletonTracker:
    """외부에서 사용할 트래커 생성 함수
    
    다른 소스와 합쳐서 사용할 때 이 함수로 트래커를 생성하세요.
    
    Example:
        tracker = create_tracker()
        skeleton_data = tracker.process_frame(your_frame)
        # skeleton_data를 다른 곳에서 활용
    """
    return SkeletonTracker(enable_pose=enable_pose, enable_hands=enable_hands)


def get_skeleton_data(tracker: SkeletonTracker, frame: np.ndarray) -> SkeletonData:
    """프레임에서 스켈레톤 데이터 추출
    
    다른 소스와 합쳐서 사용할 때 이 함수로 데이터를 추출하세요.
    
    Returns:
        SkeletonData: pose_landmarks, left_hand_landmarks, right_hand_landmarks 포함
    """
    return tracker.process_frame(frame)


def draw_skeleton_on_frame(tracker: SkeletonTracker, frame: np.ndarray, 
                           skeleton_data: SkeletonData) -> np.ndarray:
    """프레임에 스켈레톤 그리기
    
    이미 추출된 skeleton_data를 사용하여 다른 프레임에 그릴 수 있습니다.
    """
    return tracker.draw_skeleton(frame, skeleton_data)


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    app = SkeletonApp(camera_id=0, mirror=True)
    app.run()
