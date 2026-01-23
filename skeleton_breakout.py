# 스켈레톤 기반 벽돌깨기 게임
# skeleton.py와 breakOut.py의 기능을 활용하여 손 스켈레톤으로 공을 튕기는 게임
# 배경은 웹캠 화면이며, 손의 선분이 막대 역할을 합니다.

import cv2
import numpy as np
import pygame
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 기존 모듈에서 필요한 클래스 임포트 (손 인식만 사용, 포즈 비활성화)
from skeleton import (
    SkeletonTracker, 
    SkeletonData, 
    # HandLandmarkGroups,  # 직접 FingerGroup 사용으로 대체
    # Colors,              # 사용 안 함
    CameraManager,
    create_tracker
)

from breakOut import (
    Ball,
    Brick,
    GameConfig,
    BrickShape
)


# ==================== 손 부위 그룹 정의 ====================
class FingerGroup:
    """손 부위별 연결 그룹 정의 (손가락 5개 + 손바닥)"""
    # 그룹 인덱스
    THUMB = 0   # 엄지
    INDEX = 1   # 검지
    MIDDLE = 2  # 중지
    RING = 3    # 약지
    PINKY = 4   # 새끼
    PALM = 5    # 손바닥
    
    # 그룹별 연결
    CONNECTIONS = {
        THUMB: [(1, 2), (2, 3), (3, 4)],
        INDEX: [(5, 6), (6, 7), (7, 8)],
        MIDDLE: [(9, 10), (10, 11), (11, 12)],
        RING: [(13, 14), (14, 15), (15, 16)],
        PINKY: [(17, 18), (18, 19), (19, 20)],
        PALM: [(0, 1), (0, 5), (0, 17), (5, 9), (9, 13), (13, 17)],  # 손바닥 연결
    }
    
    # 그룹별 색상 (BGR)
    COLORS = {
        THUMB: (255, 0, 0),      # 파랑 - 엄지
        INDEX: (0, 255, 0),      # 초록 - 검지
        MIDDLE: (0, 255, 255),   # 노랑 - 중지
        RING: (0, 165, 255),     # 주황 - 약지
        PINKY: (255, 0, 255),    # 분홍 - 새끼
        PALM: (128, 128, 128),   # 회색 - 손바닥
    }
    
    # 그룹 이름
    NAMES = {
        THUMB: "Thumb",
        INDEX: "Index",
        MIDDLE: "Middle",
        RING: "Ring",
        PINKY: "Pinky",
        PALM: "Palm",
    }
    
    # 손가락 그룹만 (하드모드에서 그룹 제거 대상)
    FINGER_GROUPS = [THUMB, INDEX, MIDDLE, RING, PINKY]
    
    # 모든 그룹
    ALL_GROUPS = [THUMB, INDEX, MIDDLE, RING, PINKY, PALM]


# ==================== 선분 충돌 처리 클래스 ====================
@dataclass
class LineSegment:
    """선분을 나타내는 데이터 클래스"""
    x1: float
    y1: float
    x2: float
    y2: float
    # 신뢰도 (0~1, 랜드마크 visibility 기반)
    confidence: float = 1.0
    # 이전 프레임 대비 속도 (스핀 계산용)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    # 손가락 그룹 (0=엄지, 1=검지, 2=중지, 3=약지, 4=새끼)
    finger_group: int = -1
    # 왼손/오른손 (True=왼손, False=오른손)
    is_left_hand: bool = True
    
    def get_direction(self) -> Tuple[float, float]:
        """선분의 방향 벡터 반환 (정규화됨)"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        length = math.sqrt(dx * dx + dy * dy)
        if length > 0:
            return (dx / length, dy / length)
        return (0, 0)
    
    def get_normal(self) -> Tuple[float, float]:
        """
        선분의 법선 벡터 반환 (정규화됨)
        공이 위/왼쪽에서 오면 적절한 방향의 법선 반환
        """
        dx, dy = self.get_direction()
        # 기본적으로 왼쪽 법선 (-dy, dx)을 반환
        # 이는 선분을 기준으로 "위쪽" 방향
        return (-dy, dx)
    
    def get_length(self) -> float:
        """선분의 길이 반환"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return math.sqrt(dx * dx + dy * dy)
    
    def get_angle(self) -> float:
        """선분의 기울기 각도 반환 (도 단위)"""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return math.degrees(math.atan2(dy, dx))
    
    def get_center(self) -> Tuple[float, float]:
        """선분의 중심점 반환"""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


class SkeletonCollisionHandler:
    """스켈레톤 선분과 공의 충돌을 처리하는 클래스 (개선된 버전)"""
    
    # 설정값
    MIN_CONFIDENCE = 0.3  # 최소 신뢰도 임계값
    
    @staticmethod
    def point_to_segment_distance(px: float, py: float, seg: LineSegment) -> Tuple[float, float, float]:
        """
        점에서 선분까지의 최단 거리와 가장 가까운 점 좌표 반환
        
        Returns:
            (거리, 가장 가까운 점의 x, 가장 가까운 점의 y)
        """
        x1, y1, x2, y2 = seg.x1, seg.y1, seg.x2, seg.y2
        
        # 선분 벡터
        dx = x2 - x1
        dy = y2 - y1
        
        # 선분 길이의 제곱
        length_sq = dx * dx + dy * dy
        
        if length_sq == 0:
            # 선분이 점인 경우
            dist = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            return (dist, x1, y1)
        
        # 점을 선분에 투영한 위치 (0~1 범위로 정규화)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / length_sq))
        
        # 가장 가까운 점
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # 거리 계산
        dist = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
        
        return (dist, closest_x, closest_y)
    
    @staticmethod
    def segments_intersect(p1: Tuple[float, float], p2: Tuple[float, float],
                          p3: Tuple[float, float], p4: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """
        두 선분의 교차점 계산 (CCD용)
        
        Args:
            p1, p2: 첫 번째 선분의 양 끝점
            p3, p4: 두 번째 선분의 양 끝점
            
        Returns:
            교차점 좌표 또는 None
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None  # 평행
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            ix = x1 + t * (x2 - x1)
            iy = y1 + t * (y2 - y1)
            return (ix, iy)
        
        return None
    
    @staticmethod
    def check_ccd_collision(ball_prev_x: float, ball_prev_y: float,
                           ball_curr_x: float, ball_curr_y: float,
                           ball_radius: float, seg: LineSegment) -> Optional[Tuple[float, float, float]]:
        """
        연속 충돌 감지 (Continuous Collision Detection)
        공의 이동 경로와 선분의 교차 검사
        
        Returns:
            (충돌점 x, 충돌점 y, 충돌 시점 t) 또는 None
        """
        # 공의 이동 경로를 선분으로
        ball_path = ((ball_prev_x, ball_prev_y), (ball_curr_x, ball_curr_y))
        seg_line = ((seg.x1, seg.y1), (seg.x2, seg.y2))
        
        # 선분-선분 교차 검사
        intersection = SkeletonCollisionHandler.segments_intersect(
            ball_path[0], ball_path[1], seg_line[0], seg_line[1]
        )
        
        if intersection:
            # 충돌 시점 계산 (0~1)
            dx = ball_curr_x - ball_prev_x
            dy = ball_curr_y - ball_prev_y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                t = math.sqrt((intersection[0] - ball_prev_x) ** 2 + 
                             (intersection[1] - ball_prev_y) ** 2) / dist
                return (intersection[0], intersection[1], t)
        
        return None
    
    @staticmethod
    def check_ball_segment_collision_advanced(
        ball: Ball, 
        seg: LineSegment, 
        ball_prev_x: Optional[float] = None,
        ball_prev_y: Optional[float] = None,
        min_segment_length: float = 10.0
    ) -> bool:
        """
        개선된 공과 선분의 충돌 체크 및 반사 처리
        - 신뢰도 기반 필터링
        - CCD (연속 충돌 감지)
        
        Args:
            ball: 공 객체
            seg: 선분 객체
            ball_prev_x, ball_prev_y: 이전 프레임 공 위치 (CCD용)
            min_segment_length: 충돌 처리할 최소 선분 길이
            
        Returns:
            충돌 여부
        """
        if not ball.active:
            return False
        
        # 1. 신뢰도 체크 (랜드마크 신뢰도가 낮으면 무시)
        if seg.confidence < SkeletonCollisionHandler.MIN_CONFIDENCE:
            return False
        
        # 2. 너무 짧은 선분은 무시
        if seg.get_length() < min_segment_length:
            return False
        
        # 3. CCD 체크 (이전 위치가 제공된 경우)
        ccd_collision = None
        if ball_prev_x is not None and ball_prev_y is not None:
            ccd_collision = SkeletonCollisionHandler.check_ccd_collision(
                ball_prev_x, ball_prev_y, ball.x, ball.y, ball.radius, seg
            )
        
        # 4. 일반 거리 기반 충돌 체크
        dist, closest_x, closest_y = SkeletonCollisionHandler.point_to_segment_distance(
            ball.x, ball.y, seg
        )
        
        # 충돌 조건: CCD 충돌 또는 거리 기반 충돌
        collision_detected = ccd_collision is not None or dist < ball.radius + 8  # 충돌 반경 증가
        
        if collision_detected:
            # 공이 선분을 향해 이동 중인지 확인
            to_line_x = closest_x - ball.x
            to_line_y = closest_y - ball.y
            
            dot = ball.vx * to_line_x + ball.vy * to_line_y
            if dot <= 0 and ccd_collision is None:
                return False  # 공이 선분에서 멀어지는 중 (CCD 충돌이 아닌 경우)
            
            # 선분의 법선 벡터 계산
            nx, ny = seg.get_normal()
            
            # 공의 중심이 선분의 어느 쪽에 있는지 판단
            to_ball_x = ball.x - seg.x1
            to_ball_y = ball.y - seg.y1
            
            side = to_ball_x * nx + to_ball_y * ny
            if side < 0:
                nx, ny = -nx, -ny
            
            # 반사 공식: v' = v - 2(v·n)n
            dot_vn = ball.vx * nx + ball.vy * ny
            
            reflect_vx = ball.vx - 2 * dot_vn * nx
            reflect_vy = ball.vy - 2 * dot_vn * ny
            
            # 속도 크기 유지
            reflect_speed = math.sqrt(reflect_vx ** 2 + reflect_vy ** 2)
            if reflect_speed > 0:
                reflect_vx = reflect_vx / reflect_speed * ball.speed
                reflect_vy = reflect_vy / reflect_speed * ball.speed
            
            ball.vx = reflect_vx
            ball.vy = reflect_vy
            
            # 공을 선분 밖으로 밀어내기
            push_dist = ball.radius + 5 - dist
            if push_dist > 0:
                ball.x += nx * push_dist
                ball.y += ny * push_dist
            
            return True
        
        return False
    
    @staticmethod
    def check_ball_segment_collision(ball: Ball, seg: LineSegment, min_segment_length: float = 10.0) -> bool:
        """기존 호환성을 위한 래퍼 메서드"""
        return SkeletonCollisionHandler.check_ball_segment_collision_advanced(
            ball, seg, None, None, None, min_segment_length
        )


# ==================== 손 스켈레톤 선분 추출 (개선된 버전 - 손가락별 그룹화) ====================
class HandSkeletonExtractor:
    """손 랜드마크에서 충돌 가능한 선분들을 추출하는 클래스 (손가락별 그룹화, 스무딩, 신뢰도, 속도 계산 포함)"""
    
    def __init__(self, smoothing_factor: float = 0.3):
        """
        Args:
            smoothing_factor: 스무딩 계수 (0=스무딩 없음, 1=이전 값만 사용)
        """
        # 손가락별 연결 그룹 사용 (손바닥 연결은 제외, 손가락만)
        self.finger_connections = FingerGroup.CONNECTIONS
        
        # 스무딩 설정
        self.smoothing_factor = smoothing_factor
        
        # 이전 프레임 랜드마크 저장 (스무딩 및 속도 계산용)
        self.prev_left_landmarks: Optional[List[Tuple[float, float]]] = None
        self.prev_right_landmarks: Optional[List[Tuple[float, float]]] = None
        
        # 스무딩된 랜드마크
        self.smoothed_left_landmarks: Optional[List[Tuple[float, float]]] = None
        self.smoothed_right_landmarks: Optional[List[Tuple[float, float]]] = None
    
    def _smooth_landmarks(self, current: List[Tuple[float, float]], 
                         previous: Optional[List[Tuple[float, float]]],
                         smoothed: Optional[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
        """
        랜드마크 좌표 스무딩 (Exponential Moving Average)
        """
        if previous is None or smoothed is None:
            return current
        
        result = []
        alpha = self.smoothing_factor
        
        for i, (cx, cy) in enumerate(current):
            if i < len(smoothed):
                sx, sy = smoothed[i]
                # EMA: new = alpha * old + (1 - alpha) * current
                new_x = alpha * sx + (1 - alpha) * cx
                new_y = alpha * sy + (1 - alpha) * cy
                result.append((new_x, new_y))
            else:
                result.append((cx, cy))
        
        return result
    
    def _calculate_velocities(self, current: List[Tuple[float, float]],
                             previous: Optional[List[Tuple[float, float]]]) -> List[Tuple[float, float]]:
        """
        랜드마크 속도 계산 (이전 프레임과의 차이)
        """
        if previous is None:
            return [(0, 0)] * len(current)
        
        velocities = []
        for i, (cx, cy) in enumerate(current):
            if i < len(previous):
                px, py = previous[i]
                velocities.append((cx - px, cy - py))
            else:
                velocities.append((0, 0))
        
        return velocities
    
    def extract_segments_advanced(self, skeleton_data: SkeletonData, 
                                  hand_landmarks: List[Tuple[float, float, float]],
                                  is_left_hand: bool = True) -> List[LineSegment]:
        """
        손 랜드마크에서 선분 목록 추출 (스무딩, 신뢰도, 속도 포함)
        
        Args:
            skeleton_data: 스켈레톤 데이터 (좌표 변환용)
            hand_landmarks: 손 랜드마크 리스트 (x, y, z)
            is_left_hand: 왼손 여부
            
        Returns:
            선분 목록 (신뢰도, 속도 포함)
        """
        segments = []
        
        if hand_landmarks is None:
            return segments
        
        # 1. 픽셀 좌표 및 신뢰도 추출
        pixel_coords = []
        confidences = []
        for idx in range(len(hand_landmarks)):
            point = skeleton_data.get_pixel_coords(hand_landmarks, idx)
            if point:
                pixel_coords.append(point)
                # z 값을 신뢰도로 사용 (MediaPipe에서 z는 깊이, visibility가 따로 없으면 1.0)
                # 실제로는 visibility가 별도로 있지만 여기선 간단히 처리
                confidences.append(min(1.0, max(0.0, 1.0 - abs(hand_landmarks[idx][2]))))
            else:
                pixel_coords.append((0, 0))
                confidences.append(0.0)
        
        # 2. 스무딩 적용
        if is_left_hand:
            smoothed = self._smooth_landmarks(
                pixel_coords, self.prev_left_landmarks, self.smoothed_left_landmarks
            )
            velocities = self._calculate_velocities(pixel_coords, self.prev_left_landmarks)
            self.prev_left_landmarks = pixel_coords.copy()
            self.smoothed_left_landmarks = smoothed
        else:
            smoothed = self._smooth_landmarks(
                pixel_coords, self.prev_right_landmarks, self.smoothed_right_landmarks
            )
            velocities = self._calculate_velocities(pixel_coords, self.prev_right_landmarks)
            self.prev_right_landmarks = pixel_coords.copy()
            self.smoothed_right_landmarks = smoothed
        
        # 3. 선분 생성 (손가락별 그룹화)
        for finger_group, connections in self.finger_connections.items():
            for start_idx, end_idx in connections:
                if start_idx < len(smoothed) and end_idx < len(smoothed):
                    start = smoothed[start_idx]
                    end = smoothed[end_idx]
                    
                    # 평균 신뢰도
                    avg_confidence = (confidences[start_idx] + confidences[end_idx]) / 2
                    
                    # 평균 속도
                    avg_vel_x = (velocities[start_idx][0] + velocities[end_idx][0]) / 2
                    avg_vel_y = (velocities[start_idx][1] + velocities[end_idx][1]) / 2
                    
                    if start != (0, 0) and end != (0, 0):
                        segments.append(LineSegment(
                            x1=start[0], y1=start[1],
                            x2=end[0], y2=end[1],
                            confidence=avg_confidence,
                            velocity_x=avg_vel_x,
                            velocity_y=avg_vel_y,
                            finger_group=finger_group,
                            is_left_hand=is_left_hand
                        ))
        
        return segments
    
    def extract_segments(self, skeleton_data: SkeletonData, 
                        hand_landmarks: List[Tuple[float, float, float]],
                        is_left_hand: bool = True) -> List[LineSegment]:
        """기존 호환성을 위한 래퍼 - 개선된 버전 호출"""
        return self.extract_segments_advanced(skeleton_data, hand_landmarks, is_left_hand)
    
    def reset(self):
        """히스토리 초기화 (게임 리셋 시 호출)"""
        self.prev_left_landmarks = None
        self.prev_right_landmarks = None
        self.smoothed_left_landmarks = None
        self.smoothed_right_landmarks = None


# ==================== 스켈레톤 벽돌깨기 게임 클래스 ====================
class SkeletonBreakoutGame:
    """스켈레톤 기반 벽돌깨기 게임 메인 클래스"""
    
    def __init__(self, camera_id: int = 0, width: int = 800, height: int = 600):
        """
        Args:
            camera_id: 카메라 장치 ID
            width: 게임 화면 너비
            height: 게임 화면 높이
        """
        self.width = width
        self.height = height
        self.camera_id = camera_id
        
        # 게임 설정 (640x480 해상도에 맞춤)
        self.config = GameConfig(
            screen_width=width,
            screen_height=height,
            brick_rows=4,
            brick_cols=8,
            brick_width=70,    # 해상도에 맞게 조정
            brick_height=22,
            brick_padding=5,
            brick_offset_top=40,
            brick_offset_left=20,  # 중앙 정렬
            ball_radius=8,
            ball_speed=5.0,
            ball_color=(0, 255, 255),  # 노란색 (웹캠 배경에서 잘 보이도록)
            test_mode=False
        )
        
        # 컴포넌트들
        self.camera: Optional[CameraManager] = None
        self.tracker: Optional[SkeletonTracker] = None
        self.extractor = HandSkeletonExtractor()
        self.collision_handler = SkeletonCollisionHandler()
        
        # Pygame 관련
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        
        # 게임 오브젝트
        self.bricks: List[Brick] = []
        self.ball: Optional[Ball] = None
        
        # 현재 프레임의 손 선분들
        self.current_hand_segments: List[LineSegment] = []
        
        # 게임 상태
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_won = False
        self.running = False
        
        # 메인 메뉴 및 카운트다운 상태
        self.in_menu = True  # 메인 메뉴 상태
        self.countdown_active = False  # 카운트다운 진행 중
        self.countdown_value = 3  # 카운트다운 숫자
        self.countdown_start_time = 0  # 카운트다운 시작 시간
        self.game_started = False  # 실제 게임 시작 여부
        
        # 충돌 쿨다운 (같은 선분에 연속 충돌 방지)
        self.collision_cooldown = 0
        
        # FPS 표시 관련
        self.show_fps = False
        self.fps_history: List[float] = []
        self.last_frame_time = 0
        
        # 성능 최적화: 스켈레톤 추적 주기 (N프레임마다 추적)
        self.skeleton_update_interval = 2  # 2프레임마다 추적
        self.frame_count = 0
        self.last_skeleton_data: Optional[SkeletonData] = None
        
        # 개선된 충돌 처리 관련
        self.ball_prev_x: float = 0  # 이전 프레임 공 위치 (CCD용)
        self.ball_prev_y: float = 0
        self.substeps: int = 8  # 서브스텝 횟수 (충돌 정밀도 향상)
        self.show_debug_info: bool = False  # 디버그 정보 표시 (D키로 토글)
        
        # 하드모드 관련
        self.hard_mode: bool = False  # 하드모드 활성화 여부 (메인 메뉴에서 설정)
        self.default_group_health: int = 3  # 그룹 기본 체력
        # 그룹별 체력: {(is_left_hand, group_id): health}
        # 예: {(True, 0): 3, (True, 1): 3, ...} = 왼손 엄지 체력 3, 왼손 검지 체력 3, ...
        self.group_health: dict = {}
        # 비활성화된 그룹: {(is_left_hand, group_id): True}
        self.disabled_groups: dict = {}
        
    def initialize(self):
        """게임 초기화"""
        # Pygame 초기화
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("스켈레톤 벽돌깨기 - 손으로 공을 튕기세요!")
        self.clock = pygame.time.Clock()
        
        # 카메라 초기화
        self.camera = CameraManager(self.camera_id, self.width, self.height)
        if not self.camera.open():
            print("카메라를 열 수 없습니다!")
            return False
        
        # 스켈레톤 트래커 초기화
        # - enable_pose=False: 전신 포즈 추적 비활성화 (성능 향상)
        # - enable_hands=True: 손 추적만 활성화
        self.tracker = create_tracker(enable_pose=False, enable_hands=True)
        
        # 게임 오브젝트 생성
        self._create_bricks()
        self._create_ball()
        
        # 그룹 체력 초기화 (하드모드용)
        self._init_group_health()
        
        return True
    
    def _init_group_health(self):
        """그룹별 체력 초기화 (하드모드용)"""
        self.group_health = {}
        self.disabled_groups = {}
        
        # 왼손, 오른손 각각에 대해 모든 그룹 체력 초기화
        for is_left in [True, False]:
            for group_id in FingerGroup.ALL_GROUPS:
                key = (is_left, group_id)
                self.group_health[key] = self.default_group_health
                self.disabled_groups[key] = False
    
    def _damage_group(self, is_left_hand: bool, group_id: int):
        """
        그룹 체력 감소 (하드모드용)
        체력이 0이 되면 그룹 비활성화
        """
        key = (is_left_hand, group_id)
        
        # 이미 비활성화된 그룹은 무시
        if self.disabled_groups.get(key, False):
            return
        
        # 체력 감소
        if key in self.group_health:
            self.group_health[key] -= 1
            
            # 체력이 0 이하가 되면 그룹 비활성화
            if self.group_health[key] <= 0:
                self.disabled_groups[key] = True
                hand_name = "Left" if is_left_hand else "Right"
                group_name = FingerGroup.NAMES.get(group_id, "Unknown")
                print(f"[HARD MODE] {hand_name} {group_name} disabled!")
    
    def get_group_health(self, is_left_hand: bool, group_id: int) -> int:
        """그룹의 현재 체력 반환"""
        key = (is_left_hand, group_id)
        return self.group_health.get(key, 0)
    
    def is_group_disabled(self, is_left_hand: bool, group_id: int) -> bool:
        """그룹이 비활성화되었는지 확인"""
        key = (is_left_hand, group_id)
        return self.disabled_groups.get(key, False)
    
    def _create_bricks(self):
        """벽돌 생성"""
        self.bricks = []
        
        colors = [
            (255, 0, 0),      # 빨강
            (255, 127, 0),    # 주황
            (255, 255, 0),    # 노랑
            (0, 255, 0),      # 초록
        ]
        
        for row in range(self.config.brick_rows):
            color = colors[row % len(colors)]
            
            for col in range(self.config.brick_cols):
                x = (self.config.brick_offset_left + 
                     col * (self.config.brick_width + self.config.brick_padding))
                y = (self.config.brick_offset_top + 
                     row * (self.config.brick_height + self.config.brick_padding))
                
                brick = Brick(
                    x=x, y=y,
                    width=self.config.brick_width,
                    height=self.config.brick_height,
                    color=color,
                    shape=BrickShape.RECTANGLE
                )
                self.bricks.append(brick)
        
        if self.config.test_mode:
            for brick in self.bricks[1:]:
                brick.active = False
    
    def _create_ball(self):
        """공 생성"""
        x = self.width / 2
        y = self.height / 2  # 화면 중앙에서 시작
        self.ball = Ball(x, y, self.config)
        self.ball_prev_x = x
        self.ball_prev_y = y
    
    def process_skeleton(self, frame: np.ndarray) -> SkeletonData:
        """프레임에서 스켈레톤 데이터 추출"""
        return self.tracker.process_frame(frame)
    
    def extract_hand_segments(self, skeleton_data: SkeletonData) -> List[LineSegment]:
        """스켈레톤 데이터에서 손 선분 추출"""
        segments = []
        
        # 왼손 선분 추출
        if skeleton_data.left_hand_landmarks:
            left_segments = self.extractor.extract_segments(
                skeleton_data, skeleton_data.left_hand_landmarks, is_left_hand=True
            )
            segments.extend(left_segments)
        
        # 오른손 선분 추출
        if skeleton_data.right_hand_landmarks:
            right_segments = self.extractor.extract_segments(
                skeleton_data, skeleton_data.right_hand_landmarks, is_left_hand=False
            )
            segments.extend(right_segments)
        
        return segments
    
    def update(self):
        """게임 상태 업데이트 (서브스텝 시뮬레이션 적용)"""
        if self.game_over or self.game_won:
            return
        
        # 충돌 쿨다운 감소
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        
        # 이전 프레임 공 위치 저장 (CCD용)
        self.ball_prev_x = self.ball.x
        self.ball_prev_y = self.ball.y
        
        # 서브스텝 시뮬레이션
        original_vx = self.ball.vx
        original_vy = self.ball.vy
        
        for substep in range(self.substeps):
            # 공 위치 업데이트 (서브스텝 단위)
            substep_prev_x = self.ball.x
            substep_prev_y = self.ball.y
            
            # 속도를 서브스텝으로 나누어 적용
            self.ball.x += self.ball.vx / self.substeps
            self.ball.y += self.ball.vy / self.substeps
            
            # 벽 충돌 처리
            if self.ball.x - self.ball.radius <= 0 or self.ball.x + self.ball.radius >= self.ball.screen_width:
                self.ball.vx = -self.ball.vx
                self.ball.x = max(self.ball.radius, min(self.ball.screen_width - self.ball.radius, self.ball.x))
            
            if self.ball.y - self.ball.radius <= 0:
                self.ball.vy = -self.ball.vy
                self.ball.y = self.ball.radius
            
            # 바닥 체크
            if self.ball.y + self.ball.radius >= self.ball.screen_height:
                self.ball.active = False
                break
            
            # 손 선분과 충돌 체크 (개선된 버전)
            if self.collision_cooldown == 0:
                for seg in self.current_hand_segments:
                    # 하드모드: 비활성화된 그룹의 선분은 충돌 체크 건너뛰기
                    if self.hard_mode:
                        group_key = (seg.is_left_hand, seg.finger_group)
                        if self.disabled_groups.get(group_key, False):
                            continue
                    
                    if SkeletonCollisionHandler.check_ball_segment_collision_advanced(
                        self.ball, seg,
                        substep_prev_x, substep_prev_y
                    ):
                        self.collision_cooldown = 5
                        
                        # 하드모드: 충돌 시 그룹 체력 감소
                        if self.hard_mode:
                            self._damage_group(seg.is_left_hand, seg.finger_group)
                        
                        break
            
            # 벽돌 충돌 체크
            for brick in self.bricks:
                if brick.active and self.ball.check_brick_collision(brick):
                    self.score += 10
        
        # 공이 바닥에 닿았는지 체크
        if not self.ball.active:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self._create_ball()
                self.extractor.reset()  # 스무딩 히스토리 초기화
        
        # 승리 조건 체크
        active_bricks = sum(1 for b in self.bricks if b.active)
        if active_bricks == 0:
            self.game_won = True
    
    def opencv_frame_to_pygame(self, frame: np.ndarray) -> pygame.Surface:
        """OpenCV 프레임을 Pygame 서피스로 변환"""
        # BGR -> RGB 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 회전 및 뒤집기 (Pygame 좌표계에 맞추기)
        frame_rgb = np.rot90(frame_rgb)
        frame_rgb = np.flipud(frame_rgb)
        # Pygame 서피스 생성
        surface = pygame.surfarray.make_surface(frame_rgb)
        return surface
    
    def draw(self, frame: np.ndarray, skeleton_data: SkeletonData):
        """화면 그리기"""
        # 웹캠 프레임을 배경으로 사용
        bg_surface = self.opencv_frame_to_pygame(frame)
        self.screen.blit(bg_surface, (0, 0))
        
        # 스켈레톤 그리기 (손만 강조)
        self._draw_skeleton_overlay(skeleton_data)
        
        # 벽돌 그리기
        self._draw_bricks()
        
        # 공 그리기
        self._draw_ball()
        
        # 손 선분 그리기 (충돌 가능 영역 표시)
        self._draw_hand_segments()
        
        # UI 그리기
        self._draw_ui()
        
        pygame.display.flip()
    
    def _draw_bricks(self):
        """벽돌 그리기"""
        for brick in self.bricks:
            brick.draw(self.screen)
    
    def _draw_ball(self):
        """공 그리기"""
        if self.ball:
            self.ball.draw(self.screen)
    
    def _draw_skeleton_overlay(self, skeleton_data: SkeletonData):
        """스켈레톤을 Pygame 위에 그리기"""
        # 손 랜드마크 포인트만 그리기
        if skeleton_data.left_hand_landmarks:
            for idx in range(21):
                point = skeleton_data.get_pixel_coords(skeleton_data.left_hand_landmarks, idx)
                if point:
                    pygame.draw.circle(self.screen, (255, 128, 0), point, 4)
        
        if skeleton_data.right_hand_landmarks:
            for idx in range(21):
                point = skeleton_data.get_pixel_coords(skeleton_data.right_hand_landmarks, idx)
                if point:
                    pygame.draw.circle(self.screen, (0, 128, 255), point, 4)
    
    def _draw_hand_segments(self):
        """손 선분 그리기 (손가락별 색상, 하드모드 체력 표시)"""
        for seg in self.current_hand_segments:
            # 하드모드: 비활성화된 그룹 체크
            is_disabled = False
            group_health = self.default_group_health
            if self.hard_mode:
                is_disabled = self.is_group_disabled(seg.is_left_hand, seg.finger_group)
                group_health = self.get_group_health(seg.is_left_hand, seg.finger_group)
            
            # 손가락 그룹별 색상 사용
            if seg.finger_group >= 0 and seg.finger_group in FingerGroup.COLORS:
                base_color = FingerGroup.COLORS[seg.finger_group]
            else:
                base_color = (0, 255, 0)  # 기본 초록색
            
            # 하드모드: 비활성화된 그룹은 빨간 점선으로 표시
            if self.hard_mode and is_disabled:
                color = (100, 100, 100)  # 회색 (비활성화)
                thickness = 1
                # 점선 효과 (짧은 선분 여러 개)
                self._draw_dashed_line(
                    (int(seg.x1), int(seg.y1)),
                    (int(seg.x2), int(seg.y2)),
                    color, thickness, dash_length=5
                )
                continue
            
            # 하드모드: 체력에 따라 색상 밝기 조절
            if self.hard_mode:
                health_ratio = group_health / self.default_group_health
                # 체력이 낮을수록 어두워짐
                color = (
                    int(base_color[0] * (0.3 + 0.7 * health_ratio)),
                    int(base_color[1] * (0.3 + 0.7 * health_ratio)),
                    int(base_color[2] * (0.3 + 0.7 * health_ratio))
                )
                thickness = max(2, int(4 * health_ratio))
            else:
                # 노말모드: 신뢰도에 따라 색상 결정
                if seg.confidence < 0.3:
                    color = (base_color[0] // 3, base_color[1] // 3, base_color[2] // 3)
                    thickness = 2
                elif seg.confidence < 0.7:
                    color = (base_color[0] // 2, base_color[1] // 2, base_color[2] // 2)
                    thickness = 3
                else:
                    color = base_color
                    thickness = 4
            
            # 선분 그리기
            pygame.draw.line(
                self.screen, 
                color,
                (int(seg.x1), int(seg.y1)),
                (int(seg.x2), int(seg.y2)),
                thickness
            )
    
    def _draw_dashed_line(self, start: Tuple[int, int], end: Tuple[int, int], 
                          color: Tuple[int, int, int], thickness: int, dash_length: int = 10):
        """점선 그리기"""
        x1, y1 = start
        x2, y2 = end
        dx = x2 - x1
        dy = y2 - y1
        distance = math.sqrt(dx * dx + dy * dy)
        
        if distance == 0:
            return
        
        dashes = int(distance / dash_length)
        if dashes == 0:
            dashes = 1
        
        for i in range(0, dashes, 2):
            start_ratio = i / dashes
            end_ratio = min((i + 1) / dashes, 1.0)
            
            dash_start = (int(x1 + dx * start_ratio), int(y1 + dy * start_ratio))
            dash_end = (int(x1 + dx * end_ratio), int(y1 + dy * end_ratio))
            
            pygame.draw.line(self.screen, color, dash_start, dash_end, thickness)
    
    def _draw_ui(self):
        """UI 요소 그리기 - 통합 상단 바"""
        # ===== 상단 바 (하나의 배경) =====
        top_bar_height = 32
        top_bar = pygame.Surface((self.width, top_bar_height), pygame.SRCALPHA)
        top_bar.fill((0, 0, 0, 160))
        self.screen.blit(top_bar, (0, 0))
        
        # 폰트 설정
        main_font = pygame.font.Font(None, 28)
        small_font = pygame.font.Font(None, 22)
        
        # 왼쪽: 점수
        score_text = main_font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 6))
        
        # 오른쪽: 목숨 (하트 아이콘 스타일)
        lives_text = main_font.render(f"Lives: {self.lives}", True, (255, 100, 100))
        lives_rect = lives_text.get_rect(right=self.width - 10, top=6)
        self.screen.blit(lives_text, lives_rect)
        
        # 중앙: 모드 + 손 감지 상태
        # 모드 표시
        mode_text = "HARD" if self.hard_mode else "NORMAL"
        mode_color = (255, 80, 80) if self.hard_mode else (100, 255, 100)
        mode_surface = small_font.render(f"[{mode_text}]", True, mode_color)
        mode_rect = mode_surface.get_rect(centerx=self.width // 2 - 50, centery=top_bar_height // 2)
        self.screen.blit(mode_surface, mode_rect)
        
        # 구분선
        separator = small_font.render("|", True, (100, 100, 100))
        sep_rect = separator.get_rect(centerx=self.width // 2, centery=top_bar_height // 2)
        self.screen.blit(separator, sep_rect)
        
        # 손 감지 상태
        hand_status = "Hand OK" if self.current_hand_segments else "Show Hand"
        status_color = (100, 255, 100) if self.current_hand_segments else (255, 100, 100)
        status_text = small_font.render(hand_status, True, status_color)
        status_rect = status_text.get_rect(centerx=self.width // 2 + 50, centery=top_bar_height // 2)
        self.screen.blit(status_text, status_rect)
        
        # ===== FPS 표시 (F키로 토글) - 오른쪽 하단 =====
        if self.show_fps:
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            fps_color = (0, 255, 0) if avg_fps >= 30 else (255, 255, 0) if avg_fps >= 20 else (255, 0, 0)
            fps_text = small_font.render(f"FPS: {avg_fps:.1f}", True, fps_color)
            fps_rect = fps_text.get_rect(right=self.width - 10, bottom=self.height - 10)
            
            # 작은 배경
            fps_bg = pygame.Surface((fps_rect.width + 10, fps_rect.height + 6), pygame.SRCALPHA)
            fps_bg.fill((0, 0, 0, 150))
            self.screen.blit(fps_bg, (fps_rect.x - 5, fps_rect.y - 3))
            self.screen.blit(fps_text, fps_rect)
        
        # ===== 디버그 정보 (D키로 토글) - 왼쪽 하단 =====
        if self.show_debug_info:
            debug_font = pygame.font.Font(None, 20)
            debug_lines = []
            
            # 서브스텝, 선분 수
            debug_lines.append((f"Substeps: {self.substeps}", (180, 180, 180)))
            debug_lines.append((f"Segments: {len(self.current_hand_segments)}", (180, 180, 180)))
            
            # 배경 크기 계산
            line_height = 16
            bg_height = len(debug_lines) * line_height + 8
            debug_bg = pygame.Surface((100, bg_height), pygame.SRCALPHA)
            debug_bg.fill((0, 0, 0, 150))
            self.screen.blit(debug_bg, (5, self.height - bg_height - 5))
            
            # 텍스트 그리기
            for i, (text, color) in enumerate(debug_lines):
                text_surface = debug_font.render(text, True, color)
                self.screen.blit(text_surface, (10, self.height - bg_height + 4 + i * line_height))
        
        # 게임 오버/승리 메시지
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.width/2, self.height/2 - 30))
            self.screen.blit(game_over_text, text_rect)
            
            msg_font = pygame.font.Font(None, 28)
            restart_text = msg_font.render("Press R to Restart | M for Menu", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.width/2, self.height/2 + 30))
            self.screen.blit(restart_text, restart_rect)
        
        if self.game_won:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            win_font = pygame.font.Font(None, 72)
            win_text = win_font.render("YOU WIN!", True, (0, 255, 0))
            text_rect = win_text.get_rect(center=(self.width/2, self.height/2 - 30))
            self.screen.blit(win_text, text_rect)
            
            msg_font = pygame.font.Font(None, 28)
            restart_text = msg_font.render("Press R to Restart | M for Menu", True, (255, 255, 255))
            restart_rect = restart_text.get_rect(center=(self.width/2, self.height/2 + 30))
            self.screen.blit(restart_text, restart_rect)
    
    def reset(self):
        """게임 리셋"""
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_won = False
        self._create_bricks()
        self._create_ball()
        self.extractor.reset()  # 스무딩 히스토리 초기화
        self.collision_cooldown = 0
        self._init_group_health()  # 그룹 체력 초기화 (하드모드용)
    
    def run(self):
        """게임 메인 루프 실행"""
        if not self.initialize():
            print("초기화 실패!")
            return
        
        self.running = True
        self.in_menu = True
        
        print("=" * 50)
        print("스켈레톤 벽돌깨기 시작!")
        print("=" * 50)
        print("메인 메뉴에서 START를 클릭하세요!")
        print("=" * 50)
        
        while self.running:
            # 카메라 프레임 읽기 (모든 상태에서 공통)
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # 좌우 반전 (미러 모드)
            frame = cv2.flip(frame, 1)
            
            # 프레임 크기 조정 (필요시)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # 스켈레톤 처리 (모든 상태에서 처리하여 손이 보이도록)
            self.frame_count += 1
            if self.frame_count >= self.skeleton_update_interval or self.last_skeleton_data is None:
                self.frame_count = 0
                skeleton_data = self.process_skeleton(frame)
                self.last_skeleton_data = skeleton_data
                self.current_hand_segments = self.extract_hand_segments(skeleton_data)
            else:
                skeleton_data = self.last_skeleton_data
            
            # 상태별 처리
            if self.in_menu:
                self._run_menu(frame, skeleton_data)
            elif self.countdown_active:
                self._run_countdown(frame, skeleton_data)
            else:
                self._run_game(frame, skeleton_data)
            
            # FPS 제한 및 계산
            self.clock.tick(self.config.fps)
            
            # FPS 계산 (이동 평균)
            current_fps = self.clock.get_fps()
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
        
        self.cleanup()
    
    def _run_menu(self, frame, skeleton_data):
        """메인 메뉴 처리"""
        mouse_pos = pygame.mouse.get_pos()
        
        # 버튼 영역 정의
        center_x = self.width // 2
        start_btn = pygame.Rect(center_x - 100, 200, 200, 60)
        hard_btn = pygame.Rect(center_x - 100, 290, 200, 50)
        
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_RETURN or event.key == pygame.K_SPACE:
                    # Enter 또는 Space로 게임 시작
                    self._start_countdown()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # 좌클릭
                    if start_btn.collidepoint(mouse_pos):
                        self._start_countdown()
                    elif hard_btn.collidepoint(mouse_pos):
                        self.hard_mode = not self.hard_mode
                        print(f"Hard Mode: {'ON' if self.hard_mode else 'OFF'}")
        
        # 그리기
        self._draw_menu(frame, skeleton_data, start_btn, hard_btn, mouse_pos)
    
    def _draw_menu(self, frame, skeleton_data, start_btn, hard_btn, mouse_pos):
        """메인 메뉴 그리기"""
        # 웹캠 프레임을 배경으로
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        self.screen.blit(frame_surface, (0, 0))
        
        # 반투명 오버레이
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 150))
        self.screen.blit(overlay, (0, 0))
        
        # 손 선분 그리기 (메뉴에서도 손이 보이도록)
        self._draw_hand_segments()
        
        # 타이틀
        title_font = pygame.font.Font(None, 64)
        title_text = title_font.render("SKELETON BREAKOUT", True, (255, 255, 255))
        title_rect = title_text.get_rect(center=(self.width // 2, 100))
        self.screen.blit(title_text, title_rect)
        
        # 부제목
        subtitle_font = pygame.font.Font(None, 28)
        subtitle_text = subtitle_font.render("Use your hands to bounce the ball!", True, (200, 200, 200))
        subtitle_rect = subtitle_text.get_rect(center=(self.width // 2, 145))
        self.screen.blit(subtitle_text, subtitle_rect)
        
        # START 버튼
        start_color = (0, 200, 0) if start_btn.collidepoint(mouse_pos) else (0, 150, 0)
        pygame.draw.rect(self.screen, start_color, start_btn, border_radius=10)
        pygame.draw.rect(self.screen, (255, 255, 255), start_btn, 3, border_radius=10)
        
        start_font = pygame.font.Font(None, 48)
        start_text = start_font.render("START", True, (255, 255, 255))
        start_text_rect = start_text.get_rect(center=start_btn.center)
        self.screen.blit(start_text, start_text_rect)
        
        # HARD MODE 토글 버튼
        if self.hard_mode:
            hard_color = (200, 0, 0) if hard_btn.collidepoint(mouse_pos) else (150, 0, 0)
        else:
            hard_color = (80, 80, 80) if hard_btn.collidepoint(mouse_pos) else (60, 60, 60)
        
        pygame.draw.rect(self.screen, hard_color, hard_btn, border_radius=8)
        pygame.draw.rect(self.screen, (255, 255, 255), hard_btn, 2, border_radius=8)
        
        hard_font = pygame.font.Font(None, 32)
        hard_status = "HARD MODE: ON" if self.hard_mode else "HARD MODE: OFF"
        hard_text = hard_font.render(hard_status, True, (255, 255, 255))
        hard_text_rect = hard_text.get_rect(center=hard_btn.center)
        self.screen.blit(hard_text, hard_text_rect)
        
        # 하드모드 설명
        if self.hard_mode:
            desc_font = pygame.font.Font(None, 22)
            desc_text = desc_font.render("Hand segments have HP. They break after 3 hits!", True, (255, 150, 150))
            desc_rect = desc_text.get_rect(center=(self.width // 2, 355))
            self.screen.blit(desc_text, desc_rect)
        
        # 조작법 안내
        help_font = pygame.font.Font(None, 24)
        help_lines = [
            "Controls:",
            "R - Restart | F - FPS | D - Debug | ESC - Quit"
        ]
        for i, line in enumerate(help_lines):
            help_text = help_font.render(line, True, (180, 180, 180))
            help_rect = help_text.get_rect(center=(self.width // 2, 410 + i * 25))
            self.screen.blit(help_text, help_rect)
        
        pygame.display.flip()
    
    def _start_countdown(self):
        """카운트다운 시작"""
        self.in_menu = False
        self.countdown_active = True
        self.countdown_value = 3
        self.countdown_start_time = pygame.time.get_ticks()
        self.reset()  # 게임 상태 초기화
        print("Game starting in 3...")
    
    def _run_countdown(self, frame, skeleton_data):
        """카운트다운 처리"""
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.in_menu = True
                    self.countdown_active = False
        
        # 카운트다운 업데이트 (1초마다)
        elapsed = pygame.time.get_ticks() - self.countdown_start_time
        new_countdown = 3 - (elapsed // 1000)
        
        if new_countdown != self.countdown_value and new_countdown > 0:
            self.countdown_value = new_countdown
            print(f"{self.countdown_value}...")
        
        if elapsed >= 3000:
            # 카운트다운 완료
            self.countdown_active = False
            self.game_started = True
            print("GO!")
        
        # 그리기
        self._draw_countdown(frame, skeleton_data)
    
    def _draw_countdown(self, frame, skeleton_data):
        """카운트다운 화면 그리기"""
        # 웹캠 프레임을 배경으로
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        self.screen.blit(frame_surface, (0, 0))
        
        # 벽돌 그리기
        self._draw_bricks()
        
        # 공 그리기 (정지 상태)
        self._draw_ball()
        
        # 손 선분 그리기
        self._draw_hand_segments()
        
        # 카운트다운 숫자 (큰 글씨)
        elapsed = pygame.time.get_ticks() - self.countdown_start_time
        
        # 애니메이션 효과: 숫자가 커졌다가 작아지기
        phase = (elapsed % 1000) / 1000.0
        if phase < 0.3:
            scale = 1.0 + phase * 0.5  # 커지기
        else:
            scale = 1.15 - (phase - 0.3) * 0.2  # 작아지기
        
        base_size = 150
        font_size = int(base_size * scale)
        
        countdown_font = pygame.font.Font(None, font_size)
        countdown_text = countdown_font.render(str(self.countdown_value), True, (255, 255, 0))
        countdown_rect = countdown_text.get_rect(center=(self.width // 2, self.height // 2))
        
        # 그림자 효과
        shadow_text = countdown_font.render(str(self.countdown_value), True, (0, 0, 0))
        shadow_rect = shadow_text.get_rect(center=(self.width // 2 + 4, self.height // 2 + 4))
        self.screen.blit(shadow_text, shadow_rect)
        self.screen.blit(countdown_text, countdown_rect)
        
        # "GET READY" 텍스트
        ready_font = pygame.font.Font(None, 36)
        ready_text = ready_font.render("GET READY!", True, (255, 255, 255))
        ready_rect = ready_text.get_rect(center=(self.width // 2, self.height // 2 - 100))
        self.screen.blit(ready_text, ready_rect)
        
        # UI 표시
        self._draw_ui()
        
        pygame.display.flip()
    
    def _run_game(self, frame, skeleton_data):
        """게임 플레이 처리"""
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_r:
                    self._start_countdown()  # 카운트다운과 함께 재시작
                elif event.key == pygame.K_m:
                    # M키로 메뉴로 돌아가기
                    self.in_menu = True
                    self.game_started = False
                elif event.key == pygame.K_f:
                    self.show_fps = not self.show_fps
                    print(f"FPS 표시: {'ON' if self.show_fps else 'OFF'}")
                elif event.key == pygame.K_d:
                    self.show_debug_info = not self.show_debug_info
                    print(f"디버그 정보: {'ON' if self.show_debug_info else 'OFF'}")
        
        # 게임 업데이트
        self.update()
        
        # 그리기
        self.draw(frame, skeleton_data)
        
        # 게임 오버 또는 승리 시 메뉴로 돌아갈 수 있음
        if self.game_over or self.game_won:
            # 클릭하면 메뉴로
            for event in pygame.event.get():
                if event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.KEYDOWN:
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        self._start_countdown()
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_m:
                        self.in_menu = True
                        self.game_started = False
    
    def cleanup(self):
        """리소스 정리"""
        if self.camera:
            self.camera.release()
        if self.tracker:
            self.tracker.release()
        pygame.quit()


# ==================== 메인 실행 ====================
if __name__ == "__main__":
    game = SkeletonBreakoutGame(
        camera_id=0,
        width=640,   # 성능 최적화: 해상도 낮춤
        height=480
    )
    game.run()
