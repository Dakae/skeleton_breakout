# 스켈레톤 기반 벽돌깨기 게임
# skeleton.py와 breakOut.py의 기능을 활용하여 손 스켈레톤으로 공을 튕기는 게임
# 배경은 웹캠 화면이며, 손의 선분이 막대 역할을 합니다.

import cv2
import numpy as np
import pygame
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

# 기존 모듈에서 필요한 클래스 임포트
from skeleton import (
    SkeletonTracker, 
    SkeletonData, 
    HandLandmarkGroups,
    Colors,
    CameraManager,
    create_tracker
)

from breakOut import (
    Ball,
    Brick,
    GameConfig,
    BrickShape
)


# ==================== 선분 충돌 처리 클래스 ====================
@dataclass
class LineSegment:
    """선분을 나타내는 데이터 클래스"""
    x1: float
    y1: float
    x2: float
    y2: float
    
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


class SkeletonCollisionHandler:
    """스켈레톤 선분과 공의 충돌을 처리하는 클래스"""
    
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
    def check_ball_segment_collision(ball: Ball, seg: LineSegment, min_segment_length: float = 10.0) -> bool:
        """
        공과 선분의 충돌 체크 및 반사 처리
        
        Args:
            ball: 공 객체
            seg: 선분 객체
            min_segment_length: 충돌 처리할 최소 선분 길이
            
        Returns:
            충돌 여부
        """
        if not ball.active:
            return False
        
        # 너무 짧은 선분은 무시
        if seg.get_length() < min_segment_length:
            return False
        
        # 점과 선분 사이의 거리 계산
        dist, closest_x, closest_y = SkeletonCollisionHandler.point_to_segment_distance(
            ball.x, ball.y, seg
        )
        
        # 충돌 체크
        if dist < ball.radius + 3:  # 약간의 여유 추가
            # 공이 선분을 향해 이동 중인지 확인 (이미 튕긴 후 재충돌 방지)
            # 공의 속도 방향과 선분까지의 방향 비교
            to_line_x = closest_x - ball.x
            to_line_y = closest_y - ball.y
            
            # 내적으로 같은 방향인지 확인
            dot = ball.vx * to_line_x + ball.vy * to_line_y
            if dot <= 0:
                return False  # 공이 선분에서 멀어지는 중
            
            # 선분의 법선 벡터 계산
            nx, ny = seg.get_normal()
            
            # 공의 중심이 선분의 어느 쪽에 있는지 판단
            # 선분 시작점에서 공까지의 벡터
            to_ball_x = ball.x - seg.x1
            to_ball_y = ball.y - seg.y1
            
            # 공이 법선 반대쪽에 있으면 법선 방향 뒤집기
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


# ==================== 손 스켈레톤 선분 추출 ====================
class HandSkeletonExtractor:
    """손 랜드마크에서 충돌 가능한 선분들을 추출하는 클래스"""
    
    def __init__(self):
        # 충돌에 사용할 손 연결 (모든 연결 사용)
        self.connections = HandLandmarkGroups.ALL_CONNECTIONS
    
    def extract_segments(self, skeleton_data: SkeletonData, 
                        hand_landmarks: List[Tuple[float, float, float]],
                        is_left_hand: bool = True) -> List[LineSegment]:
        """
        손 랜드마크에서 선분 목록 추출
        
        Args:
            skeleton_data: 스켈레톤 데이터 (좌표 변환용)
            hand_landmarks: 손 랜드마크 리스트
            is_left_hand: 왼손 여부
            
        Returns:
            선분 목록
        """
        segments = []
        
        if hand_landmarks is None:
            return segments
        
        for start_idx, end_idx in self.connections:
            start = skeleton_data.get_pixel_coords(hand_landmarks, start_idx)
            end = skeleton_data.get_pixel_coords(hand_landmarks, end_idx)
            
            if start and end:
                segments.append(LineSegment(
                    x1=start[0], y1=start[1],
                    x2=end[0], y2=end[1]
                ))
        
        return segments


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
        
        # 스켈레톤 트래커 초기화 (손만 사용, 포즈는 비활성화하여 성능 향상)
        self.tracker = create_tracker(enable_pose=False, enable_hands=True)
        
        # 게임 오브젝트 생성
        self._create_bricks()
        self._create_ball()
        
        return True
    
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
        """게임 상태 업데이트"""
        if self.game_over or self.game_won:
            return
        
        # 충돌 쿨다운 감소
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        
        # 공 업데이트
        self.ball.update()
        
        # 손 선분과 충돌 체크
        if self.collision_cooldown == 0:
            for seg in self.current_hand_segments:
                if self.collision_handler.check_ball_segment_collision(self.ball, seg):
                    self.collision_cooldown = 5  # 5프레임 동안 추가 충돌 방지
                    break
        
        # 벽돌 충돌 체크
        for brick in self.bricks:
            if self.ball.check_brick_collision(brick):
                self.score += 10
        
        # 공이 바닥에 닿았는지 체크
        if not self.ball.active:
            self.lives -= 1
            if self.lives <= 0:
                self.game_over = True
            else:
                self._create_ball()
        
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
        for brick in self.bricks:
            brick.draw(self.screen)
        
        # 공 그리기
        self.ball.draw(self.screen)
        
        # 손 선분 그리기 (충돌 가능 영역 표시)
        self._draw_hand_segments()
        
        # UI 그리기
        self._draw_ui()
        
        pygame.display.flip()
    
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
        """손 선분 그리기 (충돌 가능 영역)"""
        for seg in self.current_hand_segments:
            # 선분 그리기 (두껍게)
            pygame.draw.line(
                self.screen, 
                (0, 255, 0),  # 초록색
                (int(seg.x1), int(seg.y1)),
                (int(seg.x2), int(seg.y2)),
                4
            )
    
    def _draw_ui(self):
        """UI 요소 그리기"""
        font = pygame.font.Font(None, 36)
        
        # 반투명 배경 박스
        ui_bg = pygame.Surface((200, 40), pygame.SRCALPHA)
        ui_bg.fill((0, 0, 0, 128))
        self.screen.blit(ui_bg, (5, 5))
        
        # 점수
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # FPS 표시 (F키로 토글)
        if self.show_fps:
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            fps_color = (0, 255, 0) if avg_fps >= 30 else (255, 255, 0) if avg_fps >= 20 else (255, 0, 0)
            fps_font = pygame.font.Font(None, 28)
            fps_text = fps_font.render(f"FPS: {avg_fps:.1f}", True, fps_color)
            
            fps_bg = pygame.Surface((90, 25), pygame.SRCALPHA)
            fps_bg.fill((0, 0, 0, 180))
            self.screen.blit(fps_bg, (5, 50))
            self.screen.blit(fps_text, (10, 52))
        
        # 남은 생명
        lives_text = font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        lives_bg = pygame.Surface((100, 40), pygame.SRCALPHA)
        lives_bg.fill((0, 0, 0, 128))
        self.screen.blit(lives_bg, (self.width - 105, 5))
        self.screen.blit(lives_text, (self.width - 100, 10))
        
        # 손 감지 상태 표시
        hand_status = "손 감지됨" if self.current_hand_segments else "손을 보여주세요"
        status_color = (0, 255, 0) if self.current_hand_segments else (255, 0, 0)
        status_font = pygame.font.Font(None, 28)
        status_text = status_font.render(hand_status, True, status_color)
        
        status_bg = pygame.Surface((150, 30), pygame.SRCALPHA)
        status_bg.fill((0, 0, 0, 128))
        self.screen.blit(status_bg, (self.width // 2 - 75, 5))
        self.screen.blit(status_text, (self.width // 2 - 65, 10))
        
        # 게임 오버/승리 메시지
        if self.game_over:
            overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (0, 0))
            
            game_over_font = pygame.font.Font(None, 72)
            game_over_text = game_over_font.render("GAME OVER", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.width/2, self.height/2 - 30))
            self.screen.blit(game_over_text, text_rect)
            
            restart_text = font.render("Press R to Restart", True, (255, 255, 255))
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
            
            restart_text = font.render("Press R to Restart", True, (255, 255, 255))
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
    
    def run(self):
        """게임 메인 루프 실행"""
        if not self.initialize():
            print("초기화 실패!")
            return
        
        self.running = True
        print("스켈레톤 벽돌깨기 시작!")
        print("손을 카메라에 보여주면 손의 선이 공을 튕기는 막대가 됩니다.")
        print("조작: R - 재시작, F - FPS 표시 토글, ESC - 종료")
        
        while self.running:
            # 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_f:
                        self.show_fps = not self.show_fps
                        print(f"FPS 표시: {'ON' if self.show_fps else 'OFF'}")
            
            # 카메라 프레임 읽기
            ret, frame = self.camera.read()
            if not ret:
                continue
            
            # 좌우 반전 (미러 모드)
            frame = cv2.flip(frame, 1)
            
            # 프레임 크기 조정 (필요시)
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            
            # 스켈레톤 처리 (N프레임마다 추적하여 성능 최적화)
            self.frame_count += 1
            if self.frame_count >= self.skeleton_update_interval or self.last_skeleton_data is None:
                self.frame_count = 0
                skeleton_data = self.process_skeleton(frame)
                self.last_skeleton_data = skeleton_data
                # 손 선분 추출
                self.current_hand_segments = self.extract_hand_segments(skeleton_data)
            else:
                skeleton_data = self.last_skeleton_data
            
            # 게임 업데이트
            self.update()
            
            # 그리기
            self.draw(frame, skeleton_data)
            
            # FPS 제한 및 계산
            self.clock.tick(self.config.fps)
            
            # FPS 계산 (이동 평균)
            current_fps = self.clock.get_fps()
            self.fps_history.append(current_fps)
            if len(self.fps_history) > 30:  # 최근 30프레임 평균
                self.fps_history.pop(0)
        
        self.cleanup()
    
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
