# 여기에 벽돌깨기 코드가 작성이 됩니다.
# 다른 UI에서 가져다 쓸 수 있도록 모듈화 하여 코드를 작성합니다.
# 가로 세로 크기, 벽돌 개수, 벽돌 색상, 벽돌 위치, 벽돌 크기, 벽돌 모양 같은것들을 설정 가능합니다.
# 이 코드는 파이썬 코드로 작성이 되며 이 코드 자체로도 벽돌깨기 게임을 실행할 수 있도록 합니다.
# 테스트용 코드로 블럭 하나만 남기고 제거 가능한 코드 추가
# 나중에 스켈레톤 포즈 데이터가 벽돌깨기 게임의 스틱을 대신 할 예정이기 때문에 스틱이 기울기가 가능하도록 코드 추가,스틱의 기울기에 따라 공이 튕기는 각도가 달라지도록 코드 추가
# 우선 스틱의 움직임은 좌우 방향키, 기울기를 위한 회전은 상 하 방향키로 한다.

import pygame
import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Callable
from enum import Enum


class BrickShape(Enum):
    """벽돌 모양 열거형"""
    RECTANGLE = "rectangle"
    ROUNDED = "rounded"
    DIAMOND = "diamond"


@dataclass
class GameConfig:
    """게임 설정을 담는 데이터 클래스"""
    # 화면 설정
    screen_width: int = 800
    screen_height: int = 600
    
    # 벽돌 설정
    brick_rows: int = 5
    brick_cols: int = 10
    brick_width: int = 70
    brick_height: int = 25
    brick_padding: int = 5
    brick_offset_top: int = 50
    brick_offset_left: int = 35
    brick_shape: BrickShape = BrickShape.RECTANGLE
    brick_colors: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (255, 0, 0),      # 빨강
        (255, 127, 0),    # 주황
        (255, 255, 0),    # 노랑
        (0, 255, 0),      # 초록
        (0, 0, 255),      # 파랑
    ])
    
    # 패들(스틱) 설정
    paddle_width: int = 100
    paddle_height: int = 15
    paddle_speed: int = 8
    paddle_color: Tuple[int, int, int] = (255, 255, 255)
    paddle_max_angle: float = 30.0  # 최대 기울기 각도 (도)
    paddle_rotation_speed: float = 2.0  # 기울기 변화 속도
    
    # 공 설정
    ball_radius: int = 8
    ball_speed: float = 5.0
    ball_color: Tuple[int, int, int] = (255, 255, 255)
    
    # 게임 설정
    background_color: Tuple[int, int, int] = (30, 30, 30)
    fps: int = 60
    
    # 테스트 모드 설정
    test_mode: bool =  False # True면 벽돌 하나만 남김


@dataclass
class Brick:
    """벽돌 클래스"""
    x: float
    y: float
    width: int
    height: int
    color: Tuple[int, int, int]
    shape: BrickShape
    active: bool = True
    
    def get_rect(self) -> pygame.Rect:
        """충돌 감지용 사각형 반환"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def draw(self, surface: pygame.Surface):
        """벽돌 그리기"""
        if not self.active:
            return
            
        if self.shape == BrickShape.RECTANGLE:
            pygame.draw.rect(surface, self.color, 
                           (self.x, self.y, self.width, self.height))
            pygame.draw.rect(surface, (255, 255, 255), 
                           (self.x, self.y, self.width, self.height), 1)
        
        elif self.shape == BrickShape.ROUNDED:
            pygame.draw.rect(surface, self.color, 
                           (self.x, self.y, self.width, self.height), 
                           border_radius=8)
            pygame.draw.rect(surface, (255, 255, 255), 
                           (self.x, self.y, self.width, self.height), 1, 
                           border_radius=8)
        
        elif self.shape == BrickShape.DIAMOND:
            cx = self.x + self.width // 2
            cy = self.y + self.height // 2
            points = [
                (cx, self.y),
                (self.x + self.width, cy),
                (cx, self.y + self.height),
                (self.x, cy)
            ]
            pygame.draw.polygon(surface, self.color, points)
            pygame.draw.polygon(surface, (255, 255, 255), points, 1)


class Paddle:
    """패들(스틱) 클래스 - 기울기 기능 포함"""
    
    def __init__(self, x: float, y: float, config: GameConfig):
        self.x = x
        self.y = y
        self.width = config.paddle_width
        self.height = config.paddle_height
        self.speed = config.paddle_speed
        self.color = config.paddle_color
        self.angle = 0.0  # 현재 기울기 각도 (도)
        self.max_angle = config.paddle_max_angle
        self.rotation_speed = config.paddle_rotation_speed
        self.screen_width = config.screen_width
    
    def move_left(self):
        """왼쪽으로 이동"""
        self.x = max(0, self.x - self.speed)
    
    def move_right(self):
        """오른쪽으로 이동"""
        self.x = min(self.screen_width - self.width, self.x + self.speed)
    
    def rotate_up(self):
        """위쪽 방향키 - 시계 반대방향 회전 (왼쪽이 올라감)"""
        self.angle = max(-self.max_angle, self.angle - self.rotation_speed)
    
    def rotate_down(self):
        """아래쪽 방향키 - 시계 방향 회전 (오른쪽이 올라감)"""
        self.angle = min(self.max_angle, self.angle + self.rotation_speed)
    
    def set_position(self, x: float):
        """위치 직접 설정 (스켈레톤 포즈 연동용)"""
        self.x = max(0, min(self.screen_width - self.width, x))
    
    def set_angle(self, angle: float):
        """기울기 직접 설정 (스켈레톤 포즈 연동용)"""
        self.angle = max(-self.max_angle, min(self.max_angle, angle))
    
    def get_center(self) -> Tuple[float, float]:
        """패들 중심점 반환"""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    def get_collision_rect(self) -> pygame.Rect:
        """충돌 감지용 사각형 (기울기 무시한 기본 사각형)"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def point_to_local(self, px: float, py: float) -> Tuple[float, float]:
        """
        월드 좌표를 패들의 로컬 좌표계로 변환 (역회전)
        로컬 좌표계에서 패들은 원점에 수평으로 놓여있음
        """
        cx, cy = self.get_center()
        angle_rad = math.radians(-self.angle)  # 역회전
        
        # 중심 기준으로 이동
        dx = px - cx
        dy = py - cy
        
        # 역회전 적용
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        return (local_x, local_y)
    
    def get_surface_normal_at(self, px: float, py: float) -> Tuple[float, float]:
        """
        패들 표면의 법선 벡터 반환 (기울기 적용)
        """
        angle_rad = math.radians(self.angle)
        # 기본 법선은 위쪽 (0, -1), 회전 적용
        nx = math.sin(angle_rad)
        ny = -math.cos(angle_rad)
        return (nx, ny)
    
    def get_top_line(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """회전된 패들의 윗면 선분 반환 (왼쪽점, 오른쪽점)"""
        corners = self.get_rotated_corners()
        # corners: 좌상, 우상, 우하, 좌하
        return (corners[0], corners[1])
    
    def get_rotated_corners(self) -> List[Tuple[float, float]]:
        """회전된 패들의 네 꼭지점 좌표 반환"""
        cx, cy = self.get_center()
        angle_rad = math.radians(self.angle)
        
        # 패들의 네 꼭지점 (중심 기준)
        half_w = self.width / 2
        half_h = self.height / 2
        corners = [
            (-half_w, -half_h),
            (half_w, -half_h),
            (half_w, half_h),
            (-half_w, half_h)
        ]
        
        # 회전 적용
        rotated = []
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        for px, py in corners:
            rx = px * cos_a - py * sin_a + cx
            ry = px * sin_a + py * cos_a + cy
            rotated.append((rx, ry))
        
        return rotated
    
    def draw(self, surface: pygame.Surface):
        """패들 그리기 (회전 적용)"""
        corners = self.get_rotated_corners()
        pygame.draw.polygon(surface, self.color, corners)
        pygame.draw.polygon(surface, (200, 200, 200), corners, 2)
        
        # 기울기 표시 (중앙에 작은 선)
        cx, cy = self.get_center()
        angle_rad = math.radians(self.angle)
        line_len = 20
        end_x = cx + line_len * math.cos(angle_rad - math.pi/2)
        end_y = cy + line_len * math.sin(angle_rad - math.pi/2)
        pygame.draw.line(surface, (255, 0, 0), (cx, cy), (end_x, end_y), 2)


class Ball:
    """공 클래스"""
    
    def __init__(self, x: float, y: float, config: GameConfig):
        self.x = x
        self.y = y
        self.radius = config.ball_radius
        self.speed = config.ball_speed
        self.color = config.ball_color
        self.screen_width = config.screen_width
        self.screen_height = config.screen_height
        
        # 초기 방향 (위쪽으로 약간 비스듬히)
        angle = random.uniform(-math.pi/4, math.pi/4) - math.pi/2
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)
        
        self.active = True
    
    def update(self):
        """공 위치 업데이트"""
        if not self.active:
            return
            
        self.x += self.vx
        self.y += self.vy
        
        # 벽 충돌 처리
        if self.x - self.radius <= 0 or self.x + self.radius >= self.screen_width:
            self.vx = -self.vx
            self.x = max(self.radius, min(self.screen_width - self.radius, self.x))
        
        if self.y - self.radius <= 0:
            self.vy = -self.vy
            self.y = self.radius
        
        # 바닥에 닿으면 비활성화
        if self.y + self.radius >= self.screen_height:
            self.active = False
    
    def check_paddle_collision(self, paddle: Paddle) -> bool:
        """
        패들과 충돌 체크 및 반사 처리
        회전된 패들과의 정확한 충돌 판정 + 물리 기반 반사 (입사각 = 반사각)
        """
        if not self.active:
            return False
        
        # 공이 아래로 이동 중일 때만 체크 (위로 튕긴 직후 재충돌 방지)
        if self.vy <= 0:
            return False
        
        # 방법: 공의 좌표를 패들의 로컬 좌표계로 변환하여 충돌 체크
        # 로컬 좌표계에서 패들은 수평 상태
        local_x, local_y = paddle.point_to_local(self.x, self.y)
        
        # 로컬 좌표계에서의 패들 반너비/반높이
        half_w = paddle.width / 2
        half_h = paddle.height / 2
        
        # 로컬 좌표계에서 사각형-원 충돌 체크
        # 사각형의 가장 가까운 점 찾기
        closest_x = max(-half_w, min(local_x, half_w))
        closest_y = max(-half_h, min(local_y, half_h))
        
        # 거리 계산
        dist_x = local_x - closest_x
        dist_y = local_y - closest_y
        distance = math.sqrt(dist_x * dist_x + dist_y * dist_y)
        
        if distance < self.radius:
            # 충돌 발생!
            
            # === 물리 기반 반사 (입사각 = 반사각) ===
            # 패들 표면의 법선 벡터 (기울기 적용)
            # 기본 법선은 (0, -1) 위쪽, 패들 기울기만큼 회전
            paddle_angle_rad = math.radians(paddle.angle)
            
            # 충돌 면 판단 (윗면, 옆면 등)
            # local_y가 -half_h에 가까우면 윗면 충돌
            if closest_y <= -half_h + 2:  # 윗면 충돌
                # 윗면 법선 (회전 적용)
                nx = math.sin(paddle_angle_rad)
                ny = -math.cos(paddle_angle_rad)
            elif closest_x <= -half_w + 1:  # 왼쪽 면
                nx = -math.cos(paddle_angle_rad)
                ny = -math.sin(paddle_angle_rad)
            elif closest_x >= half_w - 1:  # 오른쪽 면
                nx = math.cos(paddle_angle_rad)
                ny = math.sin(paddle_angle_rad)
            else:  # 기본값: 윗면으로 처리
                nx = math.sin(paddle_angle_rad)
                ny = -math.cos(paddle_angle_rad)
            
            # 법선 벡터 정규화 (이미 단위 벡터이지만 확실히)
            n_len = math.sqrt(nx * nx + ny * ny)
            if n_len > 0:
                nx /= n_len
                ny /= n_len
            
            # 입사 속도 벡터
            vx, vy = self.vx, self.vy
            
            # 반사 공식: v' = v - 2(v·n)n
            # v·n (내적)
            dot = vx * nx + vy * ny
            
            # 반사 벡터 계산
            reflect_vx = vx - 2 * dot * nx
            reflect_vy = vy - 2 * dot * ny
            
            # 속도 크기 유지 (에너지 보존)
            reflect_speed = math.sqrt(reflect_vx * reflect_vx + reflect_vy * reflect_vy)
            if reflect_speed > 0:
                reflect_vx = reflect_vx / reflect_speed * self.speed
                reflect_vy = reflect_vy / reflect_speed * self.speed
            
            # 반사 후 공이 아래로 가는 것 방지 (최소한 위쪽 성분 보장)
            if reflect_vy > -0.5:
                reflect_vy = -abs(reflect_vy) - 0.5
                # 속도 크기 재조정
                reflect_speed = math.sqrt(reflect_vx * reflect_vx + reflect_vy * reflect_vy)
                if reflect_speed > 0:
                    reflect_vx = reflect_vx / reflect_speed * self.speed
                    reflect_vy = reflect_vy / reflect_speed * self.speed
            
            self.vx = reflect_vx
            self.vy = reflect_vy
            
            # 충돌 후 공을 패들 밖으로 밀어내기
            cx, cy = paddle.get_center()
            
            # local_x 위치를 월드 좌표로 변환하여 충돌 지점 계산
            cos_a = math.cos(paddle_angle_rad)
            sin_a = math.sin(paddle_angle_rad)
            world_hit_x = closest_x * cos_a - (-half_h) * sin_a + cx
            world_hit_y = closest_x * sin_a + (-half_h) * cos_a + cy
            
            # 공을 패들 표면 위로 이동 (법선 방향으로)
            self.x = world_hit_x + nx * (self.radius + 2)
            self.y = world_hit_y + ny * (self.radius + 2)
            
            return True
        
        return False
    
    def check_brick_collision(self, brick: Brick) -> bool:
        """벽돌과 충돌 체크"""
        if not self.active or not brick.active:
            return False
        
        brick_rect = brick.get_rect()
        
        # 공의 중심에서 벽돌까지의 가장 가까운 점 찾기
        closest_x = max(brick_rect.left, min(self.x, brick_rect.right))
        closest_y = max(brick_rect.top, min(self.y, brick_rect.bottom))
        
        # 거리 계산
        distance = math.sqrt((self.x - closest_x)**2 + (self.y - closest_y)**2)
        
        if distance < self.radius:
            # 충돌 발생 - 어느 면에서 충돌했는지 판단
            if closest_x == brick_rect.left or closest_x == brick_rect.right:
                self.vx = -self.vx
            if closest_y == brick_rect.top or closest_y == brick_rect.bottom:
                self.vy = -self.vy
            
            brick.active = False
            return True
        
        return False
    
    def draw(self, surface: pygame.Surface):
        """공 그리기"""
        if self.active:
            pygame.draw.circle(surface, self.color, (int(self.x), int(self.y)), self.radius)
            # 하이라이트 효과
            highlight_pos = (int(self.x - self.radius/3), int(self.y - self.radius/3))
            pygame.draw.circle(surface, (255, 255, 255), highlight_pos, self.radius // 3)
    
    def reset(self, x: float, y: float):
        """공 리셋"""
        self.x = x
        self.y = y
        self.active = True
        angle = random.uniform(-math.pi/4, math.pi/4) - math.pi/2
        self.vx = self.speed * math.cos(angle)
        self.vy = self.speed * math.sin(angle)


class BreakoutGame:
    """벽돌깨기 게임 메인 클래스"""
    
    def __init__(self, config: Optional[GameConfig] = None):
        """
        게임 초기화
        
        Args:
            config: 게임 설정. None이면 기본값 사용
        """
        self.config = config or GameConfig()
        self.screen: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.running = False
        
        # 게임 오브젝트
        self.bricks: List[Brick] = []
        self.paddle: Optional[Paddle] = None
        self.ball: Optional[Ball] = None
        
        # 게임 상태
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_won = False
        
        # 외부 컨트롤러용 콜백
        self.on_paddle_position: Optional[Callable[[float], None]] = None
        self.on_paddle_angle: Optional[Callable[[float], None]] = None
    
    def initialize(self):
        """게임 초기화 (pygame 초기화 포함)"""
        pygame.init()
        self.screen = pygame.display.set_mode(
            (self.config.screen_width, self.config.screen_height)
        )
        pygame.display.set_caption("벽돌깨기 게임")
        self.clock = pygame.time.Clock()
        
        self._create_bricks()
        self._create_paddle()
        self._create_ball()
    
    def _create_bricks(self):
        """벽돌 생성"""
        self.bricks = []
        
        for row in range(self.config.brick_rows):
            color = self.config.brick_colors[row % len(self.config.brick_colors)]
            
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
                    shape=self.config.brick_shape
                )
                self.bricks.append(brick)
        
        # 테스트 모드: 벽돌 하나만 남기기
        if self.config.test_mode:
            self._leave_one_brick()
    
    def _leave_one_brick(self):
        """테스트용: 벽돌 하나만 남기고 나머지 비활성화"""
        if self.bricks:
            for brick in self.bricks[1:]:
                brick.active = False
    
    def _create_paddle(self):
        """패들 생성"""
        x = (self.config.screen_width - self.config.paddle_width) / 2
        y = self.config.screen_height - 50
        self.paddle = Paddle(x, y, self.config)
    
    def _create_ball(self):
        """공 생성"""
        x = self.config.screen_width / 2
        y = self.config.screen_height - 100
        self.ball = Ball(x, y, self.config)
    
    def handle_input(self):
        """키보드 입력 처리"""
        keys = pygame.key.get_pressed()
        
        # 좌우 이동
        if keys[pygame.K_LEFT]:
            self.paddle.move_left()
        if keys[pygame.K_RIGHT]:
            self.paddle.move_right()
        
        # 상하로 기울기 조절
        if keys[pygame.K_UP]:
            self.paddle.rotate_up()
        if keys[pygame.K_DOWN]:
            self.paddle.rotate_down()
    
    def update(self):
        """게임 상태 업데이트"""
        if self.game_over or self.game_won:
            return
        
        # 공 업데이트
        self.ball.update()
        
        # 패들 충돌 체크
        self.ball.check_paddle_collision(self.paddle)
        
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
                self.ball.reset(
                    self.config.screen_width / 2,
                    self.config.screen_height - 100
                )
        
        # 승리 조건 체크
        active_bricks = sum(1 for b in self.bricks if b.active)
        if active_bricks == 0:
            self.game_won = True
    
    def draw(self):
        """화면 그리기"""
        self.screen.fill(self.config.background_color)
        
        # 벽돌 그리기
        for brick in self.bricks:
            brick.draw(self.screen)
        
        # 패들 그리기
        self.paddle.draw(self.screen)
        
        # 공 그리기
        self.ball.draw(self.screen)
        
        # UI 그리기
        self._draw_ui()
        
        pygame.display.flip()
    
    def _draw_ui(self):
        """UI 요소 그리기"""
        font = pygame.font.Font(None, 36)
        
        # 점수
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        
        # 남은 생명
        lives_text = font.render(f"Lives: {self.lives}", True, (255, 255, 255))
        self.screen.blit(lives_text, (self.config.screen_width - 100, 10))
        
        # 패들 기울기 표시
        angle_text = font.render(f"Angle: {self.paddle.angle:.1f}°", True, (255, 255, 0))
        self.screen.blit(angle_text, (self.config.screen_width // 2 - 50, 10))
        
        # 게임 오버/승리 메시지
        if self.game_over:
            game_over_text = font.render("GAME OVER - Press R to Restart", True, (255, 0, 0))
            text_rect = game_over_text.get_rect(center=(self.config.screen_width/2, self.config.screen_height/2))
            self.screen.blit(game_over_text, text_rect)
        
        if self.game_won:
            win_text = font.render("YOU WIN! - Press R to Restart", True, (0, 255, 0))
            text_rect = win_text.get_rect(center=(self.config.screen_width/2, self.config.screen_height/2))
            self.screen.blit(win_text, text_rect)
    
    def reset(self):
        """게임 리셋"""
        self.score = 0
        self.lives = 3
        self.game_over = False
        self.game_won = False
        
        self._create_bricks()
        self._create_paddle()
        self._create_ball()
    
    def run(self):
        """게임 메인 루프 실행"""
        self.initialize()
        self.running = True
        
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
            
            # 게임 로직
            self.handle_input()
            self.update()
            self.draw()
            
            # FPS 제한
            self.clock.tick(self.config.fps)
        
        pygame.quit()
    
    # ===== 외부 연동용 메서드 (스켈레톤 포즈 등) =====
    
    def set_paddle_position(self, x: float):
        """
        외부에서 패들 위치 설정 (스켈레톤 포즈 연동용)
        
        Args:
            x: 패들의 x 좌표 (0 ~ screen_width)
        """
        if self.paddle:
            self.paddle.set_position(x)
    
    def set_paddle_angle(self, angle: float):
        """
        외부에서 패들 기울기 설정 (스켈레톤 포즈 연동용)
        
        Args:
            angle: 기울기 각도 (도 단위, -max_angle ~ max_angle)
        """
        if self.paddle:
            self.paddle.set_angle(angle)
    
    def get_game_state(self) -> dict:
        """
        현재 게임 상태 반환 (외부 연동용)
        
        Returns:
            게임 상태 딕셔너리
        """
        return {
            "score": self.score,
            "lives": self.lives,
            "game_over": self.game_over,
            "game_won": self.game_won,
            "ball_position": (self.ball.x, self.ball.y) if self.ball else None,
            "paddle_position": self.paddle.x if self.paddle else None,
            "paddle_angle": self.paddle.angle if self.paddle else None,
            "active_bricks": sum(1 for b in self.bricks if b.active)
        }


# ===== 메인 실행 =====
if __name__ == "__main__":
    # 기본 설정으로 게임 실행
    config = GameConfig(
        # 테스트 모드를 True로 하면 벽돌 하나만 남음
        test_mode=False,
        
        # 벽돌 모양 변경 가능: RECTANGLE, ROUNDED, DIAMOND
        brick_shape=BrickShape.RECTANGLE,
        
        # 기타 설정 커스터마이즈 가능
        # brick_rows=3,
        # brick_cols=8,
        # ball_speed=6.0,
        # paddle_max_angle=45.0,
    )
    
    game = BreakoutGame(config)
    game.run()
