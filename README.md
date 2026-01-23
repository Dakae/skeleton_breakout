# Skeleton Breakout 🎮

손 인식 기반 벽돌깨기 게임입니다. 웹캠으로 손을 인식하여 손의 골격 선분으로 공을 튕겨 벽돌을 깨는 게임입니다.

<!-- 아래에 게임 스크린샷 또는 GIF를 추가하세요 -->
<!-- ![Demo](./demo.gif) -->

## 주요 기능

- **손 인식 패들**: 기존 벽돌깨기의 패들 대신 손의 골격 선분이 공을 튕김
- **실시간 웹캠 배경**: 게임 배경에 웹캠 영상이 실시간으로 표시
- **물리 기반 반사**: 손 선분의 기울기에 따른 입사각 반사 계산
- **하드 모드**: 손 선분에 체력이 부여되어 충돌 시 체력 감소, 체력 소진 시 해당 손가락 그룹 비활성화
- **향상된 충돌 감지**: CCD(연속 충돌 감지), 서브스텝 시뮬레이션으로 정확한 충돌 처리

## 기술 스택

| 분류 | 기술 |
|------|------|
| Language | Python 3.8+ |
| Game Engine | Pygame |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Math | NumPy |

## 설치

### 1. 저장소 클론
```bash
git clone https://github.com/Dakae/skeleton_breakout.git
cd skeleton_breakout
```

### 2. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3. MediaPipe 모델 다운로드
`models/` 폴더에 아래 파일들을 다운로드하세요:
- [hand_landmarker.task](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)

## 실행

```bash
python skeleton_breakout.py
```

## 조작법

| 키 | 기능 |
|---|------|
| **마우스 클릭** | 메뉴에서 버튼 선택 |
| **Enter / Space** | 게임 시작 |
| **R** | 재시작 |
| **M** | 메인 메뉴로 돌아가기 |
| **F** | FPS 표시 토글 |
| **D** | 디버그 정보 토글 |
| **ESC** | 종료 |

## 게임 모드

### Normal Mode
- 기본 모드
- 손 선분이 사라지지 않음

### Hard Mode
- 각 손가락/손바닥 그룹에 체력(기본 3) 부여
- 공과 충돌할 때마다 체력 감소
- 체력이 0이 되면 해당 그룹 비활성화 (점선으로 표시)

## 프로젝트 구조

```
skeleton_breakout/
├── skeleton_breakout.py  # 메인 게임 파일
├── skeleton.py           # 스켈레톤 인식 모듈
├── breakOut.py           # 벽돌깨기 기본 클래스
├── models/               # MediaPipe 모델 파일
│   └── hand_landmarker.task
├── requirements.txt
└── README.md
```

## 개발 기간

**2025.01.19 ~ 2025.01.24** (6일)

## 라이선스

MIT License
