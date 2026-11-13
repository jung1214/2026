"""
Index light.py  -  손그림 캐릭터가 캠 속 내 동작을 따라 움직이는 프로그램 (ARAP 방식)

[무엇을 하나]
  - 캐릭터 그림에 삼각형 그물을 씌우고, 조종점 13개(머리·목·엉덩이·어깨·팔꿈치·손·무릎·발)를
    캠 속 내 관절에 맞춰 옮겨서 그림 전체를 자연스럽게 휘게 한다 (ARAP).
  - 그물은 외곽선을 따라 만들어서 팔과 몸통 사이 틈은 그물에 넣지 않는다.
  - 얼굴·손·발은 휘지 않고 통째로 회전만 한다. 팔은 항상 몸통 앞에 그린다.
  - 관절 좌표는 One Euro 필터로 떨림을 줄인다.
  - 캠 앞에서 좌우로 움직이면 캐릭터도 좌우로 이동한다.
  - 쪼그려 앉으면 몸이 내려가고, 점프할 때만 뛰어오른다. 디딘 발은 바닥에 붙어 있다.
  - 검지만 펴면(중지·약지·새끼는 접기) 캐릭터의 같은 쪽 손에서 불빛이 나온다.

[준비]
  - arap_dodge_game.py 와 같은 폴더에 두고 실행합니다. (ARAP 기본 코드를 가져다 씀)
  - 캐릭터: CHARACTER_FILE (기본 character.png). 최근 스캔 그림을 쓰려면 USE_LATEST_SCAN = True
  - 모델: pose_landmarker.task (이미 있음), hand_landmarker.task
          hand_landmarker.task 는 이 폴더 → pygame_project 폴더 순으로 찾고, 없으면 자동 다운로드
  - 다리·점프까지 쓰려면 발목까지 화면에 들어오게 서야 합니다 (상반신만 보이면 다리는 기본 자세)

실행:  python "Index light.py"     (보통은 scan and light.py 로 실행)

[조작]  C 지금 자리를 좌우 이동 기준으로 | D 그물·조종점 보기 | V 캠 화면 | ESC 종료
[개인정보] 캠 영상은 화면에만 표시하고 저장하지 않습니다.
"""
import collections
import math
import os
import random
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np
import pygame

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
try:
    import arap_dodge_game as game      # ArapMesh, pose_to_targets, open_camera 등 재사용
except ImportError as e:
    raise SystemExit(f"arap_dodge_game.py 를 불러오지 못했습니다: {e}\n"
                     "이 파일과 같은 폴더에 arap_dodge_game.py 가 있는지 확인하세요.")

if not game.MEDIAPIPE_OK:
    raise SystemExit("mediapipe가 설치되어 있지 않습니다 (pip install mediapipe)")
mp, mp_python, vision = game.mp, game.mp_python, game.vision

# ------------------------------------------------------------------ 설정값
SCREEN_W, SCREEN_H = 900, 640
CHAR_HEIGHT = 340            # 캐릭터 키(픽셀)
BG_COLOR = (18, 18, 32)

HOLD_FRAMES = 4              # 검지 자세가 캠 프레임으로 이만큼 연속돼야 불이 켜짐 (오발동 방지)
MISS_TOLERANCE = 3           # 손을 놓친 프레임이 이 이하면 그대로 켜진 상태 유지 (깜빡임 방지)

LIGHT_COLOR = (255, 205, 90)  # 불빛 색 (RGB, 따뜻한 노란빛)
LIGHT_RADIUS = 46            # 불빛 크기(픽셀)
REACH = 0.08                 # 캐릭터 손에서 검지 방향으로 나와 있는 거리 (캐릭터 키 대비, 0이면 손에 딱 붙음)
FOLLOW = 0.5                 # 불빛이 따라가는 빠르기 (0~1)
TRAIL_LEN = 12               # 꼬리 길이(프레임)
FADE_IN, FADE_OUT = 0.15, 0.05

PREVIEW_SIZE = (240, 180)

SAVE_DIR = os.path.join(BASE_DIR, "saved_characters")
USE_LATEST_SCAN = False      # True: 최근 스캔 그림 사용 / False: 아래 CHARACTER_FILE 사용

# 사용할 캐릭터 그림 (흰 배경은 실행할 때 자동으로 지움)
#   "character.png"      : 원래 마스코트 (팔 벌린 자세 → 조종점 자동으로 잡힘)
#   "standcharacter.png" : 팔 내린 마스코트 (아래 CUSTOM_POINTS 로 조종점 직접 지정)
CHARACTER_FILE = "character.png"

# 팔을 내리고 선 그림처럼 자동 추정이 틀리는 그림은 여기에 조종점을 직접 적는다.
# 값 = (가로 위치, 세로 위치)를 그림 크기 대비 비율(0~1)로. 화면 기준 l=왼쪽, r=오른쪽
CUSTOM_POINTS = {
    "standcharacter.png": {
        "head":   (0.494, 0.199), "neck":   (0.499, 0.346),
        "sh_l":   (0.280, 0.392), "sh_r":   (0.724, 0.392),     # 어깨 (팔 회전 중심)
        "hip":    (0.505, 0.679),
        "hand_l": (0.137, 0.656), "hand_r": (0.867, 0.656),
        "foot_l": (0.291, 0.946), "foot_r": (0.752, 0.953),
    },
}

# --- 그물 (ARAP) ---
MESH_STEP = 7                # 그물 안쪽 점 간격(픽셀). 작을수록 부드럽지만 느려짐
OUTLINE_STEP = 3             # 외곽선 위 점 간격(픽셀). 좁은 틈을 살리려면 작게
KNEE_RATIO = 0.5             # 무릎 높이: 엉덩이(0) ~ 발(1) 사이 어디쯤인지
FIX_HEAD = True              # True: 얼굴(목 위쪽)은 휘지 않고 통째로 움직임 / False: 예전처럼 같이 휘어짐
HEAD_MARGIN = 2              # 목 높이에서 이만큼(px) 위부터를 머리로 봄
FIX_HANDS_FEET = True        # True: 손·발도 얼굴처럼 휘지 않고 통째로 회전
HAND_REGION = 0.7            # 아래팔(팔꿈치→손) 길이의 몇 % 지점부터를 손으로 볼지
FOOT_REGION = 0.7            # 종아리(무릎→발) 길이의 몇 % 지점부터를 발로 볼지
SCAN_TRIM = 4                # 스캔한 그림: 바깥쪽 종이색 띠를 몇 픽셀까지 지울지 (흰 띠가 남으면 키우기)

# --- 떨림 줄이기 ---
SMOOTH_FILTER = True         # True: One Euro 필터로 관절 떨림 줄이기
EURO_MIN_CUTOFF = 1.0        # 작을수록 가만히 있을 때 덜 떨림 (대신 느려짐). 떨리면 0.5
EURO_BETA = 0.05             # 클수록 빠른 동작을 빨리 따라감. 늦게 따라오면 0.1
TARGET_SMOOTH = 0.7          # 필터를 쓸 때 캐릭터가 목표 자세로 다가가는 빠르기 (0~1)

# --- 팔 각도 ---
ARM_REMAP = True             # True: 내가 팔을 내리면 캐릭터 팔은 몸에서 살짝 떨어진 자연스러운 각도로 (몸 안으로 안 파고듦)
ARM_HANG_MIN = 10            # 캐릭터가 팔을 내렸을 때 각도 최소값 (도, 아래 방향 기준 바깥쪽)
ARM_HANG_MAX = 30            # 최대값. 팔이 아직 몸에 겹치면 키우기

# --- 좌우 이동 ---
FOLLOW_X = True              # True: 캠 속에서 좌우로 움직이면 캐릭터도 좌우로 이동
MOVE_REF = 0.5               # 기준 위치(캠 화면 가로 0~1, 0.5 = 가운데). 실행 중 C 키로 지금 자리를 기준으로
MOVE_RANGE = 0.25            # 기준에서 캠 화면 폭의 이만큼 옆으로 가면 캐릭터가 화면 끝에 닿음
MOVE_SMOOTH = 0.25           # 따라가는 빠르기 (0~1, 클수록 즉시)

# --- 바닥 · 앉기 · 점프 ---
GROUND_FEET = True           # True: 디딘 발은 바닥에 붙이고 다리(무릎)로 높이를 맞춤
SQUAT = True                 # True: 쪼그려 앉으면 캐릭터 몸도 내려감 (위로는 점프할 때만)
SQUAT_GAIN = 1.0             # 앉는 깊이 배율 (키우면 조금만 앉아도 깊이 앉음)
SQUAT_MAX = 0.6              # 최대로 내려가는 깊이 (캐릭터 다리 길이 배)
SQUAT_WINDOW = 6.0           # 서 있을 때 다리 길이를 최근 몇 초 기록으로 정할지 (이보다 오래 앉아 있으면 서서히 '서 있음'으로 바뀜)
ANKLE_VIS = 0.7              # 발목 인식 확실도가 이보다 낮으면 앉기·점프 판단 안 함
EDGE_MARGIN = 0.03           # 발목이 캠 화면 아래 끝 3% 안에 있으면 판단 안 함 (발이 잘린 것)
SQUAT_DEADZONE = 0.06        # 이보다 조금 내려간 건 무시 (흔들림 방지)
FOOT_UP_THRESHOLD = 0.1      # 캠 속 발목이 바닥보다 몸통 길이의 이만큼 이상 올라가면 '발을 든 것'
JUMP = True                  # True: 캠 속에서 점프하면 캐릭터도 뛰어오름 (발목까지 보여야 함)
JUMP_GAIN = 1.5              # 점프 높이 과장 배율 (캐릭터 다리가 짧아서 1보다 크게)
JUMP_THRESHOLD = 0.12        # 몸통 길이의 이만큼 이상 발이 떠야 점프로 봄 (잘못 뛰면 키우기)
JUMP_SCALE_TOL = 0.12        # 캠 속 어깨너비가 이 비율 이상 바뀌면 앞뒤로 움직인 것으로 보고 기준을 다시 잡음
JUMP_BODY_RATIO = 0.5        # 점프로 보려면 어깨도 발이 뜬 높이의 이 비율 이상 같이 올라가야 함
JUMP_COOLDOWN = 0.8          # 앞뒤로 움직였다고 판단한 뒤 이 시간(초) 동안은 점프로 보지 않음
JUMP_CONFIRM = 0.06          # 이 시간(초) 이상 떠 있어야 점프로 인정 (순간적인 인식 튐 무시)
JUMP_MAX_TIME = 0.9          # 이보다 오래 떠 있으면 점프가 아니라 뒤로 물러선 것으로 보고 바닥을 다시 잡음
JUMP_MAX = 1.2               # 최대 점프 높이 (몸통 길이 배)

POSE_MODEL_PATHS = [game.POSE_MODEL_PATH]
HAND_MODEL_PATHS = [
    os.path.join(BASE_DIR, "hand_landmarker.task"),
    os.path.join(BASE_DIR, "..", "pygame_project", "pygame_project", "hand_landmarker.task"),
]
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")


# ------------------------------------------------------------------ 검지 판정
def is_pointing(h):
    """검지만 펴고 중지·약지·새끼는 접었는지. 엄지는 사람마다 습관이 달라서 무시.
    손가락 끝이 손목에서 둘째 마디(PIP)보다 멀면 '폄' — 손이 기울어도 동작함."""
    d = lambda a, b: np.linalg.norm(h[a] - h[b])
    up = lambda tip, pip: d(tip, 0) > d(pip, 0) * 1.1
    return up(8, 6) and not up(12, 10) and not up(16, 14) and not up(20, 18)


def hand_side(h, pts, fw):
    """이 손이 캐릭터의 어느 손('l' 화면 왼쪽 / 'r' 화면 오른쪽)인지.
    pose_to_targets 와 같은 기준(화면 왼쪽에 보이는 어깨 쪽 = 'l')으로 맞춘다."""
    wrist = h[0]
    if pts is not None:
        a, b = game.SIDE_A, game.SIDE_B
        left, right = (a, b) if pts[a["sh"], 0] < pts[b["sh"], 0] else (b, a)
        dl = np.linalg.norm(pts[left["wr"], :2] - wrist)
        dr = np.linalg.norm(pts[right["wr"], :2] - wrist)
        return "l" if dl <= dr else "r"
    return "l" if wrist[0] < fw / 2 else "r"


# ------------------------------------------------------------------ 모델 불러오기
def load_model(paths, url):
    for p in paths:
        if os.path.exists(p):
            path = p
            break
    else:
        path = paths[0]
        print(f"모델 다운로드 중... ({os.path.basename(path)})")
        urllib.request.urlretrieve(url, path)
    with open(path, "rb") as f:          # 한글 경로 대비: 파일 내용을 직접 넘김
        return f.read(), path


def create_landmarker(kind, model_bytes, path):
    def options(base):
        if kind == "pose":
            return vision.PoseLandmarkerOptions(
                base_options=base, running_mode=vision.RunningMode.VIDEO, num_poses=1,
                min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4)
        return vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5)

    cls = vision.PoseLandmarker if kind == "pose" else vision.HandLandmarker
    try:
        return cls.create_from_options(options(mp_python.BaseOptions(model_asset_buffer=model_bytes)))
    except Exception:                    # 버전에 따라 안 되면 파일 경로로 다시 시도
        return cls.create_from_options(options(mp_python.BaseOptions(model_asset_path=path)))


# ------------------------------------------------------------------ 떨림 줄이기: One Euro 필터
class OneEuro:
    """좌표 떨림을 줄이는 필터 (Casiez et al., 2012).
    천천히 움직일 때는 강하게 걸러서 떨림을 없애고, 빠르게 움직일 때는 약하게 걸러서 늦게 따라오지 않는다.
      min_cutoff: 작을수록 가만히 있을 때 더 안 떨림(대신 살짝 느려짐)
      beta      : 클수록 빠른 동작에 더 빨리 따라감"""

    def __init__(self, min_cutoff=1.0, beta=0.02, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        x = np.asarray(x, float)
        if self.x is None or t - self.t > 0.5:          # 처음이거나 오래 끊겼으면 그대로 시작
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = self.dx + a_d * (dx - self.dx)
        speed = np.linalg.norm(self.dx.reshape(-1, 2), axis=1)          # 점마다 속도
        cutoff = self.min_cutoff + self.beta * speed
        a = self._alpha(cutoff, dt)[:, None].repeat(2, 1).reshape(x.shape)
        self.x = self.x + a * (x - self.x)
        self.t = t
        return self.x.copy()


# ------------------------------------------------------------------ 캠: 몸 + 손 읽기 (별도 스레드)
class BodyHandTracker:
    def __init__(self):
        self.ok, self.error = False, ""
        self.lock = threading.Lock()
        self.running, self.thread = False, None
        self.cap = self.pose_lm = self.hand_lm = None

        self.pose = None                          # {"time", "pts"}
        self.light_on = {"l": False, "r": False}  # 검지 불빛 켜짐 여부
        self.point_dir = {"l": None, "r": None}   # 검지 방향(단위벡터)
        self.preview = None
        self._count = {"l": 0, "r": 0}
        self._miss = {"l": 0, "r": 0}
        self._dir = {"l": None, "r": None}
        try:
            self.pose_lm = create_landmarker("pose", *load_model(POSE_MODEL_PATHS, game.POSE_MODEL_URL))
            self.hand_lm = create_landmarker("hand", *load_model(HAND_MODEL_PATHS, HAND_MODEL_URL))
            self.cap = game.open_camera()
            if self.cap is None:
                raise RuntimeError("웹캠을 열 수 없습니다 (다른 프로그램이 사용 중이거나 권한이 없을 수 있음)")
            self.ok = True
        except Exception as e:
            self.error = str(e)

    def start(self):
        if self.ok:
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _loop(self):
        last_ts = 0
        self._filter = OneEuro(EURO_MIN_CUTOFF, EURO_BETA)
        while self.running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1)                       # 거울처럼 좌우 반전
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            fh, fw = rgb.shape[:2]
            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            try:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                pr = self.pose_lm.detect_for_video(image, ts)
                hr = self.hand_lm.detect_for_video(image, ts)
            except Exception as e:
                self.error = str(e)
                continue

            # 몸
            pts, pose = None, None
            if pr.pose_landmarks:
                pts = np.array([[p.x * fw, p.y * fh,
                                 p.visibility if p.visibility is not None else 1.0]
                                for p in pr.pose_landmarks[0]])
                if SMOOTH_FILTER:                            # 관절 좌표 떨림 줄이기
                    pts[:, :2] = self._filter(pts[:, :2], time.monotonic())
                if min(pts[11, 2], pts[12, 2]) >= 0.3:
                    pose = {"time": time.monotonic(), "pts": pts, "fh": fh,
                            "cx": float((pts[11, 0] + pts[12, 0]) / 2 / fw)}   # 어깨 가운데 가로 위치(0~1)

            # 손 → 캐릭터의 어느 손인지 짝짓기
            hands = [np.array([[p.x * fw, p.y * fh] for p in h]) for h in (hr.hand_landmarks or [])]
            by_side = {}
            for h in hands:
                s = hand_side(h, pts, fw)
                if s in by_side:                             # 두 손이 같은 쪽에 붙으면 반대쪽으로
                    s = "r" if s == "l" else "l"
                by_side[s] = h

            # 검지 판정 (연속 프레임 + 잠깐 놓쳐도 유지)
            pointing_now = {}
            for s in "lr":
                h = by_side.get(s)
                pointing_now[s] = h is not None and is_pointing(h)
                if pointing_now[s]:
                    self._count[s] += 1
                    self._miss[s] = 0
                    v = h[8] - h[5]                          # 검지 뿌리 → 끝
                    n = np.linalg.norm(v)
                    if n > 1e-6:
                        self._dir[s] = v / n
                else:
                    self._miss[s] += 1
                    if self._miss[s] > MISS_TOLERANCE:
                        self._count[s] = 0

            # 미리보기: 손 점 + 검지 끝 표시
            view = rgb.copy()
            for s, h in by_side.items():
                for p in h:
                    cv2.circle(view, (int(p[0]), int(p[1])), 3, (200, 200, 200), -1)
                color = (255, 205, 90) if pointing_now[s] else (120, 120, 255)
                cv2.circle(view, (int(h[8][0]), int(h[8][1])), 10, color, -1)
            small = cv2.resize(view, PREVIEW_SIZE)

            with self.lock:
                if pose is not None:
                    self.pose = pose
                self.light_on = {s: self._count[s] >= HOLD_FRAMES for s in "lr"}
                self.point_dir = dict(self._dir)
                self.preview = small.tobytes()

    def get(self):
        with self.lock:
            return self.pose, dict(self.light_on), dict(self.point_dir), self.preview

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        for lm in (self.pose_lm, self.hand_lm):
            if lm is not None:
                try:
                    lm.close()
                except Exception:
                    pass


# ------------------------------------------------------------------ 불빛 효과
def make_glow(radius, color):
    """가운데는 하얗고 바깥으로 갈수록 색이 퍼지는 동그란 빛 (검은 바탕 → 더하기 합성용)"""
    size = radius * 2
    y, x = np.mgrid[0:size, 0:size] - radius + 0.5
    d = np.sqrt(x * x + y * y) / radius
    halo = np.exp(-(d * 2.2) ** 2)
    core = np.exp(-(d * 7.0) ** 2)
    arr = np.zeros((size, size, 3))
    for c in range(3):
        arr[..., c] = halo * color[c] + core * 255
    return pygame.surfarray.make_surface(np.clip(arr, 0, 255).astype(np.uint8))


class HandLight:
    def __init__(self, screen_size):
        self.glows = [make_glow(max(4, int(LIGHT_RADIUS * k)), LIGHT_COLOR)
                      for k in np.linspace(0.25, 1.0, 8)]
        self.overlay = pygame.Surface(screen_size)
        self.trail = collections.deque(maxlen=TRAIL_LEN)
        self.sparks = []                 # [x, y, vx, vy, life]
        self.pos = None
        self.intensity = 0.0

    def update(self, on, target):
        if on and target is not None:
            t = np.asarray(target, float)
            self.pos = t if self.pos is None else self.pos + (t - self.pos) * FOLLOW
            self.intensity = min(1.0, self.intensity + FADE_IN)
            for _ in range(2):           # 반짝이 가루
                a = random.uniform(0, 2 * math.pi)
                s = random.uniform(0.5, 2.5)
                self.sparks.append([self.pos[0], self.pos[1], math.cos(a) * s, math.sin(a) * s - 0.8, 25])
        else:
            self.intensity = max(0.0, self.intensity - FADE_OUT)

        if self.pos is not None and self.intensity > 0:
            self.trail.append(self.pos.copy())
        else:
            self.trail.clear()
            self.pos = None

        for sp in self.sparks:
            sp[0] += sp[2]
            sp[1] += sp[3]
            sp[3] += 0.05
            sp[4] -= 1
        self.sparks = [sp for sp in self.sparks if sp[4] > 0]

    def draw(self, screen):
        if not self.trail and not self.sparks:
            return
        ov = self.overlay
        ov.fill((0, 0, 0))
        n = len(self.trail)
        for i, p in enumerate(self.trail):              # 오래된 꼬리일수록 작게
            g = self.glows[int((i + 1) / n * (len(self.glows) - 1))]
            ov.blit(g, (p[0] - g.get_width() / 2, p[1] - g.get_height() / 2),
                    special_flags=pygame.BLEND_RGB_ADD)
        for sp in self.sparks:
            k = sp[4] / 25
            col = tuple(int(c * k) for c in LIGHT_COLOR)
            pygame.draw.circle(ov, col, (int(sp[0]), int(sp[1])), 2)
        if self.intensity < 1.0:                        # 켜지고 꺼질 때 밝기 조절
            v = int(255 * self.intensity)
            ov.fill((v, v, v), special_flags=pygame.BLEND_RGB_MULT)
        screen.blit(ov, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


# ------------------------------------------------------------------ 캐릭터 불러오기
def find_character_path():
    """USE_LATEST_SCAN=True 면 최근 스캔 그림 → 아니면 CHARACTER_FILE → 없으면 character.png"""
    if USE_LATEST_SCAN and os.path.isdir(SAVE_DIR):
        for name in sorted(os.listdir(SAVE_DIR), reverse=True):
            path = os.path.join(SAVE_DIR, name, "texture.png")
            if os.path.exists(path):
                return path
    path = os.path.join(BASE_DIR, CHARACTER_FILE)
    if os.path.exists(path):
        return path
    return game.find_character_png()


def custom_points(ratios, alpha):
    """CUSTOM_POINTS 의 비율 값을 실제 그림 크기의 좌표로 바꿔 돌려준다"""
    h, w = alpha.shape[:2]
    return {k: np.array([fx * w, fy * h]) for k, (fx, fy) in ratios.items()}


# 조종점 13개: 원래 7개(머리·목·엉덩이·양손·양발) + 양쪽 어깨·팔꿈치·무릎
#  → 손끝·발끝만 끌면 팔다리가 고무처럼 늘어나지만, 어깨·팔꿈치·무릎까지 잡아 주면
#    팔다리가 관절을 축으로 두 마디로 굽혀진다.
HANDLES = ["head", "neck", "hip", "sh_l", "sh_r", "el_l", "el_r",
           "hand_l", "hand_r", "kn_l", "kn_r", "foot_l", "foot_r"]
game.HANDLES = HANDLES          # ArapMesh 가 이 목록대로 조종점을 잡도록


def geodesic(mask, start):
    """그림(mask) 안에서만 움직였을 때 start 점부터의 거리(픽셀 단위 걸음 수). 그림 밖은 -1"""
    h, w = mask.shape
    dist = -np.ones((h, w), np.int32)
    sy, sx = int(start[1]), int(start[0])
    if not mask[sy, sx]:                                   # 시작점이 그림 밖이면 가장 가까운 그림 점으로
        ys, xs = np.nonzero(mask)
        k = np.argmin((ys - sy) ** 2 + (xs - sx) ** 2)
        sy, sx = ys[k], xs[k]
    dist[sy, sx] = 0
    q = collections.deque([(sy, sx)])
    while q:
        y, x = q.popleft()
        d = dist[y, x] + 1
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = d
                q.append((ny, nx))
    return dist


def auto_points(alpha):
    """그림 모양만 보고 조종점 13개를 자동으로 잡는다. 팔을 벌린 그림 / 내린 그림 모두 대응.

    1) 그림 안에 들어가는 가장 큰 원을 찾아서
       - 그림 맨 위에 닿는 가장 큰 원 = 머리, 그 아래에서 가장 큰 원 = 몸통
       - 머리와 몸통 사이에서 가장 잘록한 곳 = 목
    2) 어깨 = 몸통 원의 위쪽 양옆
    3) 손 = 그림 안으로만 따라갔을 때 어깨에서 가장 멀리 떨어진 점
            (엉덩이보다 아래는 보지 않음 → 꼬리·다리를 손으로 잡지 않음)
    4) 팔꿈치 = 어깨→손으로 이어지는 길의 가운데
    5) 발 = 맨 아래 좌우 / 엉덩이 = 목과 발 사이
    6) 무릎 = 발→엉덩이로 이어지는 길 위에서, 엉덩이와 발 높이의 중간쯤"""
    mask = alpha > 40
    m8 = mask.astype(np.uint8)
    ys, xs = np.nonzero(mask)
    y0, y1 = ys.min(), ys.max()
    h = y1 - y0 + 1
    H, W = mask.shape
    dt = cv2.distanceTransform(m8, cv2.DIST_L2, 5)
    Y, X = np.mgrid[0:H, 0:W]

    # 발
    low = ys > y1 - 0.07 * h
    lx, ly = xs[low], ys[low]
    cx = (xs.min() + xs.max()) / 2
    if (lx < cx).any() and (lx >= cx).any():
        foot_l = np.array([lx[lx < cx].mean(), ly[lx < cx].mean()])
        foot_r = np.array([lx[lx >= cx].mean(), ly[lx >= cx].mean()])
    else:
        foot_l = np.array([lx.mean() - 0.08 * h, ly.mean()])
        foot_r = np.array([lx.mean() + 0.08 * h, ly.mean()])
    foot_y = (foot_l[1] + foot_r[1]) / 2

    # 머리 원 / 몸통 원
    def biggest_circle(region):
        d = np.where(region, dt, 0)
        k = np.unravel_index(np.argmax(d), d.shape)
        return np.array([k[1], k[0]], float), float(d[k])

    head_c, head_r = biggest_circle((Y - dt) < y0 + 0.12 * h)      # 원의 윗부분이 그림 맨 위에 닿는 것 중 가장 큰 원
    torso_c, torso_r = biggest_circle((Y > head_c[1] + head_r * 0.9) & (Y < y0 + 0.85 * h))

    # 목: 머리 중심 → 몸통 중심 선 위에서 가장 잘록한 곳
    t = np.linspace(0, 1, 60)
    line = head_c + (torso_c - head_c) * t[:, None]
    widths = dt[np.clip(line[:, 1].astype(int), 0, H - 1), np.clip(line[:, 0].astype(int), 0, W - 1)]
    inner = (t > 0.25) & (t < 0.9)
    neck = line[inner][np.argmin(widths[inner])]
    hip = np.array([torso_c[0], neck[1] + 0.55 * (foot_y - neck[1])])
    head = head_c

    pts = {"head": head, "neck": neck, "hip": hip, "foot_l": foot_l, "foot_r": foot_r}
    for s, sign in (("l", -1), ("r", 1)):
        # 어깨: 몸통 원 위쪽 옆 (그림 안의 가장 가까운 점으로 맞춤)
        guess = np.array([torso_c[0] + sign * torso_r * 0.9, neck[1] + torso_r * 0.35])
        d2 = (X - guess[0]) ** 2 + (Y - guess[1]) ** 2
        d2[~mask] = 1e12
        k = np.unravel_index(np.argmin(d2), d2.shape)
        sh = np.array([k[1], k[0]], float)

        # 손: 어깨에서 그림 안으로 가장 멀리 떨어진 점 (자기 쪽, 머리 아래 ~ 엉덩이 높이까지)
        d_sh = geodesic(mask, sh)
        out = (X - torso_c[0]) * sign                        # 몸통 중심에서 바깥쪽으로 떨어진 정도
        ok = mask & (out > torso_r * 1.0) & (Y > head_c[1]) & (Y < hip[1] + 0.05 * h) & (d_sh >= 0)
        if not ok.any():
            ok = mask & (out > 0) & (d_sh >= 0)
        score = np.where(ok, d_sh + 0.5 * out, -1)          # 멀고 + 바깥쪽일수록 손
        k = np.unravel_index(np.argmax(score), score.shape)

        # 손 → 어깨 길 따라가기
        path = []
        y, x = k
        while True:
            path.append((x, y))
            if d_sh[y, x] <= 0:
                break
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1),
                           (y - 1, x - 1), (y - 1, x + 1), (y + 1, x - 1), (y + 1, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and 0 <= d_sh[ny, nx] < d_sh[y, x]:
                    y, x = ny, nx
                    break
            else:
                break
        path = np.array(path, float)
        hand = path[min(len(path) - 1, int(0.03 * h))]        # 손끝보다 살짝 안쪽(손바닥)
        el = path[len(path) // 2]
        pts.update({"sh_" + s: sh, "el_" + s: el, "hand_" + s: hand})

    # 무릎: 발에서 엉덩이까지 그림 안으로 이어지는 길 위에서, 엉덩이~발 높이의 KNEE_RATIO 지점
    d_hip = geodesic(mask, hip)
    knee_y = hip[1] + KNEE_RATIO * (foot_y - hip[1])
    for s in "lr":
        fx, fy = pts["foot_" + s]
        d2 = (X - fx) ** 2 + (Y - fy) ** 2
        d2[~mask | (d_hip < 0)] = 1e12
        y, x = np.unravel_index(np.argmin(d2), d2.shape)
        best = (abs(y - knee_y), x, y)
        while d_hip[y, x] > 0:                               # 발 → 엉덩이 쪽으로 한 칸씩
            for ny, nx in ((y - 1, x), (y - 1, x - 1), (y - 1, x + 1), (y, x - 1), (y, x + 1),
                           (y + 1, x), (y + 1, x - 1), (y + 1, x + 1)):
                if 0 <= ny < H and 0 <= nx < W and 0 <= d_hip[ny, nx] < d_hip[y, x]:
                    y, x = ny, nx
                    break
            else:
                break
            if abs(y - knee_y) < best[0]:
                best = (abs(y - knee_y), x, y)
        pts["kn_" + s] = np.array([best[1], best[2]], float)
    return pts


def complete_points(pts):
    """직접 지정한 조종점에 팔꿈치·무릎이 없으면 채움
    (팔꿈치 = 어깨와 손의 중간 / 무릎 = 발 바로 위, 엉덩이~발 높이의 KNEE_RATIO 지점)"""
    pts = dict(pts)
    for s in "lr":
        if "el_" + s not in pts:
            pts["el_" + s] = (pts["sh_" + s] + pts["hand_" + s]) / 2
        if "kn_" + s not in pts:
            foot, hip = pts["foot_" + s], pts["hip"]
            pts["kn_" + s] = np.array([foot[0] + 0.15 * (hip[0] - foot[0]),
                                       hip[1] + KNEE_RATIO * (foot[1] - hip[1])])
    return pts


def pose_to_targets(pts, rest, lift=0.0, floor_cam=None, sink=0.0):
    """캠에서 읽은 몸 좌표(pts)를 캐릭터 조종점 13개의 목표 위치로 바꾼다.
    사람과 캐릭터는 비율이 달라서 '방향'만 가져오고 길이는 캐릭터의 팔·다리 길이를 쓴다.
    팔: 어깨는 몸통에 붙어서 같이 기울고, 위팔(어깨→팔꿈치)·아래팔(팔꿈치→손)은
        내 팔의 방향을 그대로 따라간다.
    다리: 허벅지(→무릎)·종아리(무릎→발)가 내 다리의 방향을 그대로 따라가고,
        디딘 발은 바닥에 붙인다. 쪼그려 앉으면 몸이 sink(px)만큼 내려가고, 점프(lift)할 때만 위로.
        pts가 None이면 기본 자세."""
    SIDE_A, SIDE_B, VIS = game.SIDE_A, game.SIDE_B, game.VIS
    rotate, clamp, MAX_LEAN = game.rotate, game.clamp, game.MAX_LEAN
    hip0 = rest["hip"]
    hip = hip0 + np.array([0.0, sink])                      # 쪼그려 앉으면 sink(px)만큼 몸이 내려감
    upper = {s: np.linalg.norm(rest["el_" + s] - rest["sh_" + s]) for s in "lr"}
    fore = {s: np.linalg.norm(rest["hand_" + s] - rest["el_" + s]) for s in "lr"}
    rest_top = {s: np.array([rest["kn_" + s][0], hip0[1]]) for s in "lr"}
    thigh = {s: np.linalg.norm(rest["kn_" + s] - rest_top[s]) for s in "lr"}
    shin = {s: np.linalg.norm(rest["foot_" + s] - rest["kn_" + s]) for s in "lr"}
    leg_top = {s: np.array([rest["kn_" + s][0], hip[1]]) for s in "lr"}   # 다리가 몸에 붙는 곳
    # 캐릭터가 팔을 '자연스럽게 내렸을 때'의 각도 (아래 방향 기준, 바깥쪽 +). 그림의 팔 각도에서 정하되
    # ARM_HANG_MIN~ARM_HANG_MAX 도로 제한 (팔 벌린 그림이라도 팔을 내리면 몸에서 살짝 떨어진 정도로)
    arm_hang = {}
    for s in "lr":
        v = rest["hand_" + s] - rest["sh_" + s]
        out = v[0] if s == "r" else -v[0]
        arm_hang[s] = clamp(math.degrees(math.atan2(out, v[1])), ARM_HANG_MIN, ARM_HANG_MAX)

    lean = head_ang = 0.0
    arms, legs = {}, {}
    feet_down = {}                                          # 발마다 바닥에 붙어 있는지
    if pts is not None:
        sides = ({"l": SIDE_A, "r": SIDE_B} if pts[SIDE_A["sh"], 0] < pts[SIDE_B["sh"], 0]
                 else {"l": SIDE_B, "r": SIDE_A})
        sh_l, sh_r = pts[sides["l"]["sh"], :2], pts[sides["r"]["sh"], :2]
        mid_sh = (sh_l + sh_r) / 2

        hip_l, hip_r = pts[sides["l"]["hip"]], pts[sides["r"]["hip"]]
        hips_seen = min(hip_l[2], hip_r[2]) >= VIS
        if hips_seen:
            up = mid_sh - (hip_l[:2] + hip_r[:2]) / 2
            lean = math.atan2(up[0], -up[1])
        else:
            line = sh_r - sh_l
            lean = 1.5 * math.atan2(line[1], line[0])
        lean = clamp(lean, -MAX_LEAN, MAX_LEAN)

        if pts[game.NOSE, 2] >= VIS:
            hv = pts[game.NOSE, :2] - mid_sh
            head_ang = clamp(math.atan2(hv[0], -hv[1]) * 1.2, -MAX_LEAN * 1.3, MAX_LEAN * 1.3)
        else:
            head_ang = lean

        for s in "lr":
            sh, el, wr = (pts[sides[s][k]] for k in ("sh", "el", "wr"))
            if wr[2] >= VIS:
                if el[2] >= VIS:                            # 팔꿈치까지 보이면 팔을 두 마디로
                    u, f = el[:2] - sh[:2], wr[:2] - el[:2]
                else:                                       # 안 보이면 곧게 편 팔로
                    u = f = wr[:2] - sh[:2]
                u = u / max(np.linalg.norm(u), 1e-6)
                f = f / max(np.linalg.norm(f), 1e-6)
                arms[s] = (u, f)
            hp, kn, an = (pts[sides[s][k]] for k in ("hip", "knee", "ank"))
            if hips_seen and an[2] >= VIS and kn[2] >= VIS:   # 무릎·발목까지 보이면 다리를 두 마디로
                u, f = kn[:2] - hp[:2], an[:2] - kn[:2]
                legs[s] = (u / max(np.linalg.norm(u), 1e-6), f / max(np.linalg.norm(f), 1e-6))
                # 이 발이 바닥에 붙어 있는지 (캠 속 바닥 높이와 비교)
                if floor_cam is not None:
                    torso_cam = max(np.linalg.norm(mid_sh - (hip_l[:2] + hip_r[:2]) / 2), 1.0)
                    feet_down[s] = (floor_cam - an[1]) / torso_cam < FOOT_UP_THRESHOLD

    turn = lambda p: hip + rotate(p - hip0, lean)           # 몸 기울기만큼 돌리기 (+ 앉은 만큼 내림)
    targets = {"hip": hip.copy()}
    targets["neck"] = turn(rest["neck"])
    targets["head"] = targets["neck"] + rotate(rest["head"] - rest["neck"], head_ang)
    for s in "lr":
        shoulder = turn(rest["sh_" + s])
        targets["sh_" + s] = shoulder
        if s in arms:
            u, f = arms[s]
            if ARM_REMAP:                                   # 팔 내린 자세 ↔ 캐릭터의 자연스러운 팔 각도 맞추기
                hang = arm_hang[s]
                u, f = remap_arm(u, s, hang), remap_arm(f, s, hang)
            elbow = shoulder + u * upper[s]
            targets["el_" + s] = elbow
            targets["hand_" + s] = elbow + f * fore[s]
        else:                                               # 손이 안 보이면 기본 자세(기울기만 반영)
            targets["el_" + s] = turn(rest["el_" + s])
            targets["hand_" + s] = turn(rest["hand_" + s])
        if s in legs:
            u, f = legs[s]
            knee = leg_top[s] + u * thigh[s]
            foot = knee + f * shin[s]
            if GROUND_FEET:
                knee, foot = plant_leg(leg_top[s], knee, foot, thigh[s], shin[s], rest["foot_" + s][1],
                                       feet_down.get(s, True))
            targets["kn_" + s] = knee
            targets["foot_" + s] = foot
        else:                                               # 다리가 안 보이면 서 있는 자세
            targets["kn_" + s] = rest["kn_" + s].copy()
            targets["foot_" + s] = rest["foot_" + s].copy()

    # 몸 높이는 고정(위아래로 안 움직임). 점프할 때만 몸 전체를 lift(px)만큼 위로
    if lift > 0:
        for k in targets:
            targets[k] = targets[k] - np.array([0.0, lift])
    return np.array([targets[n] for n in HANDLES])


def remap_arm(v, side, hang):
    """팔 방향 v(캠 속 내 팔)를 캐릭터 팔 방향으로 바꾼다.
    사람은 팔을 내리면 거의 수직(0도)이지만, 캐릭터는 몸통이 넓어서 그대로 따라 하면 팔이 몸 안쪽으로 파고든다.
    → 내 팔 각도 0도(내림)를 캐릭터의 hang 도로, 180도(머리 위)는 그대로 180도로 맞추고 그 사이는 비례.
      팔을 몸 안쪽으로 모으면(음수 각도) 그만큼 그대로 안쪽으로 (팔짱·박수 같은 동작은 살림)"""
    sign = 1.0 if side == "r" else -1.0                 # 바깥쪽 방향: 오른쪽 팔은 +x, 왼쪽 팔은 -x
    a = math.degrees(math.atan2(v[0] * sign, v[1]))     # 아래 방향 기준 각도 (바깥쪽 +)
    m = hang + a * (180 - hang) / 180 if a >= 0 else hang + a
    r = math.radians(m)
    return np.array([math.sin(r) * sign, math.cos(r)])


def plant_leg(top, knee, foot, thigh, shin, floor_y, on_floor=True):
    """다리 한쪽을 바닥에 맞춘다 (몸 높이는 그대로 두고 다리만 조정).
    - 캠 속에서 발이 바닥에 붙어 있으면 캐릭터 발도 바닥 높이에 붙임
    - 발을 들었으면 든 높이 그대로 두되, 바닥 아래로는 안 내려가게
    - 허벅지·종아리 길이는 유지한 채, 무릎이 굽는 방향은 캠 속 내 무릎 방향을 따름"""
    foot = foot.copy()
    if on_floor or foot[1] > floor_y:
        foot[1] = floor_y
    v = foot - top
    d = np.linalg.norm(v)
    reach = thigh + shin
    if d < 1e-6:
        return knee, foot
    if d >= reach:                                      # 다리 길이보다 멀면 쭉 편 다리로
        u = v / d
        return top + u * thigh, top + u * reach
    d = max(d, abs(thigh - shin) + 1e-3)
    u = v / d
    n = np.array([-u[1], u[0]])
    cos_a = np.clip((thigh ** 2 + d ** 2 - shin ** 2) / (2 * thigh * d), -1, 1)
    sin_a = math.sqrt(1 - cos_a ** 2)
    side = 1.0 if np.dot(knee - top, n) >= 0 else -1.0   # 무릎이 굽는 쪽 = 캠 속 무릎이 있는 쪽
    return top + u * thigh * cos_a + n * thigh * sin_a * side, foot


class JumpTracker:
    """캠 속 발목 높이로 점프를 알아낸다.
    - 서 있을 때의 바닥 높이(더 낮은 쪽 발목)를 천천히 기억해 두고
    - 두 발목이 그보다 올라가면(= 더 낮은 쪽 발목까지 떠오르면) 점프로 본다
    - 올라간 정도는 몸통 길이(어깨~골반) 대비 비율로 돌려준다 → 캠과 거리가 달라도 같은 느낌
    쪼그려 앉기도 여기서 같이 잰다 (self.squat):
    - 서 있을 때의 골반~발목 길이를 기억해 두고, 그게 줄어든 비율 (0 = 서 있음 ~ 1)
    - 어깨너비로 나눠서 재기 때문에 캠에 다가가거나 멀어져도 앉은 것으로 착각하지 않음"""

    def __init__(self):
        self.ground = None                  # 캠 속 바닥 높이(px)
        self.air_since = None               # 발이 뜨기 시작한 시각
        self.torso = None                   # 땅에 있을 때의 캠 속 몸통 길이 (앞뒤 이동 감지용)
        self.sh_ground = None               # 땅에 있을 때의 어깨 높이
        self.sw = None                      # 어깨너비 (앞뒤 이동 감지용 - 쪼그려 앉거나 숙여도 거의 안 변함)
        self.legs = collections.deque()     # 최근 몇 초 동안의 (시각, 골반~발목 길이/어깨너비)
        self.squat = 0.0                    # 쪼그려 앉은 정도 (0 = 서 있음, 1 = 골반이 바닥까지)
        self.block_until = 0.0              # 이 시각까지는 점프로 보지 않음

    def _reset(self, low, torso, sh_y, sw, hip_y):
        self.ground, self.torso, self.sh_ground, self.sw = low, torso, sh_y, sw
        self.legs.clear()                   # 캠과의 거리가 바뀌었으니 서 있을 때 다리 길이도 새로 잼
        self.air_since = None
        self.block_until = time.monotonic() + JUMP_COOLDOWN   # 앞뒤로 움직이는 중에는 잠깐 점프 판단 안 함

    def update(self, pts, frame_h=None):
        self.squat = 0.0
        if pts is None:
            return 0.0
        VIS = game.VIS
        if min(pts[23, 2], pts[24, 2], pts[11, 2], pts[12, 2]) < VIS or min(pts[27, 2], pts[28, 2]) < ANKLE_VIS:
            return 0.0                      # 발목·골반·어깨가 확실히 보여야 판단
        low = max(pts[27, 1], pts[28, 1])   # 더 낮은 쪽 발목 (y가 클수록 아래)
        if frame_h is not None and low > frame_h * (1 - EDGE_MARGIN):
            return 0.0                      # 발목이 화면 아래 끝에 걸리거나 밖이면 위치가 추측값이라 판단 안 함
        torso = np.linalg.norm((pts[11, :2] + pts[12, :2]) / 2 - (pts[23, :2] + pts[24, :2]) / 2)
        torso = max(torso, 1.0)
        sh_y = (pts[11, 1] + pts[12, 1]) / 2
        hip_y = (pts[23, 1] + pts[24, 1]) / 2
        sw = max(np.linalg.norm(pts[11, :2] - pts[12, :2]), 1.0)
        if self.ground is None:
            self._reset(low, torso, sh_y, sw, hip_y)
        # 어깨너비가 달라졌다 = 앞뒤로 움직인 것(점프·앉기가 아님) → 기준 높이들을 바로 다시 잡음
        if abs(sw / self.sw - 1) > JUMP_SCALE_TOL:
            self._reset(low, torso, sh_y, sw, hip_y)
            return 0.0
        lift = (self.ground - low) / torso  # 바닥에서 떠오른 정도 (몸통 길이 비율)
        sh_lift = (self.sh_ground - sh_y) / torso
        now = time.monotonic()
        if lift < JUMP_THRESHOLD or now < self.block_until:   # 땅에 있을 때(또는 점프 판단을 쉬는 중) 기준 높이 갱신
            self.air_since = None
            rate = 0.3 if low > self.ground else 0.05   # 아래로는 빨리(앞으로 다가옴), 위로는 천천히
            self.ground += (low - self.ground) * rate
            self.torso += (torso - self.torso) * 0.1
            self.sw += (sw - self.sw) * 0.05
            self.sh_ground += (sh_y - self.sh_ground) * 0.3
            # 쪼그려 앉기: 골반~발목 세로 거리를 어깨너비로 나눈 값(캠과의 거리와 상관없음)이
            # '서 있을 때 값'보다 얼마나 줄었는지.
            # 서 있을 때 값 = 최근 SQUAT_WINDOW 초 동안 값 중 상위 10% 지점
            #   → 한두 번 튄 값(발목 오인식)에 끌려가지 않고, 캠 위치가 바뀌어도 몇 초 안에 다시 맞춰짐
            leg_now = max(low - hip_y, 0.0) / sw
            self.legs.append((now, leg_now))
            while self.legs and now - self.legs[0][0] > SQUAT_WINDOW:
                self.legs.popleft()
            if len(self.legs) >= 15 and not now < self.block_until:
                stand = float(np.percentile([v for _, v in self.legs], 90))
                self.squat = float(np.clip(1.0 - leg_now / max(stand, 1e-6), 0.0, 1.0))
                if self.squat < SQUAT_DEADZONE:         # 숨쉬기·흔들림 정도는 무시
                    self.squat = 0.0
            return 0.0
        if self.air_since is None:
            self.air_since = now
        air = now - self.air_since
        if air > JUMP_MAX_TIME:             # 너무 오래 떠 있음 = 점프가 아니라 뒤로 물러선 것 → 바닥 다시 잡기
            self._reset(low, torso, sh_y, sw, hip_y)
            return 0.0
        if air < JUMP_CONFIRM:              # 아주 잠깐 튄 건 무시 (인식 오류)
            return 0.0
        # 점프면 발만이 아니라 몸 전체(어깨)도 같이 올라감. 발만 올라가면(뒤로 물러섬 등) 점프가 아님
        if sh_lift < JUMP_BODY_RATIO * lift:
            return 0.0
        return min(lift, JUMP_MAX)


def remove_background(rgba, gap=9):
    """그림 바깥의 배경과 '종이색 테두리'를 투명하게. 얼굴·손처럼 선으로 둘러싸인 밝은 부분은 남긴다.

    - 종이색 = 밝고(가장 어두운 채널 > 110) 색이 거의 없는(회색빛) 픽셀.
      순백(255)만 지우면, 사진으로 찍은 종이의 회색 테두리가 캐릭터 둘레에 흰 띠로 남는다.
    - 그림 바깥에서부터 종이색을 따라 들어가되, 검은 선을 만나면 멈춘다.
      선이 살짝 끊긴 곳으로 얼굴 안까지 새지 않게, 선을 gap 픽셀만큼 두껍게 한 상태로 먼저 찾고
      마지막에 선 바로 앞까지만 한 픽셀씩 넓힌다(선을 건너뛰지 않음)."""
    rgba = rgba.copy()
    rgb = rgba[:, :, :3].astype(np.int16)
    transparent = rgba[:, :, 3] < 40
    paper = (rgb.min(axis=2) > 110) & (rgb.max(axis=2) - rgb.min(axis=2) < 35)

    if transparent.mean() > 0.02:
        # 이미 배경이 투명한 그림(캠으로 스캔한 그림): 외곽선은 스캐너가 이미 땄으므로
        # 투명한 곳 바로 옆의 얇은 종이색 띠만 지운다. (안쪽까지 따라 들어가면
        # 캠 화면의 흐린 선 틈으로 새서 흰 팔·손·귀까지 지워짐)
        bg = transparent.copy()
        kernel = np.ones((3, 3), np.uint8)
        for _ in range(SCAN_TRIM):
            bg = (cv2.dilate(bg.astype(np.uint8), kernel) > 0) & (paper | transparent)
        rgba[bg, 3] = 0
        return rgba

    paper |= transparent
    lines = (~paper).astype(np.uint8)

    sealed = paper & (cv2.dilate(lines, np.ones((gap, gap), np.uint8)) == 0)
    _, labels = cv2.connectedComponents(sealed.astype(np.uint8), connectivity=4)
    seeds = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1],
                            labels[transparent & sealed]])
    bg = np.isin(labels, np.unique(seeds[seeds > 0])) | transparent

    kernel = np.ones((3, 3), np.uint8)
    for _ in range(gap // 2 + 2):                          # 종이색 위로만 한 칸씩 넓힘
        bg = (cv2.dilate(bg.astype(np.uint8), kernel) > 0) & paper
    rgba[bg, 3] = 0
    return rgba


def load_character_rgba(path):
    """그림 불러오기 → 배경 제거 → 그림 부분만 잘라내기(여백 20px) → 키를 CHAR_HEIGHT에 맞춤"""
    data = np.fromfile(path, np.uint8)                   # 한글 경로 대비
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    rgba = remove_background(cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA))

    ys, xs = np.nonzero(rgba[:, :, 3] > 40)
    m = 20
    rgba = rgba[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]
    scale = CHAR_HEIGHT / rgba.shape[0]
    size = (max(1, round(rgba.shape[1] * scale)), CHAR_HEIGHT)
    return np.ascontiguousarray(cv2.resize(rgba, size, interpolation=cv2.INTER_AREA))


# ------------------------------------------------------------------ 틈을 건너지 않는 그물
class OutlineMesh(game.ArapMesh):
    """arap_dodge_game.ArapMesh 와 같지만 그물 만드는 방식만 다르다.

    원래 방식: 그림을 네모 칸으로 나누고 '조금이라도 그림이 걸친 칸'은 전부 그물에 넣는다
              → 팔과 몸통 사이처럼 좁은 틈(배경)도 칸이 덮어 버려서 팔과 배가 붙어 움직임.
    이 방식:   외곽선 위의 점 + 안쪽 점들로 삼각형을 만든 뒤,
              변이 배경(투명한 곳)을 지나가는 삼각형은 버린다 → 틈은 그물에서 빠짐."""

    def _build_grid(self, alpha, step):
        mask = (alpha > 40).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))   # 외곽선 픽셀까지만 살짝 포함
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if count > 2:                                        # 떨어진 얼룩은 버리고 가장 큰 덩어리만
            mask = (labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        h, w = mask.shape

        # 1) 점 뿌리기: 외곽선(바깥 + 안쪽 구멍 테두리) 위에 촘촘히 + 안쪽에 격자로
        pts = []
        contours, _ = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            pts.extend(c[::OUTLINE_STEP, 0, :].tolist())
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        gy, gx = np.mgrid[step // 2:h:step, step // 2:w:step]
        inner = dist[gy, gx] > step * 0.6                   # 외곽선에 너무 붙은 점은 빼서 찌그러진 삼각형 방지
        pts.extend(np.stack([gx[inner], gy[inner]], 1).tolist())
        pts = np.unique(np.array(pts, dtype=np.float64), axis=0)

        # 2) 들로네 삼각분할
        subdiv = cv2.Subdiv2D((-2, -2, w + 4, h + 4))
        for x, y in pts:
            subdiv.insert((float(x), float(y)))
        index = {(int(x), int(y)): i for i, (x, y) in enumerate(pts)}
        tris = []
        for t in subdiv.getTriangleList().reshape(-1, 3, 2):
            key = [(int(round(x)), int(round(y))) for x, y in t]
            if all(k in index for k in key):
                tris.append([index[k] for k in key])
        tris = np.array(tris, dtype=np.int32)

        # 3) 배경을 지나가는 삼각형 버리기 (세 변 위를 1px 간격으로 검사)
        good = []
        for tri in tris:
            p = pts[tri]
            samples = [p.mean(axis=0, keepdims=True)]
            for a, b in ((0, 1), (1, 2), (2, 0)):
                n = int(np.abs(p[b] - p[a]).max()) + 1
                t = np.linspace(0, 1, n + 1)[:, None]
                samples.append(p[a] + (p[b] - p[a]) * t)
            s = np.vstack(samples)
            xi = np.clip(np.round(s[:, 0]).astype(int), 0, w - 1)
            yi = np.clip(np.round(s[:, 1]).astype(int), 0, h - 1)
            inside = mask[yi, xi]
            good.append(inside[0] == 1 and inside.mean() >= 0.9)
        tris = tris[np.array(good, bool)]

        # 4) 서로 이어진 가장 큰 덩어리만 남기기 (조종점 없는 조각이 있으면 ARAP 계산이 깨짐)
        parent = list(range(len(pts)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for a, b, c in tris:
            for u, v in ((a, b), (b, c)):
                ru, rv = find(u), find(v)
                if ru != rv:
                    parent[ru] = rv
        roots = np.array([find(a) for a in tris[:, 0]])
        tris = tris[roots == np.bincount(roots).argmax()]

        used = np.unique(tris)                               # 안 쓰이는 점 제거 후 번호 다시 매기기
        remap = -np.ones(len(pts), np.int64)
        remap[used] = np.arange(len(used))
        self.rest = pts[used].copy()
        self.tris = remap[tris].astype(np.int32)

    # --- 그리는 순서: 팔을 맨 마지막에 그려서 몸통 위에 보이게 ---
    def arms_on_top(self):
        """삼각형마다 가장 가까운 뼈(머리·몸통·팔·다리)를 찾아서, 팔 삼각형을 맨 뒤로 보낸다.
        그림은 삼각형 순서대로 덮어 그리므로, 팔이 몸통 쪽으로 오면 팔이 몸 위에 보인다.
        (순서를 안 정하면 팔이 몸통 뒤로 숨어서 '팔이 뒤로 가는' 것처럼 보임)"""
        r = self.rest_pts
        bones = [("head", "neck", 0), ("neck", "hip", 0),
                 ("hip", "kn_l", 0), ("kn_l", "foot_l", 0), ("hip", "kn_r", 0), ("kn_r", "foot_r", 0),
                 ("neck", "sh_l", 0), ("neck", "sh_r", 0),
                 ("sh_l", "el_l", 1), ("el_l", "hand_l", 1), ("sh_r", "el_r", 1), ("el_r", "hand_r", 1)]
        cen = self.rest[self.tris].mean(axis=1)
        best = np.full(len(cen), np.inf)
        layer = np.zeros(len(cen), int)
        for a, b, lay in bones:
            pa, pb = r[a], r[b]
            v = pb - pa
            t = np.clip(((cen - pa) @ v) / max(v @ v, 1e-6), 0, 1)
            d = np.linalg.norm(cen - (pa + t[:, None] * v), axis=1)
            closer = d < best
            best[closer] = d[closer]
            layer[closer] = lay
        self.tris = self.tris[np.argsort(layer, kind="stable")]

    # --- 얼굴 고정: 목 위쪽(머리 부분) 그물 점들을 한 덩어리로 묶어서 휘지 않게 ---
    def _connected(self, sel, start_v):
        """sel(참/거짓 배열) 안에서 start_v 와 그물로 이어진 점 번호들"""
        adj = [[] for _ in range(self.n)]
        for a, b in self.edges:
            if sel[a] and sel[b]:
                adj[a].append(b)
                adj[b].append(a)
        seen = {int(start_v)}
        stack = [int(start_v)]
        while stack:
            for nb in adj[stack.pop()]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        return np.array(sorted(seen), dtype=np.int64)

    def fix_parts(self, head=True, hands_feet=True):
        """휘지 않고 통째로 움직일 부분(머리·손·발)을 정한다.
        각 부분은 (점 번호들, 기준점, 방향 시작, 방향 끝) 으로 저장되고, 매 프레임
        '방향 시작→끝' 조종점이 돌아간 만큼만 회전 + 기준점 위치로 이동한다.
          머리: 목 위쪽 전체, 목을 축으로 (목→머리 방향)
          손  : 아래팔 끝부분, 손을 축으로 (팔꿈치→손 방향)
          발  : 종아리 끝부분, 발을 축으로 (무릎→발 방향)
        각 부분은 해당 조종점과 그물로 이어진 점만 고르므로 다른 부분이 섞이지 않는다."""
        r = self.rest_pts
        vid = lambda name: self.handle_idx[HANDLES.index(name)]
        self.rigid_parts = []
        if head:
            above = self.rest[:, 1] < r["neck"][1] - HEAD_MARGIN
            self.rigid_parts.append((self._connected(above, vid("head")), "neck", "neck", "head"))
        if hands_feet:
            for tip, base, ratio in (("hand_l", "el_l", HAND_REGION), ("hand_r", "el_r", HAND_REGION),
                                     ("foot_l", "kn_l", FOOT_REGION), ("foot_r", "kn_r", FOOT_REGION)):
                a, b = r[base], r[tip]
                v = b - a
                L2 = max(v @ v, 1e-6)
                t = ((self.rest - a) @ v) / L2                 # 아래팔/종아리 방향으로 어디쯤인지 (0=팔꿈치, 1=손)
                near = np.linalg.norm(self.rest - b, axis=1) < 0.9 * math.sqrt(L2)
                sel = (t >= ratio) & near
                if self.rigid_parts:                           # 머리에 이미 들어간 점은 빼기
                    sel[np.concatenate([p[0] for p in self.rigid_parts])] = False
                sel[vid(tip)] = True
                idx = self._connected(sel, vid(tip))
                if len(idx) >= 3:
                    self.rigid_parts.append((idx, tip, base, tip))
        all_idx = np.concatenate([p[0] for p in self.rigid_parts]) if self.rigid_parts else np.zeros(0, np.int64)
        self.rigid_idx = all_idx
        A = self.L.copy()
        A[self.handle_idx, self.handle_idx] += game.HANDLE_WEIGHT
        A[all_idx, all_idx] += game.HANDLE_WEIGHT
        self.A_inv = np.linalg.inv(A)

    def fix_head(self):                                        # 예전 이름 호환
        self.fix_parts(head=True, hands_feet=False)

    def rigid_targets(self, targets):
        """각 고정 부분: 조종점 방향이 돌아간 만큼 회전 + 기준점으로 이동한 위치들(이어 붙여서)"""
        r = self.rest_pts
        T = {n: targets[i] for i, n in enumerate(HANDLES)}
        out = []
        for idx, anchor, a, b in self.rigid_parts:
            ang = math.atan2(*(T[b] - T[a])[::-1]) - math.atan2(*(r[b] - r[a])[::-1])
            c, s = math.cos(ang), math.sin(ang)
            d = self.rest[idx] - r[anchor]
            out.append(T[anchor] + np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], 1))
        return np.concatenate(out) if out else None

    def solve(self, targets, iters=game.ARAP_ITERS):
        """arap_dodge_game.ArapMesh.solve 와 같고, 고정 부분(머리·손·발)을 한 덩어리로 붙잡는 조건만 추가"""
        rigid = getattr(self, "rigid_idx", None)
        fixed = self.rigid_targets(targets) if rigid is not None and len(rigid) else None
        I, J, E, n = self.I, self.J, self.E, self.n
        P = self.cur
        for _ in range(iters):
            D = P[I] - P[J]
            dot = np.bincount(I, E[:, 0] * D[:, 0] + E[:, 1] * D[:, 1], n)
            cross = np.bincount(I, E[:, 0] * D[:, 1] - E[:, 1] * D[:, 0], n)
            theta = np.arctan2(cross, dot)
            co, si = np.cos(theta), np.sin(theta)
            cc, ss = 0.5 * (co[I] + co[J]), 0.5 * (si[I] + si[J])
            B = np.empty((n, 2))
            B[:, 0] = np.bincount(I, cc * E[:, 0] - ss * E[:, 1], n)
            B[:, 1] = np.bincount(I, ss * E[:, 0] + cc * E[:, 1], n)
            B[self.handle_idx] += game.HANDLE_WEIGHT * targets
            if fixed is not None:
                np.add.at(B, rigid, game.HANDLE_WEIGHT * fixed)
            P = self.A_inv @ B
        self.cur = P
        return P


# ------------------------------------------------------------------ 메인
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("검지 불빛 테스트 (Index light)")
    clock = pygame.time.Clock()
    font = game.load_font(18)

    path = find_character_path()
    if path is None:
        raise SystemExit("캐릭터 그림을 찾지 못했습니다 (saved_characters 또는 character.png)")
    print("캐릭터 그림 사용:", path)
    rgba = load_character_rgba(path)
    ratios = CUSTOM_POINTS.get(os.path.basename(path))
    if ratios is not None and not USE_LATEST_SCAN:          # 조종점을 직접 지정한 그림
        pts = complete_points(custom_points(ratios, rgba[:, :, 3]))
    else:                                                    # 그 밖의 그림(스캔 포함): 자동으로
        pts = auto_points(rgba[:, :, 3])
    game.find_character_points = lambda alpha: pts          # 그물이 이 위치로 조종점을 잡음
    mesh = OutlineMesh(rgba, step=MESH_STEP)                 # 틈을 건너지 않는 ARAP 그물
    if FIX_HEAD or FIX_HANDS_FEET:                           # 얼굴·손·발은 휘지 않고 통째로 회전만
        mesh.fix_parts(head=FIX_HEAD, hands_feet=FIX_HANDS_FEET)
    mesh.arms_on_top()                                       # 팔이 몸통 앞에 보이게
    rest = mesh.rest_pts
    hand_vertex = {s: mesh.handle_idx[HANDLES.index("hand_" + s)] for s in "lr"}

    tracker = BodyHandTracker()
    if tracker.ok:
        tracker.start()
        print("캠 인식 시작: 검지만 펴 보세요.")
    else:
        print("캠을 사용할 수 없습니다:", tracker.error)

    origin = np.array([(SCREEN_W - mesh.w) / 2, SCREEN_H - mesh.h - 40])   # 그림 왼쪽 위 위치
    center_x = origin[0]                       # 기준 위치에 서 있을 때 캐릭터 자리(화면 가운데)
    half_range = (SCREEN_W - mesh.w) / 2       # 가운데에서 화면 끝까지 갈 수 있는 거리
    ref_cx = MOVE_REF                          # 캠 속 기준 위치(0~1). C 키로 지금 서 있는 곳을 기준으로
    targets = pose_to_targets(None, rest)
    jump = JumpTracker()
    torso_char = np.linalg.norm(rest["neck"] - rest["hip"])   # 캐릭터 몸통 길이(px)
    leg_char = max(rest["foot_l"][1], rest["foot_r"][1]) - rest["hip"][1]   # 캐릭터 다리 길이(세로 px)
    lift_now = 0.0
    lights = {s: HandLight((SCREEN_W, SCREEN_H)) for s in "lr"}
    show_mesh, show_camera = False, True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_d:
                    show_mesh = not show_mesh
                elif event.key == pygame.K_v:
                    show_camera = not show_camera
                elif event.key == pygame.K_c and tracker.get()[0] is not None:
                    ref_cx = tracker.get()[0]["cx"]   # 지금 서 있는 곳을 기준(가운데)으로
                    print(f"기준 위치를 다시 잡았습니다: {ref_cx:.2f}")

        pose, light_on, point_dir, preview = tracker.get()
        tracking = pose is not None and time.monotonic() - pose["time"] < game.LOST_TIME

        # 좌우 이동: 캠 속 기준 위치에서 MOVE_RANGE 만큼 옆으로 가면 캐릭터는 화면 끝까지
        if FOLLOW_X and tracking:
            t = game.clamp((pose["cx"] - ref_cx) / MOVE_RANGE, -1.0, 1.0)
            goal_x = center_x + t * half_range
            origin[0] += (goal_x - origin[0]) * MOVE_SMOOTH

        # 캐릭터 동작 (ARAP)
        lift_now = jump.update(pose["pts"] if tracking else None, pose.get("fh") if tracking else None)   # 바닥 높이는 항상 추적 (발 든 것 판단용)
        if not JUMP:
            lift_now = 0.0
        sink = jump.squat * leg_char * SQUAT_GAIN if SQUAT else 0.0
        goal = pose_to_targets(pose["pts"] if tracking else None, rest, lift_now * torso_char * JUMP_GAIN,
                               jump.ground, min(sink, leg_char * SQUAT_MAX))
        targets += (goal - targets) * (TARGET_SMOOTH if SMOOTH_FILTER else game.POSE_SMOOTH)
        points = mesh.solve(targets)
        char_surface = mesh.render(points)

        # 불빛: 캐릭터 손 + 검지 방향으로 살짝
        for s in "lr":
            hand = points[hand_vertex[s]] + origin
            d = point_dir[s]
            target = hand + d * REACH * CHAR_HEIGHT if d is not None else hand
            lights[s].update(light_on[s], target)

        # 그리기
        screen.fill(BG_COLOR)
        screen.blit(char_surface, (int(origin[0]) - mesh.pad, int(origin[1]) - mesh.pad))
        for s in "lr":
            lights[s].draw(screen)

        if show_mesh:
            sp = points + origin
            for a, b in mesh.edges:
                pygame.draw.line(screen, (0, 220, 255), sp[a], sp[b], 1)
            for p in targets:
                pygame.draw.circle(screen, (255, 230, 0), (p + origin).astype(int), 5)

        if show_camera and preview is not None:
            cam = pygame.image.frombuffer(preview, PREVIEW_SIZE, "RGB")
            cam_x = SCREEN_W - PREVIEW_SIZE[0] - 10
            screen.blit(cam, (cam_x, 10))
            pygame.draw.rect(screen, (255, 255, 255), (cam_x, 10, *PREVIEW_SIZE), 1)
            if FOLLOW_X:                            # 노란 선: 기준 / 회색 선: 화면 끝까지 가는 위치 / 초록 선: 지금 나
                for v, colr, wd in ((ref_cx - MOVE_RANGE, (120, 120, 120), 1),
                                    (ref_cx + MOVE_RANGE, (120, 120, 120), 1),
                                    (ref_cx, (255, 220, 0), 2)):
                    x = cam_x + int(game.clamp(v, 0, 1) * PREVIEW_SIZE[0])
                    pygame.draw.line(screen, colr, (x, 10), (x, 10 + PREVIEW_SIZE[1]), wd)
                if tracking:
                    x = cam_x + int(pose["cx"] * PREVIEW_SIZE[0])
                    pygame.draw.line(screen, (0, 255, 0), (x, 10), (x, 10 + PREVIEW_SIZE[1]), 2)

        if not tracker.ok:
            status = "캠 없음: " + tracker.error
        elif not tracking:
            status = "몸이 안 보여요 - 캠 앞에 서 주세요"
        else:
            lit = [name for s, name in (("l", "왼쪽"), ("r", "오른쪽")) if light_on[s]]
            status = ("불빛: " + ", ".join(lit) + " 손") if lit else "검지만 펴 보세요"
            if lift_now > 0:
                status = "점프!   " + status
        screen.blit(font.render(status + "   |   C 기준 위치  D 그물  V 캠 화면  ESC 종료",
                                True, (170, 170, 170)), (10, 8))

        pygame.display.flip()
        clock.tick(60)

    tracker.close()
    pygame.quit()


if __name__ == "__main__":
    main()
