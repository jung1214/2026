"""
combined_game.py  -  손 제스처 + 몸 관절(자세) 동시 인식으로 조작하는 "공 피하기" 게임  (Python 3.11)

hand_game.py(손 모양)와 pose_game.py(몸 관절)를 하나로 합친 버전입니다.
매 프레임마다 손과 몸을 동시에 인식하고, 한 사람에 대해 "손 모양이 뚜렷하면 손,
아니면(손이 안 보이거나 애매하면) 몸 기울기"로 자동 전환해서 판정합니다.
즉 손을 써도 되고 몸만 써도 되고, 화면 상태판에 지금 어느 쪽 신호로
움직이고 있는지("[손]"/"[몸]") 항상 표시됩니다.

위에서 계속 떨어지는 공을 좌우로 움직여 피하는 게임입니다. 점프는 쓰지 않고
(공을 피하는 데 필요 없어서 뺐습니다) 좌/우 이동만으로 공을 피합니다.
공에 한 번이라도 맞으면 그 캐릭터는 바로 죽어서 화면에서 사라집니다.

설치:  pip install pygame pygame-particles sounddevice numpy opencv-python mediapipe
실행:  python combined_game.py

[조작]  (1인용: P1 한 명만 인식/조작합니다)
  손 - 주먹                         정지
  손 - 검지로 왼쪽/오른쪽 가리키기     이동
  몸 - 똑바로 서기                   정지   (손이 안 보이거나 애매할 때 사용됨)
  몸 - 상체를 왼쪽/오른쪽으로 기울이기  이동

[공 피하기]
  위에서 공이 계속 떨어짐 → 좌우로 이동해서 피하기
  공에 한 번 맞으면 즉사(캐릭터가 사라짐) → R 키로 재시작
  시간이 지날수록(점수가 오를수록) 공이 점점 빠르고 자주 떨어짐
  점수 100점에 도달하거나 목숨이 다 떨어지면 GAME OVER → R 키로 다시 시작
  최고 점수는 화면에 표시되고 best_score.txt에 저장됨(다시 실행해도 유지)

[마이크]
  박수(목소리 크기가 기준을 넘으면)  화면 플래시 효과만 (리셋되지는 않음)
  목소리 크기  캐릭터 밝기에만 반영 (파티클 양·캐릭터 크기는 더 이상 변하지 않음)
  전체 리셋은 키보드 R 키로만 합니다.

[키보드]
  ESC 종료 | T 효과 테스트(카메라·마이크 없이 전부 발사) | S 스크린샷(카메라 화면 제외)
  V 카메라 미리보기 | H 상태 패널 | ↑↓ 박수 감도 | R 리셋

[실행 확인]  화면 왼쪽 상태 패널 + 콘솔 출력 + 종료 시 run_log.txt 저장
[개인정보]   카메라 영상은 화면에만 표시하고 저장하지 않습니다. 마이크는 음량만 사용하고 녹음하지 않습니다.
"""
import base64
import io
import math
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict, deque

# ---------------------------------------------------------------- 라이브러리 로딩
_missing = []
try:
    import numpy as np
except ImportError:
    _missing.append("numpy")
try:
    import cv2
except ImportError:
    _missing.append("opencv-python")
try:
    import pygame
except ImportError:
    _missing.append("pygame")
try:
    import pygame_particles
    from pygame_particles import Line, Particle, ParticleContainer
except ImportError:
    _missing.append("pygame-particles")
try:
    import sounddevice as sd
except (ImportError, OSError):
    _missing.append("sounddevice")
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError:
    _missing.append("mediapipe")

if _missing:
    sys.exit("설치가 필요한 라이브러리: " + ", ".join(_missing) +
             "\n→ pip install " + " ".join(_missing))

# ---------------------------------------------------------------- 상수
WIDTH, HEIGHT = 1180, 720
FPS = 30
GROUND_Y = 630
R0 = 28
SPEED, GRAVITY, JUMP_V = 9, 1.4, -20   # SPEED = 좌우 이동 최고 속도(가감속 계산에 사용)
MOVE_ACCEL = 2.4      # 가속도: 클수록 최고 속도까지 빨리 도달함(반응이 빠르게 느껴짐)
MOVE_FRICTION = 0.8   # 감속(마찰) 계수: 1에 가까울수록 서서히 멈춤(부드러움), 0에 가까울수록 즉시 멈춤
COLORS = [(90, 160, 255)]   # 1인용: P1(파랑) 한 명만 사용
NAMES = ["P1"]
BG = (14, 14, 26)

# ---- 공 피하기 미니게임 ----
BALL_COLORS = [(255, 90, 90), (255, 170, 60), (255, 230, 90), (140, 220, 255), (170, 130, 255)]
BALL_R = 13              # 공 크기(고정 - 랜덤 없이 항상 일정한 크기)
BALL_START_LIVES = 2     # 목숨 개수(예전 1개보다 살짝 늘림 - 한 번 맞아도 한 번 더 기회가 있음)
WIN_SCORE = 100          # 이 점수에 도달하면 게임 종료(GAME OVER 화면 + 최종 기록)
BEST_SCORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_score.txt")   # 최고 점수 저장 파일
READY_DURATION = 2.5     # 게임(재)시작 시 공이 떨어지기 전 "준비" 안내를 보여주는 시간(초)
PREV_W, PREV_H = 280, 210

HAND_MODEL_PATH = "hand_landmarker.task"
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                   "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
POSE_MODEL_PATH = "pose_landmarker.task"
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                   "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# 사용하는 몸 관절 번호 (BlazePose 33점 중 손가락 관련 17~22번은 쓰지 않음)
NOSE = 0
L_SH, R_SH = 11, 12
L_EL, R_EL = 13, 14
L_WR, R_WR = 15, 16
L_HIP, R_HIP = 23, 24
L_KNEE, R_KNEE = 25, 26
L_ANK, R_ANK = 27, 28

POSE_CONNECTIONS = [
    (L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR),
    (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP),
    (L_HIP, L_KNEE), (L_KNEE, L_ANK), (R_HIP, R_KNEE), (R_KNEE, R_ANK),
]

VIS_THRESH = 0.35        # 몸 관절 신뢰도 하한
LEAN_THRESH = 0.35        # 상체 기울기 판정 기준(어깨너비 대비)
NOSE_LEAN_THRESH = 0.14   # 엉덩이가 안 보일 때 코-어깨중심 기울기 기준(더 민감하게)
ARM_UP_THRESH = 0.20      # 팔 들기 판정 기준(몸통 길이 대비)
JUMP_RISE_THRESH = 0.28   # 실제 점프 판정 기준(몸통 길이 대비)
JUMP_RESET_THRESH = 0.10

KOR = True

def L(ko, en):
    return ko if KOR else en

def mix(c1, c2, t):
    t = max(0.0, min(1.0, t))
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))


# ---------------------------------------------------------------- 손 제스처 판정 (hand_game.py와 동일)
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def finger_states(lm):
    wrist = lm[0]
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return [dist(lm[tip], wrist) > dist(lm[pip], wrist) for tip, pip in pairs]


def classify_hand(lm):
    index, middle, ring, pinky = finger_states(lm)
    up = sum([index, middle, ring, pinky])
    if up == 4:
        return "JUMP"
    if up == 0:
        return "STOP"
    if index and not (middle or ring or pinky):
        dx = lm[8].x - lm[5].x
        if dx < -0.06:
            return "LEFT"
        if dx > 0.06:
            return "RIGHT"
        return "POINT"
    return "UNKNOWN"


def assign_players_hand(hands):
    """hands: [(landmarks, score)] → {0: (landmarks, score)} (1인용: 가장 점수가 높은 손 하나만 사용)"""
    if not hands:
        return {}
    best = max(hands, key=lambda h: h[1])
    return {0: best}


# ---------------------------------------------------------------- 몸 자세 판정 (pose_game.py와 동일)
def midpoint(a, b):
    return ((a.x + b.x) / 2, (a.y + b.y) / 2)


def body_visible(lm, idxs, thresh=VIS_THRESH):
    try:
        return all(lm[i].visibility >= thresh for i in idxs)
    except (IndexError, AttributeError):
        return False


def joint_visibility(lm):
    try:
        return {
            "어깨": (lm[L_SH].visibility + lm[R_SH].visibility) / 2,
            "엉덩이": (lm[L_HIP].visibility + lm[R_HIP].visibility) / 2,
            "손목": (lm[L_WR].visibility + lm[R_WR].visibility) / 2,
        }
    except (IndexError, AttributeError):
        return {"어깨": 0.0, "엉덩이": 0.0, "손목": 0.0}


def classify_body(lm):
    """어깨만 보이면 코 위치로 좌우 기울임을 대신 추정하고(책상 앞에 앉아 엉덩이가
    화면 밖으로 나가는 가장 흔한 경우), 엉덩이까지 보이면 더 정확한 상체-엉덩이
    기울기로 판정한다. 손목이 보이면(엉덩이 유무와 무관하게) 팔 들기 JUMP도 판정."""
    if not body_visible(lm, [L_SH, R_SH]):
        return "UNKNOWN"
    shx, shy = midpoint(lm[L_SH], lm[R_SH])
    shoulder_w = dist(lm[L_SH], lm[R_SH])
    if shoulder_w < 1e-6:
        return "UNKNOWN"
    have_hips = body_visible(lm, [L_HIP, R_HIP])
    if have_hips:
        hpx, hpy = midpoint(lm[L_HIP], lm[R_HIP])
        torso_h = math.hypot(shx - hpx, shy - hpy)
    else:
        torso_h = shoulder_w * 1.3   # 엉덩이가 안 보일 때 몸통 길이 근사치(팔 들기 판정용)

    if body_visible(lm, [L_WR, R_WR]) and torso_h > 1e-6:
        if lm[L_WR].y < shy - ARM_UP_THRESH * torso_h and \
           lm[R_WR].y < shy - ARM_UP_THRESH * torso_h:
            return "JUMP"

    if have_hips:
        lean = (shx - hpx) / max(shoulder_w, 1e-6)
    elif body_visible(lm, [NOSE]):
        # 엉덩이가 안 보일 때: 코가 어깨 중심 기준으로 얼마나 치우쳤는지로 대신 판정
        # (상체 기울기보다 신호가 작아서 더 민감한 기준값을 쓴다)
        lean = (lm[NOSE].x - shx) / max(shoulder_w, 1e-6) * (LEAN_THRESH / NOSE_LEAN_THRESH)
    else:
        return "UNKNOWN"

    if lean < -LEAN_THRESH:
        return "LEFT"
    if lean > LEAN_THRESH:
        return "RIGHT"
    return "STOP"


def update_hip_jump(p, hip_y, torso_h,
                     rise_thresh=JUMP_RISE_THRESH, reset_thresh=JUMP_RESET_THRESH, adapt_rate=0.06):
    """엉덩이 높이 변화로 '실제 점프'를 감지한다 (손/몸 제스처와 완전히 별개인 보조 입력)."""
    if hip_y is None or torso_h <= 1e-6:
        return False
    if p.hip_baseline is None:
        p.hip_baseline = hip_y
        return False
    rise = (p.hip_baseline - hip_y) / torso_h
    triggered = False
    if rise > rise_thresh and not p.body_jump_lock:
        p.body_jump_lock = True
        triggered = True
    if rise < reset_thresh:
        p.body_jump_lock = False
        p.hip_baseline += (hip_y - p.hip_baseline) * adapt_rate
    return triggered


def assign_players_pose(poses):
    """poses: [관절목록(33점), ...] → {0: 관절목록} (1인용: 어깨가 가장 잘 보이는 사람 하나만 사용)"""
    if not poses:
        return {}
    def vis(lm):
        try:
            return lm[L_SH].visibility + lm[R_SH].visibility
        except (IndexError, AttributeError):
            return 0.0
    best = max(poses, key=vis)
    return {0: best}


def classify_combined(hand_lm, pose_lm):
    """손·몸을 함께 본다: 손 모양이 뚜렷하면 손을 우선 사용하고, 손이 안 보이거나
    애매하면(UNKNOWN) 몸 기울기/팔 자세로 자동 전환한다. 반환값 (제스처, 소스)."""
    if hand_lm is not None:
        g = classify_hand(hand_lm)
        if g != "UNKNOWN":
            return g, "hand"
    if pose_lm is not None:
        g = classify_body(pose_lm)
        if g != "UNKNOWN":
            return g, "body"
    if hand_lm is not None or pose_lm is not None:
        return "UNKNOWN", None
    return "NO INPUT", None


# ---------------------------------------------------------------- 입력: 손+몸(mediapipe) / 마이크(sounddevice)
def _open_camera():
    """OS/드라이버에 따라 기본 백엔드로 안 열리는 경우가 있어 여러 방식을 순서대로 시도한다."""
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for idx in (0, 1):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                cap.release()
    return None


class CombinedSource:
    """같은 카메라 프레임 한 장에 손 인식(HandLandmarker)과 몸 인식(PoseLandmarker)을
    동시에 돌린다. 둘 중 하나만 로딩에 실패해도 나머지 하나로는 계속 동작한다."""
    def __init__(self):
        self.ok = False
        self.error = ""
        self.cap = None
        self.hand_landmarker = None
        self.pose_landmarker = None
        self.hand_ok, self.hand_error = False, ""
        self.pose_ok, self.pose_error = False, ""
        self.frames = 0

        try:
            if not os.path.exists(HAND_MODEL_PATH):
                print("손 인식 모델 다운로드 중...")
                urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
            hand_options = vision.HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.6,
                min_hand_presence_confidence=0.6,
                min_tracking_confidence=0.6,
            )
            self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
            self.hand_ok = True
        except Exception as e:
            self.hand_error = str(e)

        try:
            if not os.path.exists(POSE_MODEL_PATH):
                print("자세 인식 모델 다운로드 중...")
                urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
            pose_options = vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=2,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            self.pose_ok = True
        except Exception as e:
            self.pose_error = str(e)

        try:
            self.cap = _open_camera()
            if self.cap is None or not self.cap.isOpened():
                raise RuntimeError("웹캠을 열 수 없습니다 (다른 프로그램이 사용 중이거나 권한이 없을 수 있음)")
            self.ok = self.hand_ok or self.pose_ok
            if not self.ok:
                raise RuntimeError("손·몸 인식 모델을 모두 불러오지 못했습니다")
        except Exception as e:
            self.error = str(e)
            self.ok = False

    def read(self):
        """(좌우반전된 BGR 프레임, [(손 관절, 신뢰도), ...], [몸 관절목록, ...])"""
        if not self.ok:
            return None, [], []
        ok, frame = self.cap.read()
        if not ok:
            return None, [], []
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = int(time.monotonic() * 1000)
        self.frames += 1

        hands = []
        if self.hand_ok:
            result = self.hand_landmarker.detect_for_video(image, ts)
            for i, lm in enumerate(result.hand_landmarks or []):
                score = result.handedness[i][0].score if result.handedness else 0.0
                hands.append((lm, float(score)))

        poses = []
        if self.pose_ok:
            result = self.pose_landmarker.detect_for_video(image, ts)
            poses = list(result.pose_landmarks or [])

        return frame, hands, poses

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.hand_landmarker is not None:
            self.hand_landmarker.close()
        if self.pose_landmarker is not None:
            self.pose_landmarker.close()


class Mic:
    def __init__(self):
        self.ok = False
        self.error = ""
        self.name = "-"
        self.rms = 0.0
        self._peak = 0.0
        self._stream = None
        self.callbacks = 0     # 콜백 호출 횟수(진단용) - 0이면 OS가 마이크 접근을 막고 있다는 뜻
        self.cb_error = ""
        try:
            device = self._pick_input_device()
            self._stream = sd.InputStream(channels=1, blocksize=1024, callback=self._cb, device=device)
            self._stream.start()
            self.ok = True
            try:
                dev_index = device if device is not None else sd.default.device[0]
                self.name = sd.query_devices(dev_index)["name"]
            except Exception:
                self.name = "default"
        except Exception as e:
            self.error = str(e)

    @staticmethod
    def _pick_input_device():
        """기본 입력 장치에 녹음 채널이 없으면(예: 출력 전용 장치가 기본으로 잡힌 경우)
        입력 채널이 있는 첫 번째 장치를 대신 찾는다. 실패하면 None(시스템 기본값 사용)."""
        try:
            default_idx = sd.default.device[0]
            info = sd.query_devices(default_idx)
            if info.get("max_input_channels", 0) > 0:
                return None
        except Exception:
            pass
        try:
            for i, info in enumerate(sd.query_devices()):
                if info.get("max_input_channels", 0) > 0:
                    return i
        except Exception:
            pass
        return None

    def _cb(self, indata, frames, time_info, status):
        try:
            self.callbacks += 1
            self.rms = float(np.sqrt(np.mean(indata ** 2)))
            self._peak = max(self._peak, float(np.max(np.abs(indata))))
        except Exception as e:
            self.cb_error = str(e)

    def read(self):
        peak, self._peak = self._peak, 0.0
        return self.rms, peak

    def close(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


# ---------------------------------------------------------------- 게임 로직 (그리기와 분리)
class Player:
    def __init__(self, slot, start_x):
        self.slot = slot
        self.start_x = start_x
        self.reset()

    def reset(self):
        self.x, self.y, self.vy = self.start_x, float(GROUND_Y), 0.0
        self.vx = 0.0   # 좌우 이동 속도(가감속을 위해 유지되는 상태, 부드러운 움직임용)
        self.visible = False
        self.alpha = 0
        self.gesture, self.score, self.source = "NO INPUT", 0.0, None
        self.ref_pos = None
        self.stable_gesture, self.stable_count = "NO INPUT", 0
        self.unknown_frames = 0
        self.in_air = False
        self.squash = 0
        self.move_frames = 0
        self.jump_request = None
        self.trail = deque(maxlen=12)
        self.facing = 1
        self.anim_phase = 0.0
        self.hip_baseline = None       # 실제 점프 감지용 기준 높이 (몸 관절 기반, 손과 무관)
        self.body_jump_lock = False
        self.diag = {"어깨": 0.0, "엉덩이": 0.0, "손목": 0.0}  # 몸 관절 인식률(진단/화면표시용)
        self.lives = BALL_START_LIVES   # 공 피하기 미니게임: 남은 목숨
        self.hit_invuln = 0             # 맞은 직후 잠깐 무적(깜빡임) 프레임 수
        self.dead = False               # 공에 맞아 죽은 상태(리셋 전까지 사라진 채로 유지)


class Ball:
    """공 피하기 미니게임에서 위에서 떨어지는 공 하나."""
    __slots__ = ("x", "y", "r", "vx", "vy", "color")

    def __init__(self, x, y, r, vx, vy, color):
        self.x, self.y, self.r = x, y, r
        self.vx, self.vy = vx, vy
        self.color = color


def load_best_score():
    try:
        with open(BEST_SCORE_PATH, "r", encoding="utf-8") as f:
            return max(0, int(f.read().strip()))
    except Exception:
        return 0


def save_best_score(v):
    try:
        with open(BEST_SCORE_PATH, "w", encoding="utf-8") as f:
            f.write(str(int(v)))
    except Exception:
        pass


class GameLogic:
    def __init__(self):
        self.players = [Player(0, WIDTH * 0.5)]   # 1인용: P1 한 명만 사용
        self.events = []
        self.counters = defaultdict(int)
        self.log = deque(maxlen=7)
        self.full_log = []
        self.last_clap = -1.0
        self.mic_block_until = 0.0
        self.meet_until = 0.0
        self.clap_threshold = 0.25
        self.level = 0.0
        self.peak_disp = 0.0
        self.scroll = 0.0
        self.shake = 0
        self.flash = 0.0
        # 공 피하기 미니게임 상태
        self.balls = []
        self.ball_timer = 1.3
        self.dodge_score = 0.0
        self.game_over = False
        self.cleared = False          # True면 WIN_SCORE 달성으로 끝난 것, False면 목숨이 다 떨어져 끝난 것
        self.final_score = 0
        self.new_record = False
        self.best_score = load_best_score()
        self.round_phase = "ready"     # ready(준비 안내 중) -> playing(진행 중) -> (game_over는 별개 플래그)
        self.ready_timer = READY_DURATION

    def emit(self, name, **data):
        self.events.append({"name": name, **data})
        self.counters[name] += 1
        parts = []
        if "slot" in data:
            parts.append(NAMES[data["slot"]])
        for k in ("gesture", "source"):
            if k in data:
                parts.append(str(data[k]))
        if "peak" in data:
            parts.append(f"peak={data['peak']}")
        line = f"{time.strftime('%H:%M:%S')} {name} {' '.join(parts)}".strip()
        self.log.append(line)
        self.full_log.append(line)

    def drain(self):
        ev, self.events = self.events, []
        return ev

    def reset_all(self):
        for p in self.players:
            p.reset()
        self.balls = []
        self.ball_timer = 1.3
        self.dodge_score = 0.0
        self.game_over = False
        self.cleared = False
        self.final_score = 0
        self.new_record = False
        self.round_phase = "ready"
        self.ready_timer = READY_DURATION
        self.flash = 0.8
        self.emit("RESET")

    def _on_clap(self, now, peak):
        # 게임이니까 목소리/박수 크기가 기준을 넘는다고 리셋되지는 않고,
        # 화면에 짧은 플래시 효과만 준다 (리셋은 R 키로만 한다).
        self.last_clap = now
        self.emit("CLAP", peak=round(peak, 2))
        self.flash = max(self.flash, 0.35)

    def self_test(self):
        """카메라·마이크 없이 모든 효과를 한 번씩 발생시킨다."""
        self.round_phase = "playing"  # 테스트 중엔 "준비" 카운트다운을 기다리지 않고 바로 확인
        for p in self.players:
            if not p.visible:
                p.visible, p.alpha = True, 0
                self.emit("APPEAR", slot=p.slot, pos=(p.x, p.y))
        a = self.players[0]
        self.emit("MOVE", slot=0, pos=(a.x, GROUND_Y + R0))
        self.emit("GESTURE_OK", slot=0, gesture="LEFT", source="hand", pos=(WIDTH * 0.3, HEIGHT * 0.3))
        self.emit("GESTURE_FAIL", slot=0, gesture="UNKNOWN", pos=(WIDTH * 0.7, HEIGHT * 0.3))
        self._on_clap(time.monotonic() - 10, 0.5)

    def update(self, hands, poses, rms, peak, now):
        if self.shake > 0:
            self.shake -= 1
        self.flash = max(0.0, self.flash - 0.08)

        self.level += (min(1.0, rms / 0.15) - self.level) * 0.3
        self.peak_disp = max(peak, self.peak_disp * 0.85)
        if now >= self.mic_block_until and peak > self.clap_threshold and now - self.last_clap > 0.15:
            self._on_clap(now, peak)

        assigned_hands = assign_players_hand(hands)
        assigned_poses = assign_players_pose(poses)

        for slot, p in enumerate(self.players):
            if p.dead:
                continue  # 죽은 플레이어는 리셋 전까지 입력/움직임을 완전히 멈춘다
            hand_entry = assigned_hands.get(slot)
            hand_lm = hand_entry[0] if hand_entry else None
            hand_score = hand_entry[1] if hand_entry else 0.0
            pose_lm = assigned_poses.get(slot)

            has_input = hand_lm is not None or pose_lm is not None
            hpy = torso_h = None

            if has_input:
                g, source = classify_combined(hand_lm, pose_lm)
                p.gesture, p.source = g, source
                if source == "hand":
                    p.score = hand_score
                    p.ref_pos = (hand_lm[0].x * WIDTH, hand_lm[0].y * HEIGHT)
                elif source == "body":
                    p.score = sum(pose_lm[i].visibility for i in (L_SH, R_SH, L_HIP, R_HIP)) / 4
                    shx, shy = midpoint(pose_lm[L_SH], pose_lm[R_SH])
                    p.ref_pos = (shx * WIDTH, shy * HEIGHT)
                else:
                    p.score = 0.0
                    if pose_lm is not None:
                        shx, shy = midpoint(pose_lm[L_SH], pose_lm[R_SH])
                        p.ref_pos = (shx * WIDTH, shy * HEIGHT)
                    elif hand_lm is not None:
                        p.ref_pos = (hand_lm[0].x * WIDTH, hand_lm[0].y * HEIGHT)

                if pose_lm is not None:
                    p.diag = joint_visibility(pose_lm)
                else:
                    p.diag = {"어깨": 0.0, "엉덩이": 0.0, "손목": 0.0}

                if not p.visible:
                    p.visible, p.alpha = True, 0
                    self.emit("APPEAR", slot=slot, pos=(p.x, p.y))
                if g == p.stable_gesture:
                    p.stable_count += 1
                else:
                    p.stable_gesture, p.stable_count = g, 1
                if p.stable_count == 4 and g in ("STOP", "LEFT", "RIGHT"):
                    self.emit("GESTURE_OK", slot=slot, gesture=g, source=source, pos=p.ref_pos)
                if g == "UNKNOWN":
                    p.unknown_frames += 1
                    if p.unknown_frames >= 12 and (p.unknown_frames - 12) % 30 == 0:
                        self.emit("GESTURE_FAIL", slot=slot, gesture=g, pos=p.ref_pos)
                else:
                    p.unknown_frames = 0
            else:
                p.gesture, p.source, p.ref_pos = "NO INPUT", None, None
                p.stable_gesture, p.stable_count, p.unknown_frames = "NO INPUT", 0, 0
                p.diag = {"어깨": 0.0, "엉덩이": 0.0, "손목": 0.0}

            if not p.visible:
                continue

            # 공 피하기 게임에서는 점프를 쓰지 않는다: 좌/우 이동만으로 공을 피한다
            # (JUMP 제스처가 나와도 이동은 그대로 무시되어 STOP과 동일하게 처리됨)
            # 이동은 즉시 SPEED로 튀지 않고, 서서히 가속/감속해서 부드럽게 움직인다.
            g = p.gesture
            direction = -1 if g == "LEFT" else 1 if g == "RIGHT" else 0
            if direction:
                p.vx = max(-SPEED, min(SPEED, p.vx + direction * MOVE_ACCEL))
                p.facing = direction
            else:
                p.vx *= MOVE_FRICTION
                if abs(p.vx) < 0.05:
                    p.vx = 0.0
            p.x += p.vx
            moving = abs(p.vx) > 0.4
            if moving:
                p.anim_phase += 0.9
                self.scroll += p.vx * 0.5
                p.move_frames += 1
                if p.move_frames % 6 == 0:
                    self.emit("MOVE", slot=slot, pos=(p.x, GROUND_Y + R0))
            else:
                p.move_frames = 0
            p.x = max(R0, min(WIDTH - R0, p.x))
            if moving:
                p.trail.append((p.x, p.y))
            elif p.trail:
                p.trail.popleft()
            p.alpha = min(255, p.alpha + 15)
            if p.squash > 0:
                p.squash -= 1

        self._update_dodge_balls()

    def _finish(self, cleared):
        """게임 종료 처리: 최종 점수 확정, 최고 점수 갱신/저장, GAME OVER 상태로 고정."""
        if self.game_over:
            return
        self.game_over = True
        self.cleared = cleared
        if cleared:
            self.dodge_score = float(WIN_SCORE)
        self.final_score = int(self.dodge_score)
        self.new_record = self.final_score > self.best_score
        if self.new_record:
            self.best_score = self.final_score
            save_best_score(self.best_score)
        self.balls = []
        self.emit("GAME_OVER", cleared=cleared, score=self.final_score)

    def _update_dodge_balls(self):
        """위에서 떨어지는 공을 캐릭터가 좌우로 피하는 미니게임 로직.
        목숨이 남은 플레이어가 화면에 보일 때만 공을 떨어뜨리고 점수를 올린다."""
        if self.round_phase == "ready":
            # 사람이 카메라에 보이기 시작해야 카운트다운이 흐른다(아무도 없는데 미리 다 흘러가지 않게).
            if any(p.visible for p in self.players):
                self.ready_timer -= 1.0 / FPS
                if self.ready_timer <= 0:
                    self.round_phase = "playing"
                    self.emit("ROUND_START")
            return  # 준비 안내 중에는 공이 생기지도, 점수가 오르지도 않는다

        if self.game_over:
            return   # 끝난 뒤에는 R 키로 리셋하기 전까지 그대로 멈춘다
        if all(p.lives <= 0 for p in self.players):
            self._finish(False)
            return
        if not any(p.visible for p in self.players):
            return

        alive = [p for p in self.players if p.lives > 0]
        self.dodge_score += 0.02 * len(alive)

        speed_mul = 1.0 + min(2.2, self.dodge_score / 45.0)
        self.ball_timer -= 1.0 / FPS
        if self.ball_timer <= 0:
            self.ball_timer = max(0.28, random.uniform(0.65, 1.15) / speed_mul)
            r = BALL_R
            x = random.randint(r + 10, WIDTH - r - 10)
            vy = random.uniform(3.2, 4.8) * speed_mul
            vx = random.uniform(-1.2, 1.2) * speed_mul * 0.5
            color = random.choice(BALL_COLORS)
            self.balls.append(Ball(x, -r - 10, r, vx, vy, color))

        for p in self.players:
            if p.hit_invuln > 0:
                p.hit_invuln -= 1

        remaining = []
        for ball in self.balls:
            ball.x += ball.vx
            ball.y += ball.vy
            if ball.x - ball.r < 0:
                ball.x, ball.vx = ball.r, -ball.vx
            elif ball.x + ball.r > WIDTH:
                ball.x, ball.vx = WIDTH - ball.r, -ball.vx

            hit = False
            for p in self.players:
                if not p.visible or p.lives <= 0 or p.hit_invuln > 0:
                    continue
                unit = R0   # 목소리 크기로 캐릭터가 커지지 않으므로 판정 크기도 항상 고정
                cy = p.y - unit * 2.25        # 캐릭터 그림 중심 높이 근사치 (축소된 그림 크기에 맞춤)
                radius = unit * 1.8           # 캐릭터 그림 크기에 맞춘 판정 반경
                if math.hypot(p.x - ball.x, cy - ball.y) < (ball.r + radius):
                    p.lives -= 1
                    p.hit_invuln = 75
                    self.emit("BALL_HIT", slot=p.slot, pos=(p.x, p.y), lives=p.lives)
                    self.shake = max(self.shake, 10)
                    self.flash = max(self.flash, 0.3)
                    hit = True
                    if p.lives <= 0:
                        # 죽음 처리: 캐릭터를 즉시 사라지게 해서 눈으로 바로 알 수 있게 한다
                        death_pos = (p.x, p.y)
                        p.dead = True
                        p.visible = False
                        p.alpha = 0
                        self.shake = max(self.shake, 16)
                        self.flash = max(self.flash, 0.6)
                        self.emit("PLAYER_OUT", slot=p.slot, pos=death_pos)

            if hit:
                continue
            if ball.y - ball.r > HEIGHT + 30:
                self.dodge_score += 1
                self.emit("BALL_DODGE")
                continue
            remaining.append(ball)
        self.balls = remaining

        if self.dodge_score >= WIN_SCORE:
            self._finish(True)
        elif all(p.lives <= 0 for p in self.players):
            self._finish(False)


# ---------------------------------------------------------------- 효과: 파티클(pygame-particles) + 효과음(pygame.mixer)
def make_color(c0, c1):
    def fn(particle, shape):
        if particle.state == 0:
            return c0
        t = max(0.0, min(1.0, particle.percent_completed / 100))
        return tuple(int(a + (b - a) * t) for a, b in zip(c0, c1))
    return fn


PRESETS = {
    "sparkle": dict(color=make_color((255, 255, 255), (255, 200, 60)), speed=(1, 3), size=(6, 16), life=0.7, fade=0.8),
    "dust":    dict(color=make_color((190, 190, 200), (40, 40, 50)),   speed=(1, 3), size=(4, 10), life=0.5, fade=0.6),
    "bubble":  dict(color=make_color((120, 230, 255), (20, 40, 60)),   speed=(1, 2), size=(3, 8),  life=0.5, fade=0.7),
    "ok":      dict(color=make_color((255, 240, 120), (80, 60, 10)),   speed=(1, 3), size=(4, 10), life=0.3, fade=0.4),
    "fail":    dict(color=make_color((255, 70, 70), (60, 10, 10)),     speed=(1, 3), size=(4, 10), life=0.3, fade=0.4),
    "meet":    dict(color=make_color((255, 140, 220), (60, 20, 60)),   speed=(1, 3), size=(5, 14), life=0.8, fade=0.9),
    "clap":    dict(color=make_color((255, 255, 255), (60, 120, 200)), speed=(2, 4), size=(6, 18), life=0.6, fade=0.8),
}


class Effects:
    def __init__(self):
        self.particles = ParticleContainer()
        self.ok = True
        self.error = ""
        self.fired = 0
        self.sound_ok = False
        self.sounds = {}
        self._init_sound()

    def _init_sound(self):
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
            self.sounds = {
                "appear": self._tone(660, 990, 0.25),
                "jump": self._tone(400, 800, 0.15),
                "land": self._tone(140, 60, 0.12, 0.5),
                "meet": self._tone(880, 1320, 0.2),
                "reset": self._tone(800, 300, 0.3),
                "hit": self._tone(500, 80, 0.35, 0.5),
            }
            self.sound_ok = True
        except Exception as e:
            self.sound_error = str(e)

    def _tone(self, f0, f1, dur, vol=0.3):
        rate, _, channels = pygame.mixer.get_init()
        n = int(rate * dur)
        phase = 2 * np.pi * np.cumsum(np.linspace(f0, f1, n)) / rate
        wave = (np.sin(phase) * np.linspace(1, 0, n) ** 1.5 * vol * 32767).astype(np.int16)
        if channels == 2:
            wave = np.column_stack([wave, wave])
        return pygame.sndarray.make_sound(np.ascontiguousarray(wave))

    def play(self, name, logic):
        if self.sound_ok and name in self.sounds:
            self.sounds[name].play()
            logic.mic_block_until = time.monotonic() + 0.3

    def spawn(self, kind, pos, n):
        if not self.ok or pos is None:
            return
        preset = PRESETS[kind]
        try:
            for _ in range(n):
                self.particles.add(Particle(
                    center_x=pos[0] + random.uniform(-10, 10),
                    center_y=pos[1] + random.uniform(-6, 6),
                    life_seconds=preset["life"], fade_seconds=preset["fade"],
                    speed=preset["speed"], shape_cls=Line, size=preset["size"],
                    color=preset["color"], width=3))
            self.fired += n
        except Exception as e:
            self.ok, self.error = False, str(e)

    def handle(self, ev, logic):
        # 목소리 크기는 더 이상 파티클 양에 영향을 주지 않음(캐릭터 밝기에만 반영됨).
        name, pos = ev["name"], ev.get("pos")
        if name == "APPEAR":
            self.spawn("sparkle", pos, 10)
            self.play("appear", logic)
        elif name == "JUMP":
            self.spawn("dust", pos, 6)
            if ev.get("source") == "gesture":
                self.play("jump", logic)
        elif name == "LAND":
            self.spawn("dust", pos, 8)
            self.play("land", logic)
        elif name == "MOVE":
            self.spawn("bubble", pos, 2)
        elif name == "GESTURE_OK":
            if ev.get("gesture") != "STOP":
                self.spawn("ok", pos, 6)
        elif name == "GESTURE_FAIL":
            self.spawn("fail", pos, 6)
        elif name == "MEET":
            self.spawn("meet", pos, 14)
            self.play("meet", logic)
        elif name == "CLAP":
            self.spawn("clap", (WIDTH // 2, 260), 8 + int(ev.get("peak", 0.3) * 30))
        elif name == "RESET":
            self.play("reset", logic)
        elif name == "BALL_HIT":
            self.spawn("fail", pos, 10)
        elif name == "PLAYER_OUT":
            self.spawn("fail", pos, 26)
            self.play("hit", logic)
        elif name == "GAME_OVER":
            if ev.get("cleared"):
                self.spawn("sparkle", (WIDTH // 2, HEIGHT // 2), 40)
                self.play("appear", logic)
        elif name == "ROUND_START":
            self.spawn("sparkle", (WIDTH // 2, HEIGHT // 2), 24)
            self.play("appear", logic)

    def draw(self, surface):
        if not self.ok:
            return
        try:
            self.particles.draw(surface)
        except Exception as e:
            self.ok, self.error = False, str(e)


# ---------------------------------------------------------------- 사람 캐릭터 관절 좌표 계산 (pygame 없이도 검증 가능)
def human_pose(p, level):
    unit = R0   # 목소리 크기로 캐릭터가 커지지 않으므로 크기는 항상 고정 (level은 밝기에만 사용)
    head_r = unit * 0.42
    torso_len = unit * 1.45
    leg_len = unit * 1.25
    arm_len = unit * 1.0
    hip_w = unit * 0.22
    shoulder_w = unit * 0.42
    limb_w = max(4, int(unit * 0.26))
    lean = 0.0

    moving = p.move_frames > 0
    jumping = p.in_air
    landing = p.squash > 0

    if jumping:
        aL, aR = -0.85, -0.55
        armL, armR = -2.7, -2.5
        lean = p.facing * unit * 0.05
    elif landing:
        k = p.squash / 6
        torso_len *= (1 - 0.22 * k)
        leg_len *= (1 - 0.3 * k)
        aL, aR = -0.9 * k, 0.9 * k
        armL, armR = -0.9 - 0.4 * k, 0.9 + 0.4 * k
    elif moving:
        swing = math.sin(p.anim_phase)
        aL, aR = swing * 0.7, -swing * 0.7
        armL, armR = -swing * 0.55, swing * 0.55
        lean = p.facing * unit * 0.08
    else:
        aL, aR = -0.08, 0.08
        armL, armR = -0.18, 0.18

    hip = (lean, -leg_len)
    hip_l = (hip[0] - hip_w, hip[1])
    hip_r = (hip[0] + hip_w, hip[1])
    foot_l = (hip_l[0] + leg_len * math.sin(aL), hip_l[1] + leg_len * math.cos(aL))
    foot_r = (hip_r[0] + leg_len * math.sin(aR), hip_r[1] + leg_len * math.cos(aR))

    shoulder_y = hip[1] - torso_len
    shoulder = (lean, shoulder_y)
    sh_l = (shoulder[0] - shoulder_w, shoulder_y)
    sh_r = (shoulder[0] + shoulder_w, shoulder_y)
    hand_l = (sh_l[0] + arm_len * math.sin(armL), sh_l[1] + arm_len * math.cos(armL))
    hand_r = (sh_r[0] + arm_len * math.sin(armR), sh_r[1] + arm_len * math.cos(armR))

    head_c = (shoulder[0], shoulder_y - head_r - unit * 0.16)

    return {
        "limb_w": limb_w, "head_r": head_r,
        "hip": hip, "hip_l": hip_l, "hip_r": hip_r, "foot_l": foot_l, "foot_r": foot_r,
        "shoulder": shoulder, "sh_l": sh_l, "sh_r": sh_r, "hand_l": hand_l, "hand_r": hand_r,
        "head_c": head_c,
    }


CHARACTER_DIR = "character"

# 기본 제공 캐릭터 그림(base64 PNG) - 폴더 없이도 항상 바로 나오도록 코드 안에 내장.
# character/full.png 파일을 직접 넣으면 그 그림이 이걸 대신합니다(있으면 우선 사용).
EMBEDDED_CHARACTER_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAARUAAAFlCAYAAADFzHwjAAAWfmNhQlgAABZ+anVtYgAAAB5qdW1kYzJwYQARABCAAACqADibcQNjMnBhAAAAFlhqdW1i"
    "AAAAR2p1bWRjMm1hABEAEIAAAKoAOJtxA3VybjpjMnBhOjg3OTQ0MGI5LTA3ZDAtNDhmNC1iNGM1LTVjMTUwNjNhM2RiYwAAAAOTanVtYgAAAClqdW1k"
    "YzJhcwARABCAAACqADibcQNjMnBhLmFzc2VydGlvbnMAAAAAuGp1bWIAAABEanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5pbmdyZWRpZW50LnYzAAAA"
    "ABhjMnNo9OStdJf2UcAkBnX8eUoiVAAAAGxjYm9yo2lkYzpmb3JtYXRpaW1hZ2UvcG5namluc3RhbmNlSUR4LHhtcDppaWQ6NDZmYWVlMWMtNDUzYS00"
    "OTI3LThmNGQtZjliNjM4Yzg2NGMwbHJlbGF0aW9uc2hpcGhwYXJlbnRPZgAAAeJqdW1iAAAAQWp1bWRjYm9yABEAEIAAAKoAOJtxE2MycGEuYWN0aW9u"
    "cy52MgAAAAAYYzJzaGOtO16BA2gkxwTprMml/9sAAAGZY2JvcqJnYWN0aW9uc4KiZmFjdGlvbmtjMnBhLm9wZW5lZGpwYXJhbWV0ZXJzoWtpbmdyZWRp"
    "ZW50c4GiY3VybHgtc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5pbmdyZWRpZW50LnYzZGhhc2hYILtElnmDFZYHbQGovNnKmUsYhXrd1vGJ"
    "tJ4PAJy+9w+IpGZhY3Rpb254HWNvbS5hbnRocm9waWMuY2xhdWRlLnByb3ZpZGVkanBhcmFtZXRlcnOheB9jb20uYW50aHJvcGljLm9yaWdpbi1jb25m"
    "aWRlbmNlZ3Vua25vd25rZGVzY3JpcHRpb254ZkNsYXVkZSBwcm92aWRlZCB0aGlzIGZpbGUgYXQgdGhlIHJlcXVlc3Qgb2YgYSB1c2VyIGFuZCBtYXkg"
    "aGF2ZSBjcmVhdGVkIG9yIG1vZGlmaWVkIHRoZSBmaWxlIGNvbnRlbnRzLm1zb2Z0d2FyZUFnZW50oWRuYW1lZkNsYXVkZXJhbGxBY3Rpb25zSW5jbHVk"
    "ZWT1AAAAyGp1bWIAAABAanVtZGNib3IAEQAQgAAAqgA4m3ETYzJwYS5oYXNoLmRhdGEAAAAAGGMyc2hx2wlX1SmHjZnP7CIhhOx4AAAAgGNib3KlY2Fs"
    "Z2ZzaGEyNTZjcGFkTQAAAAAAAAAAAAAAAABkaGFzaFggGNypyHpAu7ngbhLFkE2xjf9z30Hz29h1D7omM5fhKeZkbmFtZW5qdW1iZiBtYW5pZmVzdGpl"
    "eGNsdXNpb25zgaJlc3RhcnQYIWZsZW5ndGgZFooAAAI+anVtYgAAACdqdW1kYzJjbAARABCAAACqADibcQNjMnBhLmNsYWltLnYyAAAAAg9jYm9ypWNh"
    "bGdmc2hhMjU2aXNpZ25hdHVyZXhNc2VsZiNqdW1iZj0vYzJwYS91cm46YzJwYTo4Nzk0NDBiOS0wN2QwLTQ4ZjQtYjRjNS01YzE1MDYzYTNkYmMvYzJw"
    "YS5zaWduYXR1cmVqaW5zdGFuY2VJRHgseG1wOmlpZDpiNmIxNmNjNy0zZTIwLTRjNDYtYTkxMi00NTQ5NjY4NjQ1ZDZyY3JlYXRlZF9hc3NlcnRpb25z"
    "g6JjdXJseC1zZWxmI2p1bWJmPWMycGEuYXNzZXJ0aW9ucy9jMnBhLmluZ3JlZGllbnQudjNkaGFzaFggu0SWeYMVlgdtAai82cqZSxiFet3W8Ym0ng8A"
    "nL73D4iiY3VybHgqc2VsZiNqdW1iZj1jMnBhLmFzc2VydGlvbnMvYzJwYS5hY3Rpb25zLnYyZGhhc2hYICfGd2xQh7hKguAnu0PnRNa3nExD2pwqOByz"
    "SutvLg9momN1cmx4KXNlbGYjanVtYmY9YzJwYS5hc3NlcnRpb25zL2MycGEuaGFzaC5kYXRhZGhhc2hYIKKshV2pmVRtuPS2lyCiuaJHSvSQoh1kXTX7"
    "nFP8Gzx3dGNsYWltX2dlbmVyYXRvcl9pbmZvo2RuYW1lb0FudGhyb3BpYyBGaWxlc2d2ZXJzaW9uZTEuMC4wa3NwZWNWZXJzaW9uZTIuNC4wAAAQOGp1"
    "bWIAAAAoanVtZGMyY3MAEQAQgAAAqgA4m3EDYzJwYS5zaWduYXR1cmUAAAAQCGNib3LShFkCEqIBJhghWQIKMIICBjCCAY2gAwIBAgIUQOWgCu7COdC+"
    "uIP6BkIFPWdVEwAwCgYIKoZIzj0EAwMwSTEXMBUGA1UEChMOQW50aHJvcGljLCBQQkMxLjAsBgNVBAMTJUFudGhyb3BpYyBDb250ZW50IENyZWRlbnRp"
    "YWxzIFJvb3QgQ0EwHhcNMjYwODA3MTg0MzU2WhcNMjgwODA2MTk0MzU2WjBEMRcwFQYDVQQKEw5BbnRocm9waWMsIFBCQzEpMCcGA1UEAxMgQW50aHJv"
    "cGljIENsYXVkZSBDb250ZW50IFNpZ25pbmcwWTATBgcqhkjOPQIBBggqhkjOPQMBBwNCAASYegpry1AYBRTVNL1CpTlbROnY3dey+UrsF9C3phYrATN3"
    "ZHf93Mo8RQN0KOUuOn19P4oWNFWe5n2/She9N7eTo1gwVjAOBgNVHQ8BAf8EBAMCB4AwFQYDVR0lBA4wDAYKKwYBBAGD6F4CATAMBgNVHRMBAf8EAjAA"
    "MB8GA1UdIwQYMBaAFM5R4gSBTmRbI/jjxM+aPpzB11zCMAoGCCqGSM49BAMDA2cAMGQCMDFzHRSeAXrSy1WOzkbhPZ6Km2wGTmZ/2gK18k8BQGXyqz88"
    "Rdrz6CTX9flAnYNVxgIwcF9c3fVhqmJKpi+UhasNUMko69cyX6STPfta3Q8EjyzDjzoyrol46FP6VFHhvUcJoWNwYWRZDZ4AAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD2WEABNrHrA1+J"
    "zEgWUCx5lUWH5UudV5Y1rl98b4FQB59PTkS/pEok/MwlTNSTfYCJ8qGdjyelfhrmaqW2uMTDZ2FVSn01LQABAABJREFUeJzs/XeYHdd15gv/9t4VTj59"
    "OjdyBojAnEkwkyJFSbayZOVky5JGGlmeud947Ovnjscej8eiAi05SCNKpIKVLVOZmQRzAIlEgMhooNG5++RTae/vj13dAD2eGd87DgSM9TyHALsb3dV1"
    "qt5a613vepcwxnAmzsSZOBP/WCH/pQ/gTJyJM3F6xRlQORNn4kz8o8YZUDkTZ+JM/KOG8y99AGfiny4+fdunjUAAMEedGQMCEMK+NIJOp81//J3fEf9y"
    "R3omTqcQZ4jaUz8+fdttxpGCWBscR+JIgTagjUEakClcxNqgBOjkxHseC4EWwn6tFAhjQBuEEGhtUFLwyU9+8gzgnIl/cJwBlVMoPvOnnzZGCAyAAOUo"
    "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"
)



class CustomCharacter:
    """character 폴더에 준비한 손그림 캐릭터 이미지를 사용한다.
    character/full.png (전신을 하나로 그린 그림, 오른쪽을 보고 선 정면 전신, 배경 투명 PNG 권장)이
    있으면 막대인간 대신 이 그림을 캐릭터로 쓴다. character/full_p2.png를 따로 준비하면 P2가
    다른 그림을 쓰고, 없으면 두 사람이 같은 그림을 공유한다.
    지금은 그림이 통째로 한 장이라 팔다리가 따로 구부러지지는 않고, 몸 전체가 하나처럼
    위치·좌우 반전·기울기·점프 시 신축·착지 시 찌그러짐으로 움직인다. 머리·몸통·팔·다리를
    부위별로 나눠 그린 PNG를 따로 준비하면 그 다음 단계로 관절별 애니메이션도 지원할 수 있다."""

    def __init__(self):
        self.ok = False
        self.error = ""
        self.images = {}
        self.source = ""    # "external"=character 폴더 사용자 지정 그림, "embedded"=코드 내장 기본 그림
        self.search_dirs = []     # 진단용: 실제로 뒤진 폴더 목록
        self.dir_listing = {}     # 진단용: 각 폴더 안에 실제로 들어있던 파일 목록
        self.p1_path, self.p2_path = None, None

        # __file__ 기준 절대경로 + 현재 실행 위치(cwd) 둘 다 뒤진다. 스크립트를 어디서
        # 실행하든(cwd 무관) 찾히게 하려는 것이었는데, "character 폴더를 다른 프로젝트
        # 폴더에 만들어서 실제 .py 파일 위치와 다른" 경우까지는 script_dir만으론 못 잡아서
        # cwd 기준도 함께 확인한다.
        candidates_dirs = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates_dirs.append(os.path.join(script_dir, CHARACTER_DIR))
        cwd_dir = os.path.join(os.getcwd(), CHARACTER_DIR)
        if cwd_dir not in candidates_dirs:
            candidates_dirs.append(cwd_dir)
        self.search_dirs = candidates_dirs

        # 폴더 안에 실제로 뭐가 들어있는지 먼저 다 읽어둔다(대소문자/오타 문제를
        # 콘솔에서 바로 눈으로 확인할 수 있게). 폴더가 없으면 "(폴더 없음)"으로 표시.
        for d in candidates_dirs:
            try:
                self.dir_listing[d] = sorted(os.listdir(d))
            except Exception:
                self.dir_listing[d] = None  # 폴더 자체가 없음

        exts = ("png", "jpg", "jpeg")
        p1_names = {"full"}
        p2_names = {"full_p2", "full2", "full_2", "p2"}

        def find_by_stem(stems):
            # 대소문자, 확장자 표기(.PNG 등)를 전부 무시하고 "이름(확장자 제외)"만 비교한다.
            for d in candidates_dirs:
                files = self.dir_listing.get(d)
                if not files:
                    continue
                for fname in files:
                    stem, dot, ext = fname.rpartition(".")
                    if not dot:
                        continue
                    if stem.lower() in stems and ext.lower() in exts:
                        return os.path.join(d, fname)
            return None

        p1 = find_by_stem(p1_names)
        p2 = find_by_stem(p2_names)
        self.p1_path, self.p2_path = p1, p2
        try:
            if p1:
                img1 = pygame.image.load(p1).convert_alpha()
                self.source = "external"
            else:
                png_bytes = base64.b64decode(EMBEDDED_CHARACTER_PNG_B64)
                img1 = pygame.image.load(io.BytesIO(png_bytes), "full.png").convert_alpha()
                self.source = "embedded"
            self.images[0] = img1
            self.images[1] = (pygame.image.load(p2).convert_alpha()
                               if p2 else img1)
            self.ok = True
        except Exception as e:
            self.error = str(e)

    def debug_lines(self):
        """진단 정보를 줄 단위 리스트로 만든다(콘솔 출력 + 파일 저장 둘 다에 사용)."""
        lines = []
        lines.append(f"손그림 캐릭터: {'사용 중 (' + self.source + ')' if self.ok else ('실패: ' + self.error)}")
        lines.append(f"  P1 그림: {self.p1_path or '못 찾음 (내장 기본 그림 사용)'}")
        lines.append(f"  P2 그림: {self.p2_path or '못 찾음 (P1과 같은 그림 사용)'}")
        lines.append("  찾아본 character 폴더와 그 안의 실제 파일 목록:")
        for d in self.search_dirs:
            files = self.dir_listing.get(d)
            lines.append(f"    - {d}")
            if files is None:
                lines.append("        (이 폴더 자체가 없음)")
            elif not files:
                lines.append("        (폴더는 있지만 안이 비어있음)")
            else:
                for fname in files:
                    lines.append(f"        {fname}")
        lines.append("  -> full.png / full2.png 파일이 위 목록에 정확히 보이는 폴더에 있어야 인식됩니다.")
        lines.append("     (이름이 다르게 보이면 그 이름 그대로 알려주세요.)")
        return lines


# ---------------------------------------------------------------- 그리기
class Renderer:
    def __init__(self):
        global KOR
        path = pygame.font.match_font("malgungothic,malgun gothic,applegothic,nanumgothic,notosanscjkkr")
        KOR = bool(path)
        self.f_s = pygame.font.Font(path, 15)
        self.f_m = pygame.font.Font(path, 17)
        self.f_b = pygame.font.Font(path, 22)
        rnd = random.Random(7)
        self.stars = [(rnd.randrange(WIDTH), rnd.randrange(GROUND_Y), rnd.choice([1, 1, 2]),
                       rnd.choice([0.1, 0.2, 0.35])) for _ in range(90)]
        self.hud_bg = pygame.Surface((430, 560), pygame.SRCALPHA)
        self.hud_bg.fill((0, 0, 0, 165))
        self.character = CustomCharacter()

    def text(self, surface, s, pos, color=(230, 230, 235), font=None):
        surface.blit((font or self.f_s).render(s, True, color), pos)

    def world(self, surface, logic, fx):
        surface.fill(BG)
        for x, y, size, par in self.stars:
            pygame.draw.circle(surface, (120, 120, 150), (int((x - logic.scroll * par) % WIDTH), y), size)
        floor_y = GROUND_Y + R0
        pygame.draw.rect(surface, (24, 24, 44), (0, floor_y, WIDTH, HEIGHT - floor_y))
        pygame.draw.line(surface, (90, 90, 130), (0, floor_y), (WIDTH, floor_y), 2)
        off = (-logic.scroll) % 80
        for i in range(-1, WIDTH // 80 + 2):
            x = int(i * 80 + off)
            pygame.draw.line(surface, (40, 40, 70), (x, floor_y), (x - 30, HEIGHT), 1)
        for p in logic.players:
            if p.visible and p.trail:
                n = len(p.trail)
                for i, (tx, ty) in enumerate(p.trail):
                    t = (i + 1) / n
                    pygame.draw.circle(surface, mix(BG, COLORS[p.slot], t * 0.7),
                                       (int(tx), int(ty)), int(R0 * (0.35 + 0.5 * t)))
        fx.draw(surface)
        for p in logic.players:
            self.player(surface, p, logic.level)
        self.draw_balls(surface, logic.balls)
        if logic.flash > 0:
            overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            overlay.fill((255, 255, 255, int(255 * logic.flash * 0.5)))
            surface.blit(overlay, (0, 0))

    def draw_balls(self, surface, balls):
        """공 피하기 미니게임의 공들을 그린다(위쪽 살짝 밝은 하이라이트 + 테두리)."""
        for b in balls:
            pygame.draw.circle(surface, b.color, (int(b.x), int(b.y)), b.r)
            dark = tuple(max(0, c - 70) for c in b.color)
            pygame.draw.circle(surface, dark, (int(b.x), int(b.y)), b.r, 2)
            hi_r = max(2, int(b.r * 0.28))
            pygame.draw.circle(surface, (255, 255, 255),
                                (int(b.x - b.r * 0.32), int(b.y - b.r * 0.32)), hi_r)

    def _capsule(self, surf, color, p1, p2, width):
        pygame.draw.line(surf, color, p1, p2, width)
        r = max(1, width // 2)
        pygame.draw.circle(surf, color, (int(round(p1[0])), int(round(p1[1]))), r)
        pygame.draw.circle(surf, color, (int(round(p2[0])), int(round(p2[1]))), r)

    def player(self, surface, p, level):
        if not p.visible:
            return
        if p.hit_invuln > 0 and (p.hit_invuln // 4) % 2 == 0:
            return  # 피격 직후 잠깐 깜빡이는 무적 효과
        if self.character.ok:
            self._player_sprite(surface, p, level)
        else:
            self._player_stick(surface, p, level)

    def _player_sprite(self, surface, p, level):
        """손으로 그린 캐릭터 그림(character/full.png)을 몸 전체 움직임에 맞춰
        위치·좌우 반전(진행 방향)·걷기 바운스&기울임·점프 신축·착지 스쿼시로 변형해서 그린다."""
        base = self.character.images[p.slot]
        unit = R0   # 목소리 크기로 캐릭터가 커지지 않으므로 크기는 항상 고정
        target_h = unit * 6.0   # 캐릭터 그림 전체 높이(공 피하기 게임에 맞춰 살짝 축소)

        moving = p.move_frames > 0
        jumping = p.in_air
        landing = p.squash > 0

        angle = 0.0
        scale_x, scale_y = 1.0, 1.0
        bob = 0.0
        if jumping:
            scale_y, scale_x = 1.08, 0.94
            angle = -p.facing * 6
        elif landing:
            k = p.squash / 6
            scale_y, scale_x = 1 - 0.28 * k, 1 + 0.22 * k
        elif moving:
            bob = abs(math.sin(p.anim_phase)) * unit * 0.12
            angle = -p.facing * 6 * math.sin(p.anim_phase)

        img = base
        if p.facing < 0:
            img = pygame.transform.flip(img, True, False)
        w0, h0 = img.get_size()
        h1 = max(2, target_h * scale_y)
        w1 = max(2, w0 * (h1 / h0) * scale_x)
        img = pygame.transform.smoothscale(img, (int(w1), int(h1)))
        if angle:
            img = pygame.transform.rotate(img, angle)
        img = img.copy()
        if level > 0.02:   # 목소리가 크면 캐릭터가 살짝 밝아짐 (크기는 그대로)
            glow = int(min(1.0, level) * 90)
            img.fill((glow, glow, glow, 0), special_flags=pygame.BLEND_RGB_ADD)
        img.set_alpha(p.alpha)

        rect = img.get_rect(midbottom=(int(p.x), int(p.y + R0 - bob)))
        surface.blit(img, rect)

        label = self.f_m.render(NAMES[p.slot], True, (255, 255, 255))
        label.set_alpha(p.alpha)
        surface.blit(label, (int(p.x - label.get_width() / 2), rect.top - 20))

    def _player_stick(self, surface, p, level):

        col = mix(COLORS[p.slot], (255, 255, 255), level * 0.6)
        col_a = (*col, p.alpha)
        pose = human_pose(p, level)
        limb_w = pose["limb_w"]

        joints = [pose["hip"], pose["hip_l"], pose["hip_r"], pose["foot_l"], pose["foot_r"],
                  pose["shoulder"], pose["sh_l"], pose["sh_r"], pose["hand_l"], pose["hand_r"],
                  (pose["head_c"][0], pose["head_c"][1] - pose["head_r"]),
                  (pose["head_c"][0], pose["head_c"][1] + pose["head_r"])]
        pad = limb_w + 6
        xs, ys = [q[0] for q in joints], [q[1] for q in joints]
        ox, oy = -(min(xs) - pad), -(min(ys) - pad)
        w, h = max(1, int(max(xs) - min(xs) + 2 * pad)), max(1, int(max(ys) - min(ys) + 2 * pad))

        def S(pt):
            return (pt[0] + ox, pt[1] + oy)

        fig = pygame.Surface((w, h), pygame.SRCALPHA)
        self._capsule(fig, col_a, S(pose["hip_l"]), S(pose["foot_l"]), limb_w)
        self._capsule(fig, col_a, S(pose["hip_r"]), S(pose["foot_r"]), limb_w)
        self._capsule(fig, col_a, S(pose["sh_l"]), S(pose["hand_l"]), max(3, limb_w - 2))
        self._capsule(fig, col_a, S(pose["sh_r"]), S(pose["hand_r"]), max(3, limb_w - 2))
        self._capsule(fig, col_a, S(pose["shoulder"]), S(pose["hip"]), int(limb_w * 1.6))
        hc, hr = S(pose["head_c"]), int(pose["head_r"])
        pygame.draw.circle(fig, col_a, (int(round(hc[0])), int(round(hc[1]))), hr)
        pygame.draw.circle(fig, (255, 255, 255, p.alpha), (int(round(hc[0])), int(round(hc[1]))), hr, 2)

        screen_bottom = p.y + R0
        surface.blit(fig, (int(p.x - ox), int(screen_bottom - oy)))

        label = self.f_m.render(NAMES[p.slot], True, (255, 255, 255))
        label.set_alpha(p.alpha)
        head_top_y = screen_bottom + pose["head_c"][1] - pose["head_r"]
        surface.blit(label, (int(p.x - label.get_width() / 2), int(head_top_y - 20)))

    def hud(self, screen, logic, fx, mic, src, n_hands, n_poses, fps, uptime, frames):
        screen.blit(self.hud_bg, (10, 10))
        x, y = 22, 18

        def line(s, color=(230, 230, 235), font=None, dy=22):
            nonlocal y
            self.text(screen, s, (x, y), color, font)
            y += dy

        line(L("● 실행 중  combined_game.py (손+몸)", "● RUNNING combined_game.py (hand+body)"), (120, 255, 150), self.f_b, 30)
        line(f"FPS {fps:4.1f}   {L('경과', 'uptime')} {int(uptime // 60):02d}:{int(uptime % 60):02d}", (170, 170, 190))
        OK, ERR, OFF = (110, 240, 140), (255, 90, 90), (255, 210, 90)
        line(f"● pygame {pygame.version.ver}  {L('창 실행', 'window')} {frames}{L('프레임', ' frames')}", OK)
        if fx.ok:
            line(f"● pygame-particles  {L('파티클 발사', 'fired')} {fx.fired}", OK)
        else:
            line(f"● pygame-particles  ERROR {fx.error[:40]}", ERR)
        if mic.ok:
            line(f"● sounddevice  {L('마이크', 'mic')} {mic.name[:26]}  {L('콜백', 'cb')}{mic.callbacks}", OK)
        else:
            line(f"● sounddevice  {L('마이크 없음', 'no mic')}", OFF)
        if src.hand_ok:
            line(f"● mediapipe(손)  {L('인식', 'hands')} {n_hands}{L('개', '')}", OK)
        else:
            line(f"● mediapipe(손)  OFF ({src.hand_error[:30]})", OFF)
        if src.pose_ok:
            line(f"● mediapipe(몸)  {L('인식', 'poses')} {n_poses}{L('명', '')}  ({L('카메라', 'cam')} {src.frames})", OK)
        else:
            line(f"● mediapipe(몸)  OFF ({src.pose_error[:30]})", OFF)
        line(f"{L('효과음', 'sound')}: {'ON' if fx.sound_ok else 'OFF'}", (170, 170, 190))
        if self.character.ok:
            tag = L('character/full.png 사용자 그림', 'your character/full.png') if self.character.source == "external" \
                else L('기본 내장 그림', 'built-in default')
            line(f"● {L('손그림 캐릭터', 'custom sprite')}: {L('사용 중', 'active')} ({tag})", (110, 240, 140), dy=26)
        else:
            line(f"○ {L('손그림 캐릭터', 'custom sprite')}: {L('로딩 실패 - 기본 막대인간 사용', 'load failed - using stick figure')}", (170, 170, 190), dy=26)

        pygame.draw.rect(screen, (70, 70, 90), (x, y, 300, 14), 2)
        pygame.draw.rect(screen, (110, 240, 140), (x + 2, y + 2, int(296 * min(1.0, logic.peak_disp)), 10))
        tx = x + int(300 * min(1.0, logic.clap_threshold))
        pygame.draw.line(screen, (255, 90, 90), (tx, y - 4), (tx, y + 18), 2)
        self.text(screen, f"{L('박수 기준', 'clap thr')} {logic.clap_threshold:.2f}", (x + 312, y - 3), (255, 140, 140))
        y += 28

        for p in logic.players:
            color = COLORS[p.slot]
            tag = L("[손]", "[hand]") if p.source == "hand" else L("[몸]", "[body]") if p.source == "body" else ""
            self.text(screen, f"{NAMES[p.slot]}: {p.gesture:8s} {tag} {p.score:.2f}" if p.gesture != "NO INPUT"
                      else f"{NAMES[p.slot]}: -", (x, y), color, self.f_m)
            y += 22
            if p.gesture != "NO INPUT":
                d = p.diag
                self.text(screen, f"   {L('몸 관절 인식률', 'body joints')} 어깨{d['어깨']*100:.0f}% "
                          f"엉덩이{d['엉덩이']*100:.0f}% 손목{d['손목']*100:.0f}%", (x, y), (150, 200, 255))
                y += 20
        y += 6
        c = logic.counters
        rows = [
            f"{L('등장', 'APPEAR')} {c['APPEAR']}  {L('이동', 'MOVE')} {c['MOVE']}  "
            f"{L('공 피함', 'DODGE')} {c['BALL_DODGE']}  {L('피격', 'HIT')} {c['BALL_HIT']}  {L('사망', 'OUT')} {c['PLAYER_OUT']}",
            f"{L('인식성공', 'OK')} {c['GESTURE_OK']}  {L('인식실패', 'FAIL')} {c['GESTURE_FAIL']}",
            f"{L('박수', 'CLAP')} {c['CLAP']}  {L('리셋', 'RESET')} {c['RESET']}  {L('스크린샷', 'SHOT')} {c['SCREENSHOT']}",
        ]
        self.text(screen, L("효과 발동 횟수", "effect counters"), (x, y), (170, 170, 190))
        y += 20
        for r in rows:
            self.text(screen, r, (x, y), (255, 240, 150))
            y += 20
        y += 8
        self.text(screen, L("최근 이벤트", "recent events"), (x, y), (170, 170, 190))
        y += 20
        for entry in list(logic.log)[::-1]:
            self.text(screen, entry, (x, y), (200, 200, 215))
            y += 19

    def guidance(self, screen, src, n_hands, n_poses, mic, uptime, stats):
        if not src.ok:
            msg = L("웹캠을 열 수 없습니다 - 다른 프로그램이 카메라를 쓰고 있는지 확인하세요",
                    "cannot open webcam - check if another app is using it")
        elif n_hands == 0 and n_poses == 0:
            msg = L("손도 몸도 인식되지 않습니다 - 카메라에 상반신(또는 손)이 잘 보이게 서주세요",
                    "no hand or body detected - make sure your upper body or hand is in frame")
        elif mic.ok and uptime > 2.0 and mic.callbacks == 0:
            msg = L("마이크 콜백이 호출되지 않습니다 - OS 마이크 권한을 확인하세요 (설정 > 개인정보 > 마이크)",
                    "mic callback never fired - check OS microphone permission for this app")
        elif mic.ok and uptime > 3.0 and mic.callbacks > 0 and stats["max_rms"] < 0.003:
            msg = L("마이크가 응답은 하지만 소리가 감지되지 않습니다 - 다른 마이크가 선택됐거나 음소거 상태일 수 있습니다",
                    "mic is running but no sound detected - wrong device selected or muted")
        else:
            return
        box = pygame.Surface((min(WIDTH - 40, 900), 50), pygame.SRCALPHA)
        box.fill((0, 0, 0, 190))
        screen.blit(box, (WIDTH // 2 - box.get_width() // 2, HEIGHT // 2 - 25))
        label = self.f_b.render(msg, True, (255, 210, 90))
        screen.blit(label, (WIDTH // 2 - label.get_width() // 2, HEIGHT // 2 - label.get_height() // 2))

    def preview(self, screen, frame, hands, poses):
        if frame is None:
            return
        h, w = frame.shape[:2]
        assigned_hands = assign_players_hand(hands)
        for slot, (lm, _) in assigned_hands.items():
            pts = [(int(q.x * w), int(q.y * h)) for q in lm]
            bgr = COLORS[slot][::-1]
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], bgr, 2)
        assigned_poses = assign_players_pose(poses)
        for slot, lm in assigned_poses.items():
            bgr = COLORS[slot][::-1]
            for a, b in POSE_CONNECTIONS:
                pa, pb = lm[a], lm[b]
                if pa.visibility < 0.3 or pb.visibility < 0.3:
                    continue
                cv2.line(frame, (int(pa.x * w), int(pa.y * h)), (int(pb.x * w), int(pb.y * h)), bgr, 2)
            nose = lm[NOSE]
            if nose.visibility >= 0.3:
                cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 6, bgr, -1)
        small = cv2.cvtColor(cv2.resize(frame, (PREV_W, PREV_H)), cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(small.tobytes(), (PREV_W, PREV_H), "RGB")
        pos = (WIDTH - PREV_W - 12, 12)
        screen.blit(surf, pos)
        pygame.draw.rect(screen, (255, 255, 255), (*pos, PREV_W, PREV_H), 2)

    def help_line(self, screen):
        s = L("ESC 종료 | T 효과 테스트 | S 스크린샷 | V 카메라 | H 패널 | ↑↓ 박수 감도 | R 리셋",
              "ESC quit | T test | S shot | V cam | H panel | Up/Down clap thr | R reset")
        self.text(screen, s, (WIDTH // 2 - 320, HEIGHT - 26), (170, 170, 190))

    def dodge_hud(self, screen, logic):
        """공 피하기 미니게임 점수/목숨을 화면 상단 중앙에 항상 표시한다."""
        shown = min(int(logic.dodge_score), WIN_SCORE)
        score_s = L(f"점수 {shown} / {WIN_SCORE}", f"Score {shown} / {WIN_SCORE}")
        w = self.f_b.size(score_s)[0]
        self.text(screen, score_s, (WIDTH // 2 - w // 2, 14), (255, 235, 150), self.f_b)
        best_s = L(f"최고 점수 {max(logic.best_score, shown)}", f"Best {max(logic.best_score, shown)}")
        wb = self.f_m.size(best_s)[0]
        self.text(screen, best_s, (WIDTH // 2 - wb // 2, 46), (170, 220, 255), self.f_m)

        heart_full, heart_empty = "♥", "♡"
        for i, p in enumerate(logic.players):
            if p.dead:
                s = L(f"{NAMES[p.slot]} 사망(OUT)", f"{NAMES[p.slot]} OUT")
                col = (140, 140, 150)
            else:
                hearts = heart_full * max(0, p.lives) + heart_empty * max(0, BALL_START_LIVES - p.lives)
                s = f"{NAMES[p.slot]} {hearts}"
                col = COLORS[p.slot]
            x = 18 if i == 0 else WIDTH - 150
            self.text(screen, s, (x, 14), col, self.f_m)

        if logic.game_over:
            lines = [("GAME OVER", (255, 110, 110), self.f_b)]
            if logic.cleared:
                lines.append((L(f"{WIN_SCORE}점 달성!", f"{WIN_SCORE} points reached!"), (255, 235, 120), self.f_b))
            else:
                lines.append((L(f"점수 {logic.final_score}", f"Score {logic.final_score}"), (255, 235, 120), self.f_b))
            rec = L(f"최고 점수 {logic.best_score}" + ("  (신기록!)" if logic.new_record else ""),
                    f"Best {logic.best_score}" + ("  (new record!)" if logic.new_record else ""))
            lines.append((rec, (170, 220, 255), self.f_m))
            lines.append((L("R 키로 다시 시작", "press R to restart"), (230, 230, 240), self.f_m))
            bw = max(f.size(t)[0] for t, _, f in lines) + 60
            bh = 30 + sum(f.get_linesize() + 6 for _, _, f in lines)
            box = pygame.Surface((bw, bh), pygame.SRCALPHA)
            box.fill((0, 0, 0, 200))
            top = HEIGHT // 2 - bh // 2
            screen.blit(box, (WIDTH // 2 - bw // 2, top))
            yy = top + 16
            for t, col, f in lines:
                self.text(screen, t, (WIDTH // 2 - f.size(t)[0] // 2, yy), col, f)
                yy += f.get_linesize() + 6
        elif logic.round_phase == "ready":
            if any(p.visible for p in logic.players):
                n = max(1, int(logic.ready_timer) + 1)
                s = L(f"공이 곧 떨어집니다! 준비하세요... {n}", f"Balls incoming! Get ready... {n}")
            else:
                s = L("카메라에 손이나 몸을 보여주면 시작합니다", "Show your hand or body to the camera to start")
            w2 = self.f_b.size(s)[0]
            box = pygame.Surface((w2 + 44, 58), pygame.SRCALPHA)
            box.fill((0, 0, 0, 190))
            screen.blit(box, (WIDTH // 2 - (w2 + 44) // 2, HEIGHT // 2 - 29))
            self.text(screen, s, (WIDTH // 2 - w2 // 2, HEIGHT // 2 - 12), (255, 235, 120), self.f_b)


# ---------------------------------------------------------------- 실행 기록
def write_run_log(logic, fx, mic, src, character, stats, start_wall, duration, path="run_log.txt"):
    lines = [
        "combined_game.py 실행 기록 (손 + 몸 관절 동시 인식)",
        f"시작: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_wall))}   실행 시간: {duration:.1f}초",
        "",
        "[환경]",
        f"Python {sys.version.split()[0]}",
        f"pygame {pygame.version.ver} | pygame-particles {getattr(pygame_particles, '__version__', '?')} | "
        f"sounddevice {getattr(sd, '__version__', '?')} | mediapipe {mp.__version__} | opencv {cv2.__version__} | numpy {np.__version__}",
        "",
        "[라이브러리별 실행 확인]",
        f"pygame           : 창 실행 {stats['frames']}프레임",
        f"pygame-particles : {'정상' if fx.ok else '오류 ' + fx.error} / 파티클 {fx.fired}개 발사",
        f"sounddevice      : {'마이크 ' + mic.name if mic.ok else '사용 불가 ' + mic.error} / 콜백 {mic.callbacks}회 / 최대 음량(RMS) {stats['max_rms']:.3f} / 최대 피크 {stats['max_peak']:.3f}",
        f"mediapipe(손)    : {'정상' if src.hand_ok else '사용 불가 ' + src.hand_error} / 손 감지 {stats['frames_with_hand']}프레임",
        f"mediapipe(몸)    : {'정상' if src.pose_ok else '사용 불가 ' + src.pose_error} / 자세 감지 {stats['frames_with_pose']}프레임",
        f"카메라 처리      : {src.frames}프레임",
        f"효과음(mixer)    : {'ON' if fx.sound_ok else 'OFF'}",
        f"손그림 캐릭터    : {'사용 중 (' + character.source + ')' if character.ok else '로딩 실패(기본 막대인간) - ' + character.error}",
        "",
        "[손/몸 입력 소스별 제스처 인식 성공 횟수]",
        f"손으로 인식: {stats['gesture_by_hand']}   몸으로 인식: {stats['gesture_by_body']}",
        "",
        "[효과 발동 횟수]",
    ]
    lines += [f"{k}: {v}" for k, v in sorted(logic.counters.items())] or ["(없음)"]
    lines += ["", "[공 피하기 결과]", f"최종 점수: {min(int(logic.dodge_score), WIN_SCORE)}", f"최고 점수: {logic.best_score}"]
    lines += ["", "[이벤트 기록]"] + logic.full_log
    lines += ["", "※ 손 모양이 뚜렷하면 손을, 안 보이거나 애매하면 몸(어깨·엉덩이·손목) 자세를 자동으로 사용함"]
    lines += ["※ 공 피하기 게임에는 점프가 없음(좌우 이동으로만 피함), 공에 한 번 맞으면 즉시 사망함"]
    lines += ["※ 카메라 영상과 마이크 소리는 저장하지 않음 (음량 수치만 사용)"]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


# ---------------------------------------------------------------- 메인
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("combined_game - hand + body + particles + mic")
    clock = pygame.time.Clock()
    world = pygame.Surface((WIDTH, HEIGHT))

    renderer = Renderer()
    src, mic = CombinedSource(), Mic()
    logic, fx = GameLogic(), Effects()

    print("==== combined_game 실행 확인 ====")
    print(f"[OK] pygame {pygame.version.ver}")
    print(f"[{'OK' if fx.ok else 'ERR'}] pygame-particles  (효과음: {'ON' if fx.sound_ok else 'OFF'})")
    _char_debug_lines = renderer.character.debug_lines()
    for _line in _char_debug_lines:
        print(_line)
    try:
        _debug_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "character_debug.txt")
        with open(_debug_path, "w", encoding="utf-8") as _f:
            _f.write("\n".join(_char_debug_lines))
        print(f"  (같은 폴더에 character_debug.txt 로도 저장했습니다: {_debug_path})")
    except Exception as _e:
        print(f"  (character_debug.txt 저장 실패: {_e})")
    print(f"[{'OK' if mic.ok else 'OFF'}] sounddevice  {mic.name if mic.ok else mic.error}")
    print(f"[{'OK' if src.hand_ok else 'OFF'}] mediapipe(손)  {src.hand_error if not src.hand_ok else ''}")
    print(f"[{'OK' if src.pose_ok else 'OFF'}] mediapipe(몸)  {src.pose_error if not src.pose_ok else ''}")
    print(f"[{'OK' if src.ok else 'OFF'}] 카메라  {'ON' if src.ok else src.error}")
    print("T 키를 누르면 카메라·마이크 없이도 모든 효과를 확인할 수 있습니다.")
    if mic.ok:
        print(f"  마이크 콜백이 계속 0회면(화면 상태판 'sounddevice' 줄) OS가 마이크 접근을 막고 있는 것입니다.")
        print("  → Windows: 설정 > 개인정보 및 보안 > 마이크 > '앱이 마이크에 액세스하도록 허용' 켜기")
        print("  → macOS: 시스템 설정 > 개인정보 보호 및 보안 > 마이크 > 터미널/IDE 허용")
    print("※ 손 모양이 뚜렷하면 손, 안 보이면 몸(자세)을 자동으로 사용합니다.")
    print("  화면 좌측 상태판에서 각 사람이 지금 [손]/[몸] 중 어느 쪽으로 인식됐는지 확인할 수 있습니다.")

    stats = {"frames": 0, "frames_with_hand": 0, "frames_with_pose": 0,
              "max_rms": 0.0, "max_peak": 0.0, "gesture_by_hand": 0, "gesture_by_body": 0}
    show_hud, show_cam = True, True
    start_wall, start = time.time(), time.monotonic()
    running = True
    while running:
        now = time.monotonic()
        shot = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif k == pygame.K_t:
                    logic.self_test()
                elif k == pygame.K_r:
                    logic.reset_all()
                elif k == pygame.K_s:
                    shot = True
                elif k == pygame.K_v:
                    show_cam = not show_cam
                elif k == pygame.K_h:
                    show_hud = not show_hud
                elif k == pygame.K_UP:
                    logic.clap_threshold = min(0.9, logic.clap_threshold + 0.05)
                elif k == pygame.K_DOWN:
                    logic.clap_threshold = max(0.05, logic.clap_threshold - 0.05)

        frame, hands, poses = src.read()
        rms, peak = mic.read()
        stats["frames"] += 1
        stats["frames_with_hand"] += 1 if hands else 0
        stats["frames_with_pose"] += 1 if poses else 0
        stats["max_rms"] = max(stats["max_rms"], rms)
        stats["max_peak"] = max(stats["max_peak"], peak)

        logic.update(hands, poses, rms, peak, now)
        for ev in logic.drain():
            if ev["name"] == "GESTURE_OK":
                if ev.get("source") == "hand":
                    stats["gesture_by_hand"] += 1
                elif ev.get("source") == "body":
                    stats["gesture_by_body"] += 1
            fx.handle(ev, logic)

        renderer.world(world, logic, fx)
        ox = random.randint(-6, 6) if logic.shake > 0 else 0
        oy = random.randint(-6, 6) if logic.shake > 0 else 0
        screen.fill(BG)
        screen.blit(world, (ox, oy))
        if show_hud:
            renderer.hud(screen, logic, fx, mic, src, len(hands), len(poses), clock.get_fps(), now - start, stats["frames"])
        renderer.guidance(screen, src, len(hands), len(poses), mic, now - start, stats)
        renderer.help_line(screen)
        renderer.dodge_hud(screen, logic)

        if shot:
            os.makedirs("screenshots", exist_ok=True)
            name = os.path.join("screenshots", time.strftime("shot_%Y%m%d_%H%M%S.png"))
            pygame.image.save(screen, name)
            logic.emit("SCREENSHOT", source=name)
            print("스크린샷 저장:", name)

        if show_cam:
            renderer.preview(screen, frame, hands, poses)
        pygame.display.flip()
        clock.tick(FPS)

    path = write_run_log(logic, fx, mic, src, renderer.character, stats, start_wall, time.monotonic() - start)
    print("실행 기록 저장:", path)
    src.close()
    mic.close()
    pygame.quit()


if __name__ == "__main__":
    main()