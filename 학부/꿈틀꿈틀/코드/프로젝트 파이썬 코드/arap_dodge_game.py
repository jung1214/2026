"""
arap_dodge_game.py  -  캠으로 조작하는 "공 피하기" (ARAP 방식)

test_game.py(키보드로 움직이는 공 피하기)를 바탕으로 만든 새 파일입니다.

[무엇이 달라졌나]
  1) 캠 속 내 위치(좌우)를 그대로 따라 캐릭터가 좌우로 움직입니다. (거울처럼)
  2) 캐릭터는 관절별로 잘라서 돌리는 방식이 아니라, 그림 한 장(character.png)을
     그물(메쉬)로 덮고 ARAP(As-Rigid-As-Possible) 방식으로 통째로 변형합니다.
     - 그림 위에 조종점(머리/목/엉덩이/양손/양발) 7개를 잡고
     - 캠에서 본 내 몸 위치에 맞춰 조종점을 옮기면
     - 나머지 그물 점들은 "각자 원래 모양을 최대한 유지(회전만 허용)"하도록 계산됩니다.
     그래서 팔을 들거나 몸을 기울이면 그림이 끊기지 않고 자연스럽게 휘어집니다.
  3) 공을 GOAL개 피하면 클리어. 공은 처음엔 느리고, 하나 피할 때마다 점점 빨라집니다.
     게임이 끝나면(클리어/게임 오버) END_WAIT초 뒤 자동으로 종료됩니다.
     (scan_and_play.py 로 실행했다면 스캐너로 돌아갑니다)

설치:  pip install pygame numpy opencv-python mediapipe
실행:  python arap_dodge_game.py      (pose_landmarker.task 가 같은 폴더에 있어야 함)

[조작]
  캠 앞에서 좌우로 움직이기   캐릭터 좌우 이동 (위치 따라가기)
  팔 들기 / 몸 기울이기        캐릭터도 같이 휘어짐 (ARAP)
  ← →                          캠이 나를 못 찾을 때 키보드로 이동
  R 다시 시작 | D 그물(메쉬) 보기 | V 캠 화면 켜기/끄기 | ESC 종료

[개인정보] 캠 영상은 화면에만 표시하고 저장하지 않습니다.
"""
import math
import os
import random
import threading
import time
import urllib.request

import cv2
import numpy as np
import pygame

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False

# ------------------------------------------------------------------ 설정값
screen_width = 800
screen_height = 600

black = (0, 0, 0)
white = (255, 255, 255)
red = (255, 0, 0)

PLAYER_HEIGHT = 170      # 캐릭터 키(픽셀). 그림 비율은 그대로 유지
player_speed = 10        # 키보드로 움직일 때 속도
ball_size = 40

# --- 난이도 / 게임 끝 ---
START_SPEED = 3.0        # 첫 공 속도(프레임당 픽셀). 60fps 기준 화면 끝까지 약 3초
SPEED_UP = 0.5           # 공 하나 피할 때마다 빨라지는 양
MAX_SPEED = 14.0         # 최고 속도(이 이상은 안 빨라짐)
GOAL = 20                # 이만큼 피하면 클리어
END_WAIT = 3             # 끝난 뒤 이 시간(초)이 지나면 자동 종료 (0 이하면 자동 종료 안 함)

# --- 캠 위치 따라가기 ---
CAM_MARGIN = 0.22        # 캠 화면 양끝 22%는 안 씀 → 가운데 56%만 움직여도 화면 끝까지 감
FOLLOW_SMOOTH = 0.30     # 따라가는 빠르기(0~1). 클수록 즉시 따라오고, 작을수록 부드러움
LOST_TIME = 0.6          # 이 시간(초) 넘게 몸이 안 보이면 키보드 조작으로 전환

# --- ARAP 메쉬 ---
GRID_STEP = 8            # 그물 한 칸 크기(픽셀). 작을수록 부드럽지만 느려짐
ARAP_ITERS = 3           # 한 프레임에 ARAP 계산 반복 횟수
HANDLE_WEIGHT = 500.0    # 조종점을 목표 위치에 붙잡는 힘
POSE_SMOOTH = 0.35       # 동작 따라가는 빠르기(0~1)
ARM_RATIO = 1.5          # 사람 팔 길이 ≈ 어깨너비 × 1.5 로 가정
MAX_LEAN = math.radians(35)
MOVE_LEAN = 0.012        # 좌우로 움직일 때 진행 방향으로 살짝 기우는 정도(0이면 끔)
VIS = 0.5                # 관절 신뢰도 기준(이보다 낮으면 그 부위는 기본 자세 유지)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL_PATH = os.path.join(BASE_DIR, "pose_landmarker.task")
POSE_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")
PREVIEW_SIZE = (200, 150)

# mediapipe 몸 좌표 번호 (위치를 알아내는 용도로만 사용)
NOSE = 0
SIDE_A = {"sh": 11, "el": 13, "wr": 15, "hip": 23, "knee": 25, "ank": 27}
SIDE_B = {"sh": 12, "el": 14, "wr": 16, "hip": 24, "knee": 26, "ank": 28}

HANDLES = ["head", "neck", "hip", "hand_l", "hand_r", "foot_l", "foot_r"]


# ------------------------------------------------------------------ 캐릭터 그림 찾기
# 앞에 적힌 이름부터 찾는다. character.png가 있으면 그걸 쓰고, 없으면 full.png를 쓴다.
CHARACTER_FILES = ["character.png", "full.png"]
CANDIDATE_DIRS = [
    BASE_DIR,
    os.path.join(BASE_DIR, "character"),
    os.path.join(BASE_DIR, "charactor"),
    os.path.join(BASE_DIR, "charater"),
]


def find_character_png():
    for wanted in CHARACTER_FILES:
        for d in CANDIDATE_DIRS:
            if not os.path.isdir(d):
                continue
            for fname in os.listdir(d):
                if fname.lower() == wanted:
                    return os.path.join(d, fname)
    return None


def remove_white_background(rgba):
    """배경이 투명하지 않은 그림이면, 가장자리와 이어진 흰색 부분만 투명하게 만든다.
    (캐릭터 안쪽의 흰색 - 얼굴, 눈 등 - 은 가장자리와 떨어져 있어서 그대로 남는다)"""
    if (rgba[:, :, 3] < 250).mean() > 0.02:      # 이미 투명한 배경이 있으면 그대로 사용
        return rgba
    white = (rgba[:, :, :3].min(axis=2) > 225).astype(np.uint8)
    _, labels = cv2.connectedComponents(white, connectivity=4)
    border = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    background = np.isin(labels, np.unique(border[border > 0]))
    rgba = rgba.copy()
    rgba[background, 3] = 0
    return rgba


def rotate(v, angle):
    """벡터 v를 angle(라디안)만큼 회전. 화면 좌표(y가 아래)라서 +가 시계 방향."""
    c, s = math.cos(angle), math.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ------------------------------------------------------------------ 그림에서 조종점 위치 자동으로 잡기
def find_character_points(alpha):
    """투명하지 않은 부분(alpha)의 모양만 보고 머리/목/엉덩이/손/발 위치를 추정한다.
    팔을 양옆으로 벌리고 정면을 보고 선 그림을 기준으로 한다.
    (결과가 어긋나면 게임에서 D 키로 조종점 위치를 확인할 수 있음)"""
    mask = alpha > 40
    ys, xs = np.nonzero(mask)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    cx = (x0 + x1) / 2

    # 손: 몸 가운데 높이 구간에서 가장 왼쪽 / 가장 오른쪽 끝
    band = (ys > y0 + 0.30 * h) & (ys < y0 + 0.90 * h)
    bx, by = xs[band], ys[band]
    left = bx <= bx.min() + 0.04 * w
    right = bx >= bx.max() - 0.04 * w
    hand_l = np.array([bx[left].mean() + 0.03 * w, by[left].mean()])
    hand_r = np.array([bx[right].mean() - 0.03 * w, by[right].mean()])

    # 발: 맨 아래 구간을 좌우로 나눠 각각의 중심
    low = ys > y1 - 0.07 * h
    lx, ly = xs[low], ys[low]
    if (lx < cx).any() and (lx >= cx).any():
        foot_l = np.array([lx[lx < cx].mean(), ly[lx < cx].mean()])
        foot_r = np.array([lx[lx >= cx].mean(), ly[lx >= cx].mean()])
    else:
        foot_l = np.array([lx.mean() - 0.08 * w, ly.mean()])
        foot_r = np.array([lx.mean() + 0.08 * w, ly.mean()])
    foot_y = (foot_l[1] + foot_r[1]) / 2
    body_x = (foot_l[0] + foot_r[0]) / 2

    # 어깨 높이 ≈ 양손 높이(팔을 벌린 그림 기준), 엉덩이는 어깨와 발의 중간
    shoulder_y = clamp((hand_l[1] + hand_r[1]) / 2, y0 + 0.25 * h, y0 + 0.62 * h)
    neck = np.array([body_x, shoulder_y])
    hip = np.array([body_x, shoulder_y + 0.5 * (foot_y - shoulder_y)])

    # 머리: 어깨보다 위쪽 부분의 중심
    top = ys < shoulder_y - 0.05 * h
    head = np.array([xs[top].mean(), ys[top].mean()])

    return {"head": head, "neck": neck, "hip": hip,
            "hand_l": hand_l, "hand_r": hand_r, "foot_l": foot_l, "foot_r": foot_r}


# ------------------------------------------------------------------ ARAP 메쉬
class ArapMesh:
    """그림 한 장을 삼각형 그물로 덮고, 조종점 몇 개만 옮기면 나머지를 ARAP으로 계산한다."""

    def __init__(self, rgba, step=GRID_STEP):
        self.h, self.w = rgba.shape[:2]
        # 팔을 머리 위로 들거나 몸을 기울여도 잘리지 않게 사방에 여백을 둔다
        self.pad = int(0.40 * self.h)
        self.canvas_w = self.w + 2 * self.pad
        self.canvas_h = self.h + 2 * self.pad
        self.rgba = np.ascontiguousarray(rgba)

        self._build_grid(rgba[:, :, 3], step)
        self._build_arap()

        # 조종점: 추정한 위치에서 가장 가까운 그물 점을 조종점으로 쓴다
        guess = find_character_points(rgba[:, :, 3])
        self.handle_idx = []
        for name in HANDLES:
            d = np.linalg.norm(self.rest - guess[name], axis=1)
            d[self.handle_idx] = np.inf                     # 같은 점을 두 번 쓰지 않게
            self.handle_idx.append(int(d.argmin()))
        self.handle_idx = np.array(self.handle_idx)
        self.rest_pts = {n: self.rest[i].copy() for n, i in zip(HANDLES, self.handle_idx)}

        # 조종점을 붙잡는 조건을 넣은 연립방정식을 미리 풀어둔다(매 프레임 행렬곱만 하면 됨)
        A = self.L.copy()
        A[self.handle_idx, self.handle_idx] += HANDLE_WEIGHT
        self.A_inv = np.linalg.inv(A)
        self.cur = self.rest.copy()

    # --- 1. 그물 만들기: 그림이 있는 칸만 남기고, 칸마다 삼각형 2개 ---
    def _build_grid(self, alpha, step):
        mask = (alpha > 40).astype(np.uint8)
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))
        nx = int(math.ceil(self.w / step))
        ny = int(math.ceil(self.h / step))
        cells = np.zeros((ny, nx), np.uint8)
        for j in range(ny):
            for i in range(nx):
                if mask[j * step:(j + 1) * step, i * step:(i + 1) * step].any():
                    cells[j, i] = 1
        # 떨어져 있는 조각(점 찍힌 얼룩 등)은 버리고 가장 큰 한 덩어리만 사용
        count, labels, stats, _ = cv2.connectedComponentsWithStats(cells, connectivity=4)
        if count > 2:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cells = (labels == biggest).astype(np.uint8)

        index = {}
        verts, tris = [], []

        def vid(i, j):
            if (i, j) not in index:
                index[(i, j)] = len(verts)
                verts.append((i * step, j * step))
            return index[(i, j)]

        for j in range(ny):
            for i in range(nx):
                if cells[j, i]:
                    a, b = vid(i, j), vid(i + 1, j)
                    c, d = vid(i + 1, j + 1), vid(i, j + 1)
                    tris.append((a, b, c))
                    tris.append((a, c, d))
        self.rest = np.array(verts, dtype=np.float64)      # 원래(기본 자세) 점 위치
        self.tris = np.array(tris, dtype=np.int32)

    # --- 2. ARAP 준비: 점들 사이 연결(변) 목록과 라플라시안 행렬 ---
    def _build_arap(self):
        n = len(self.rest)
        t = self.tris
        pairs = np.vstack([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
        pairs = np.unique(np.sort(pairs, axis=1), axis=0)
        self.edges = pairs                                  # 화면에 그물 그릴 때 사용
        self.I = np.concatenate([pairs[:, 0], pairs[:, 1]])  # 양방향 변 i → j
        self.J = np.concatenate([pairs[:, 1], pairs[:, 0]])
        self.E = self.rest[self.I] - self.rest[self.J]      # 원래 모양에서의 변 벡터
        L = np.zeros((n, n))
        np.add.at(L, (self.I, self.I), 1.0)
        np.add.at(L, (self.I, self.J), -1.0)
        self.L = L
        self.n = n

    # --- 3. ARAP 풀기 ---
    def solve(self, targets, iters=ARAP_ITERS):
        """targets: 조종점 7개의 목표 위치(그림 기준 좌표). 변형된 모든 점 위치를 돌려준다.
        (1) 각 점 주변이 원래 모양에서 얼마나 '회전'했는지 구하고  [local 단계]
        (2) 그 회전을 유지하면서 조종점 조건을 만족하는 점 위치를 다시 구한다 [global 단계]
        이 두 단계를 몇 번 반복한다."""
        I, J, E, n = self.I, self.J, self.E, self.n
        P = self.cur
        for _ in range(iters):
            D = P[I] - P[J]
            dot = np.bincount(I, E[:, 0] * D[:, 0] + E[:, 1] * D[:, 1], n)
            cross = np.bincount(I, E[:, 0] * D[:, 1] - E[:, 1] * D[:, 0], n)
            theta = np.arctan2(cross, dot)                   # 점마다의 최적 회전 각도
            c, s = np.cos(theta), np.sin(theta)
            cc, ss = 0.5 * (c[I] + c[J]), 0.5 * (s[I] + s[J])
            B = np.empty((n, 2))
            B[:, 0] = np.bincount(I, cc * E[:, 0] - ss * E[:, 1], n)
            B[:, 1] = np.bincount(I, ss * E[:, 0] + cc * E[:, 1], n)
            B[self.handle_idx] += HANDLE_WEIGHT * targets
            P = self.A_inv @ B
        self.cur = P
        return P

    def reset(self):
        self.cur = self.rest.copy()

    # --- 4. 변형된 그물에 그림 입히기 ---
    def render(self, P):
        """변형된 점 위치 P대로 그림을 휘어서 pygame Surface로 돌려준다.
        삼각형마다 번호를 칠해 '이 픽셀은 몇 번 삼각형'인지 지도를 만든 뒤,
        픽셀마다 원본 그림의 어느 위치를 가져올지 한 번에 계산한다(cv2.remap)."""
        Q = P + self.pad                                   # 여백만큼 밀기
        tri_dst = Q[self.tris]                             # (T, 3, 2)
        tri_src = self.rest[self.tris]

        # 삼각형마다 "변형된 위치 → 원본 위치" 변환(2x2 행렬 + 기준점)
        d1, d2 = tri_dst[:, 1] - tri_dst[:, 0], tri_dst[:, 2] - tri_dst[:, 0]
        s1, s2 = tri_src[:, 1] - tri_src[:, 0], tri_src[:, 2] - tri_src[:, 0]
        det = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]
        ok = np.abs(det) > 1e-6
        det = np.where(ok, det, 1.0)
        m00 = (s1[:, 0] * d2[:, 1] - s2[:, 0] * d1[:, 1]) / det
        m01 = (-s1[:, 0] * d2[:, 0] + s2[:, 0] * d1[:, 0]) / det
        m10 = (s1[:, 1] * d2[:, 1] - s2[:, 1] * d1[:, 1]) / det
        m11 = (-s1[:, 1] * d2[:, 0] + s2[:, 1] * d1[:, 0]) / det
        pts = np.round(tri_dst * 16).astype(np.int32)      # shift=4 → 1/16 픽셀 정밀도

        def warp(order):
            idmap = np.zeros((self.canvas_h, self.canvas_w), np.uint16)
            for k in order:
                cv2.fillConvexPoly(idmap, pts[k], k + 1, lineType=cv2.LINE_8, shift=4)
            ys, xs = np.nonzero(idmap)
            k = idmap[ys, xs].astype(np.int64) - 1
            dx = xs - tri_dst[k, 0, 0]
            dy = ys - tri_dst[k, 0, 1]
            map_x = np.full(idmap.shape, -10.0, np.float32)
            map_y = np.full(idmap.shape, -10.0, np.float32)
            map_x[ys, xs] = np.where(ok[k], tri_src[k, 0, 0] + m00[k] * dx + m01[k] * dy, -10)
            map_y[ys, xs] = np.where(ok[k], tri_src[k, 0, 1] + m10[k] * dx + m11[k] * dy, -10)
            return cv2.remap(self.rgba, map_x, map_y, cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

        # 팔이 몸 위로 겹칠 때 팔 주변의 투명한 부분이 몸을 지워버리지 않도록,
        # 그리는 순서를 반대로 한 그림을 한 장 더 만들어 뒤에 받쳐 준다.
        n_tri = len(self.tris)
        front = warp(range(n_tri))
        back = warp(range(n_tri - 1, -1, -1))
        hole = front[:, :, 3] < back[:, :, 3]              # 앞 그림이 더 투명한 곳만 뒤 그림으로 채움
        front[hole] = back[hole]
        return pygame.image.frombuffer(front.tobytes(), (self.canvas_w, self.canvas_h), "RGBA")


# ------------------------------------------------------------------ 캠: 내 몸 위치 읽기 (별도 스레드)
def open_camera():
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


class BodyTracker:
    """캠 프레임에서 몸 위치를 계속 읽어 '가장 최근 결과'만 보관한다.
    게임 화면(60fps)이 캠 속도(20~30fps) 때문에 느려지지 않도록 따로 돌린다."""

    def __init__(self):
        self.ok = False
        self.error = ""
        self.cap = None
        self.landmarker = None
        self.lock = threading.Lock()
        self.pose = None          # {"time", "cx", "pts"}
        self.preview = None       # 미리보기용 RGB 바이트
        self.running = False
        self.thread = None
        if not MEDIAPIPE_OK:
            self.error = "mediapipe가 설치되어 있지 않습니다 (pip install mediapipe)"
            return
        try:
            if not os.path.exists(POSE_MODEL_PATH):
                print("자세 인식 모델 다운로드 중...")
                urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
            # 폴더 경로에 한글이 있어도 문제없도록 파일 내용을 직접 읽어서 넘긴다
            with open(POSE_MODEL_PATH, "rb") as f:
                model_bytes = f.read()

            def create(base_options):
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=1,
                    min_pose_detection_confidence=0.4,
                    min_pose_presence_confidence=0.4,
                    min_tracking_confidence=0.4,
                )
                return vision.PoseLandmarker.create_from_options(options)

            try:
                self.landmarker = create(mp_python.BaseOptions(model_asset_buffer=model_bytes))
            except Exception:      # 버전에 따라 위 방식이 안 되면 파일 경로로 다시 시도
                self.landmarker = create(mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH))
            self.cap = open_camera()
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
                result = self.landmarker.detect_for_video(image, ts)
                poses = list(result.pose_landmarks or [])
            except Exception as e:
                self.error = str(e)
                poses = []

            pose = None
            if poses:
                lm = poses[0]
                pts = np.array([[p.x * fw, p.y * fh,
                                 p.visibility if p.visibility is not None else 1.0]
                                for p in lm])
                a, b = pts[SIDE_A["sh"]], pts[SIDE_B["sh"]]
                if min(a[2], b[2]) >= 0.3:
                    pose = {"time": time.monotonic(),
                            "cx": float((a[0] + b[0]) / 2 / fw),   # 어깨 가운데의 가로 위치(0~1)
                            "pts": pts}
            small = cv2.resize(rgb, PREVIEW_SIZE)
            with self.lock:
                if pose is not None:
                    self.pose = pose
                self.preview = small.tobytes()

    def get(self):
        with self.lock:
            return self.pose, self.preview

    def close(self):
        self.running = False
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        if self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass


# ------------------------------------------------------------------ 내 동작 → 조종점 목표 위치
def pose_to_targets(pts, rest, extra_lean=0.0):
    """캠에서 읽은 몸 좌표(pts)를 캐릭터 조종점 7개의 목표 위치로 바꾼다.
    사람과 캐릭터는 비율이 달라서(머리가 크고 팔다리가 짧음) 좌표를 그대로 쓰지 않고
    '방향 + 뻗은 정도'만 가져와 캐릭터의 팔·다리 길이에 맞춰 적용한다.
    pts가 None이면 기본 자세(+이동 기울기)만 돌려준다."""
    hip = rest["hip"]
    # 캐릭터 쪽 기준점: 어깨(목에서 손 쪽으로 45% 지점), 골반(발 바로 위)
    sh_rest = {s: rest["neck"] + 0.45 * (rest["hand_" + s] - rest["neck"]) for s in "lr"}
    arm_len = {s: np.linalg.norm(rest["hand_" + s] - sh_rest[s]) for s in "lr"}

    lean, head_ang = extra_lean, extra_lean
    arms, legs = {}, {}
    if pts is not None:
        # 화면에서 왼쪽에 보이는 어깨 쪽을 캐릭터의 화면 왼쪽 팔에 연결(거울 방식)
        if pts[SIDE_A["sh"], 0] < pts[SIDE_B["sh"], 0]:
            sides = {"l": SIDE_A, "r": SIDE_B}
        else:
            sides = {"l": SIDE_B, "r": SIDE_A}
        sh_l, sh_r = pts[sides["l"]["sh"], :2], pts[sides["r"]["sh"], :2]
        mid_sh = (sh_l + sh_r) / 2
        sw = max(np.linalg.norm(sh_r - sh_l), 1.0)           # 어깨너비 = 몸 크기 기준

        hip_l, hip_r = pts[sides["l"]["hip"]], pts[sides["r"]["hip"]]
        hips_seen = min(hip_l[2], hip_r[2]) >= VIS
        if hips_seen:      # 전신이 보이면: 엉덩이→어깨 방향으로 상체 기울기
            up = mid_sh - (hip_l[:2] + hip_r[:2]) / 2
            lean += math.atan2(up[0], -up[1])
        else:              # 상반신만 보이면: 어깨선 기울기로 대신
            line = sh_r - sh_l
            lean += 1.5 * math.atan2(line[1], line[0])
        lean = clamp(lean, -MAX_LEAN, MAX_LEAN)

        if pts[NOSE, 2] >= VIS:
            hv = pts[NOSE, :2] - mid_sh
            head_ang = clamp(math.atan2(hv[0], -hv[1]) * 1.2, -MAX_LEAN * 1.3, MAX_LEAN * 1.3)
        else:
            head_ang = lean

        for s in "lr":
            sh, wr = pts[sides[s]["sh"]], pts[sides[s]["wr"]]
            if wr[2] >= VIS:
                r = (wr[:2] - sh[:2]) / (ARM_RATIO * sw)      # 팔 방향 + 뻗은 정도(0~1)
                # 손이 몸 반대편으로 넘어가면 그림이 꼬이므로 자기 쪽에 머물게 한다
                r[0] = min(r[0], 0.15) if s == "l" else max(r[0], -0.15)
                length = max(np.linalg.norm(r), 1e-6)
                r = r / length * clamp(length, 0.5, 1.1)      # 너무 짧게 접히거나 늘어나지 않게
                arms[s] = r
            hp, kn, an = (pts[sides[s][k]] for k in ("hip", "knee", "ank"))
            if hips_seen and an[2] >= VIS and kn[2] >= VIS:
                leg = np.linalg.norm(kn[:2] - hp[:2]) + np.linalg.norm(an[:2] - kn[:2])
                legs[s] = (an[:2] - hp[:2]) / max(leg, 1.0)

    targets = {"hip": hip.copy()}
    targets["neck"] = hip + rotate(rest["neck"] - hip, lean)
    targets["head"] = targets["neck"] + rotate(rest["head"] - rest["neck"], head_ang)
    for s in "lr":
        shoulder = hip + rotate(sh_rest[s] - hip, lean)
        if s in arms:
            targets["hand_" + s] = shoulder + arms[s] * arm_len[s]
        else:                                               # 손이 안 보이면 기본 자세(몸 기울기만 반영)
            targets["hand_" + s] = hip + rotate(rest["hand_" + s] - hip, lean)
        foot = rest["foot_" + s]
        if s in legs:
            leg_len = foot[1] - hip[1]
            targets["foot_" + s] = np.array([foot[0], hip[1]]) + legs[s] * leg_len
        else:
            targets["foot_" + s] = foot.copy()
    return np.array([targets[n] for n in HANDLES])


# ------------------------------------------------------------------ 게임
def load_font(size):
    path = "C:/Windows/Fonts/malgun.ttf"
    if os.path.exists(path):
        return pygame.font.Font(path, size)
    return pygame.font.SysFont("malgungothic,nanumgothic,applegothic", size)


def main():
    # 1. 초기화
    pygame.init()

    # 2. 화면 설정
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("공 피하기 게임 (캠 + ARAP)")
    clock = pygame.time.Clock()
    big_font = load_font(74)
    mid_font = load_font(36)
    small_font = load_font(18)

    # 캐릭터 그림 불러오기 → 키(PLAYER_HEIGHT)에 맞춰 비율 유지하며 크기 조절
    names = " 또는 ".join(CHARACTER_FILES)
    character_path = find_character_png()
    if character_path is None:
        print(f"{names} 를 찾지 못했습니다. 아래 폴더들을 찾아봤습니다:")
        for d in CANDIDATE_DIRS:
            exists = "폴더 있음" if os.path.isdir(d) else "폴더 없음"
            print(f"  - {d}  ({exists})")
        raise SystemExit(f"{names} 파일을 위 폴더 중 한 곳에 넣어주세요.")
    print("캐릭터 그림 사용:", character_path)
    image = pygame.image.load(character_path).convert_alpha()
    scale = PLAYER_HEIGHT / image.get_height()
    image = pygame.transform.smoothscale(
        image, (max(1, round(image.get_width() * scale)), PLAYER_HEIGHT))
    to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    rgba = np.frombuffer(to_bytes(image, "RGBA"), np.uint8).reshape(
        image.get_height(), image.get_width(), 4)
    rgba = remove_white_background(rgba)

    # 그림을 ARAP 메쉬로 만들기
    mesh = ArapMesh(rgba)
    rest = mesh.rest_pts
    player_w, player_h = mesh.w, mesh.h
    print(f"ARAP 메쉬: 점 {mesh.n}개, 삼각형 {len(mesh.tris)}개, 조종점 {len(HANDLES)}개")

    # 캠 시작
    tracker = BodyTracker()
    if tracker.ok:
        tracker.start()
        print("캠 인식 시작: 캠 앞에서 좌우로 움직여 보세요.")
    else:
        print("캠을 사용할 수 없어 키보드(← →)로 조작합니다:", tracker.error)

    # 플레이어: 화면 아래쪽 가운데에서 시작 (player_x = 그림 왼쪽 끝)
    player_x = (screen_width - player_w) / 2
    player_y = screen_height - player_h
    targets = pose_to_targets(None, rest)     # 조종점 현재 목표(부드럽게 따라가도록 보관)
    move_lean = 0.0

    # 공 설정
    ball_rect = pygame.Rect(random.randint(0, screen_width - ball_size), -ball_size,
                            ball_size, ball_size)
    ball_surface = pygame.Surface((ball_size, ball_size), pygame.SRCALPHA)
    pygame.draw.ellipse(ball_surface, red, ball_surface.get_rect())
    ball_mask = pygame.mask.from_surface(ball_surface)

    running = True
    game_over = False
    cleared = False                    # True면 클리어, False면 게임 오버
    end_time = None                    # 게임이 끝난 시각
    dodged = 0                         # 피한 공 개수
    ball_speed_now = START_SPEED       # 지금 공 속도
    # 공 위치는 소수점까지 따로 보관한다. (ball_rect.y는 정수만 저장해서
    # 속도 3.5로 움직여도 매번 0.5가 버려지고 3으로 움직이기 때문)
    ball_y = float(-ball_size)
    show_mesh = False
    show_camera = True

    # 3. 게임 루프
    while running:
        # 3-1. 이벤트 처리
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
                elif event.key == pygame.K_r:               # 다시 시작
                    game_over = False
                    cleared = False
                    end_time = None
                    dodged = 0
                    ball_speed_now = START_SPEED
                    ball_rect.x = random.randint(0, screen_width - ball_size)
                    ball_y = float(-ball_size)
                    ball_rect.y = int(ball_y)

        # 끝나고 END_WAIT초가 지나면 자동 종료 (scan_and_play로 실행했다면 스캐너로 돌아감)
        if (END_WAIT > 0 and game_over and end_time is not None
                and time.monotonic() - end_time > END_WAIT):
            running = False

        pose, preview = tracker.get()
        tracking = pose is not None and time.monotonic() - pose["time"] < LOST_TIME

        if not game_over:
            # 3-2. 게임 상태 업데이트
            old_x = player_x
            if tracking:
                # 캠 속 내 가로 위치(0~1) → 화면 가로 위치. 양끝 여백을 빼서 조금만 움직여도 끝까지 가게 함
                t = (pose["cx"] - CAM_MARGIN) / (1 - 2 * CAM_MARGIN)
                goal_x = clamp(t, 0.0, 1.0) * (screen_width - player_w)
                player_x += (goal_x - player_x) * FOLLOW_SMOOTH
            else:
                keys = pygame.key.get_pressed()
                if keys[pygame.K_LEFT]:
                    player_x -= player_speed
                if keys[pygame.K_RIGHT]:
                    player_x += player_speed
            player_x = clamp(player_x, 0, screen_width - player_w)

            # 움직이는 방향으로 살짝 기울기(달리는 느낌)
            want = clamp((player_x - old_x) * MOVE_LEAN, -0.25, 0.25)
            move_lean += (want - move_lean) * 0.2

            # 내 동작 → 조종점 목표 → ARAP으로 그물 전체 변형
            goal = pose_to_targets(pose["pts"] if tracking else None, rest, move_lean)
            targets += (goal - targets) * POSE_SMOOTH
            points = mesh.solve(targets)
            player_surface = mesh.render(points)
            draw_pos = (int(player_x) - mesh.pad, int(player_y) - mesh.pad)

            # 공 이동: 바닥까지 내려가면 '하나 피함' → 점수 +1, 다음 공은 더 빠르게
            ball_y += ball_speed_now
            ball_rect.y = int(ball_y)
            if ball_rect.y > screen_height:
                dodged += 1
                ball_speed_now = min(MAX_SPEED, START_SPEED + dodged * SPEED_UP)
                if dodged >= GOAL:                          # 목표 개수 달성 → 클리어
                    game_over = True
                    cleared = True
                ball_rect.x = random.randint(0, screen_width - ball_size)
                ball_y = float(-ball_size)
                ball_rect.y = int(ball_y)

            # 충돌 감지: 사각형이 아니라 실제로 휘어진 그림 모양과 공이 겹치는지 검사
            player_mask = pygame.mask.from_surface(player_surface, 100)
            offset = (ball_rect.x - draw_pos[0], ball_rect.y - draw_pos[1])
            if player_mask.overlap(ball_mask, offset):
                game_over = True

            if game_over:                                   # 이번 프레임에 끝났으면 시각 기록
                end_time = time.monotonic()

        # 3-3. 화면 그리기
        screen.fill(black)

        if not game_over:
            screen.blit(player_surface, draw_pos)
            screen.blit(ball_surface, ball_rect)
            # 피한 개수 / 목표 개수
            screen.blit(mid_font.render(f"{dodged} / {GOAL}", True, white), (10, 32))
            if show_mesh:      # D 키: ARAP 그물과 조종점 표시
                origin = np.array([int(player_x), int(player_y)])
                screen_pts = points + origin
                for a, b in mesh.edges:
                    pygame.draw.line(screen, (0, 220, 255), screen_pts[a], screen_pts[b], 1)
                for name, p in zip(HANDLES, targets):
                    pygame.draw.circle(screen, (255, 230, 0), (p + origin).astype(int), 5)
        else:
            message = "클리어!" if cleared else "게임 오버!"
            text = big_font.render(message, True, white)
            text_rect = text.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(text, text_rect)
            result = small_font.render(f"피한 공: {dodged} / {GOAL}", True, white)
            screen.blit(result, result.get_rect(center=(screen_width // 2, screen_height // 2 + 50)))
            if END_WAIT > 0 and end_time is not None:
                left = max(0, END_WAIT - (time.monotonic() - end_time))
                hint_text = f"R 키를 누르면 다시 시작  ({left:.0f}초 뒤 종료)"
            else:
                hint_text = "R 키를 누르면 다시 시작"
            hint = small_font.render(hint_text, True, white)
            screen.blit(hint, hint.get_rect(center=(screen_width // 2, screen_height // 2 + 80)))

        # 캠 미리보기(오른쪽 위) + 내 위치 표시선
        if show_camera and preview is not None:
            cam = pygame.image.frombuffer(preview, PREVIEW_SIZE, "RGB")
            cam_x = screen_width - PREVIEW_SIZE[0] - 10
            screen.blit(cam, (cam_x, 10))
            pygame.draw.rect(screen, white, (cam_x, 10, *PREVIEW_SIZE), 1)
            for m in (CAM_MARGIN, 1 - CAM_MARGIN):          # 이 두 선 사이에서 움직이면 됨
                x = cam_x + int(m * PREVIEW_SIZE[0])
                pygame.draw.line(screen, (120, 120, 120), (x, 10), (x, 10 + PREVIEW_SIZE[1]), 1)
            if tracking:
                x = cam_x + int(pose["cx"] * PREVIEW_SIZE[0])
                pygame.draw.line(screen, (0, 255, 0), (x, 10), (x, 10 + PREVIEW_SIZE[1]), 2)

        if tracking:
            status = "캠 조작 중"
        elif tracker.ok:
            status = "몸이 안 보여요 - 키보드(← →) 조작 중"
        else:
            status = "캠 없음 - 키보드(← →) 조작 중"
        screen.blit(small_font.render(status + "   |   R 다시 시작  D 그물 보기  V 캠 화면  ESC 종료",
                                      True, (170, 170, 170)), (10, 8))

        pygame.display.flip()
        clock.tick(60)  # 초당 60프레임으로 제한

    # 4. 종료
    tracker.close()
    pygame.quit()


if __name__ == "__main__":
    main()