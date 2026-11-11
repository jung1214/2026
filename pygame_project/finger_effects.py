"""
character_magic.py  -  내가 그린 캐릭터가 손 모양에 따라 특수효과를 내는 프로그램 (게임 없음)

설치:  pip install opencv-python mediapipe numpy pillow
실행:  python character_magic.py
준비:  character 폴더에 full.png (내가 그린 캐릭터, 배경 투명 PNG 권장)

[효과]
  검지만 펴기       검지가 가리키는 방향으로 캐릭터가 빛 구슬 발사
  손바닥 활짝       캐릭터 양손에서 폭죽이 펑 터지고 바로 사라짐
  검지+중지 모으기   캐릭터 손에서 빛이 나와 모은 손가락을 따라다님

[키보드]  M 캐릭터가 손을 따라다님 | T 효과 색 바꾸기 | K 뼈대 칸(우측 상단) | B 배경 | F 전체화면 | H 도움말 | C 지우기 | S 스크린샷 | ESC 종료
"""
import math
import os
import random
import sys
import time
import urllib.request

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/"
                  "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")

# character 폴더 이름이 오타여도(charactor/charater) 찾을 수 있게 후보를 여러 개 둔다
CHAR_DIRS = [os.path.join(BASE_DIR, n) for n in ("character", "charactor", "charater")] + [BASE_DIR]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]

# 색 테마: (이름, 색상 hue 0~180) -- 효과 색은 한 가지로 고정, T 키를 눌러야만 바뀐다
THEMES = [
    ("노랑", 25),
    ("하늘", 95),
    ("보라", 140),
    ("초록", 60),
    ("분홍", 165),
    ("빨강", 0),
]

THEME_EN = ["yellow", "sky", "purple", "green", "pink", "red"]   # 화면 하단 표시용(영문)

HELP_KO = [
    "내 캐릭터 특수효과",
    "검지만 펴기 : 가리키는 방향으로 빛 구슬 발사",
    "손바닥 활짝 : 캐릭터 양손에서 폭죽",
    "브이 (검지+중지 벌림) : 캐릭터 주위에 별이 반짝",
    "엄지 척 (엄지만 위로) : 캐릭터가 분홍 하트를 띄움",
    "엄지+검지 집게 : 빛 구슬을 잡아서 옮기고, 벌리면 내려놓기",
    "두 손 모으기 : 캐릭터 머리 위에 무지개",
    "주먹 : 모든 효과 지우기",
    "검지+중지 모으기 : 캐릭터 손에서 나온 빛이 손가락을 따라감",
    "M 캐릭터 따라다니기  T 색 바꾸기  K 뼈대  B 배경  F 전체화면",
    "C 지우기  S 저장  H 도움말  ESC 종료",
]
HELP_EN = [
    "My character magic",
    "Index finger only : shoot a light orb",
    "Open palm : fireworks from both hands",
    "V sign : twinkling stars around the character",
    "Thumb up : character floats pink hearts",
    "Pinch (thumb+index) : pick up a light orb, move it, release to place it",
    "Both hands together : a rainbow above the character",
    "Fist : clear all effects",
    "Index+middle together : light from character follows fingers",
    "M follow hand  T color theme  K skeleton  B bg  F fullscreen",
    "C clear  S shot  H help  ESC quit",
]


# ---------------------------------------------------------------- 색/도형 도구
def hue_bgr(h, v=255, s=255):
    px = np.uint8([[[int(h) % 180, s, v]]])
    b, g, r = cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def scale_color(c, k):
    k = max(0.0, min(1.0, k))
    return int(c[0] * k), int(c[1] * k), int(c[2] * k)


# ---------------------------------------------------------------- 손 모양 판정
def finger_states(pts):
    """[엄지, 검지, 중지, 약지, 소지]가 펴져 있는지 (손목 기준 거리 비교)"""
    wrist = pts[0]
    ext = [np.linalg.norm(pts[4] - pts[17]) > np.linalg.norm(pts[3] - pts[17]) * 1.05]
    for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18)):
        ext.append(np.linalg.norm(pts[tip] - wrist) > np.linalg.norm(pts[pip] - wrist) * 1.08)
    return ext


def classify(pts, was_pinch=False):
    """PINCH(집게) / PALM / FIST / THUMB(엄지 척) / PEACE / TWO(검지+중지 모음) / POINT / OTHER"""
    palm0 = max(1.0, float(np.linalg.norm(pts[0] - pts[9])))
    near = float(np.linalg.norm(pts[4] - pts[8])) / palm0
    reach = float(np.linalg.norm(pts[8] - pts[0])) / palm0
    if near < (0.55 if was_pinch else 0.35) and reach > 1.0:
        return "PINCH"          # 엄지와 검지 끝이 맞닿음 (주먹과는 검지가 멀리 뻗어 있는지로 구분)
    _, idx, mid, ring, pnk = finger_states(pts)
    n = sum([idx, mid, ring, pnk])
    if n == 0:
        thumb, base = pts[4], pts[2]
        dx, dy = abs(thumb[0] - base[0]), base[1] - thumb[1]
        palm = max(1.0, float(np.linalg.norm(pts[0] - pts[9])))
        if dy > palm * 0.6 and dy > dx * 1.2 and thumb[1] < pts[5][1] and thumb[1] < pts[3][1]:
            return "THUMB"
        return "FIST"
    if n >= 4:
        return "PALM"
    if idx and not (mid or ring or pnk):
        return "POINT"
    if idx and mid and not (ring or pnk):
        palm = max(1.0, float(np.linalg.norm(pts[0] - pts[9])))
        if np.linalg.norm(pts[8] - pts[12]) / palm < 0.40:
            return "TWO"        # 검지+중지를 모음
        return "PEACE"          # 브이(벌림)
    return "OTHER"


# ---------------------------------------------------------------- 캐릭터 그림 불러오기
def read_image_unicode(path):
    """한글 경로에서도 읽히도록 cv2.imread 대신 fromfile + imdecode 사용"""
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_UNCHANGED)


def find_character_file():
    for d in CHAR_DIRS:
        if not os.path.isdir(d):
            continue
        for fname in sorted(os.listdir(d)):
            stem, dot, ext = fname.rpartition(".")
            if dot and stem.lower() == "full" and ext.lower() in ("png", "jpg", "jpeg"):
                return os.path.join(d, fname)
    return None


def to_bgra(img):
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 3:
        # 투명 배경이 아니면 흰 배경을 투명하게 처리
        white = np.all(img > 235, axis=2)
        alpha = np.where(white, 0, 255).astype(np.uint8)
        alpha = cv2.GaussianBlur(alpha, (0, 0), 1.0)
        img = np.dstack([img, alpha])
    return img


def placeholder_character():
    """full.png가 없을 때 보여줄 간단한 임시 캐릭터"""
    im = np.zeros((400, 240, 4), np.uint8)
    col = (255, 190, 120, 255)
    cv2.circle(im, (120, 90), 60, col, -1, cv2.LINE_AA)
    cv2.rectangle(im, (70, 150), (170, 300), col, -1, cv2.LINE_AA)
    cv2.rectangle(im, (75, 300), (115, 390), col, -1, cv2.LINE_AA)
    cv2.rectangle(im, (125, 300), (165, 390), col, -1, cv2.LINE_AA)
    cv2.circle(im, (98, 85), 8, (40, 40, 40, 255), -1, cv2.LINE_AA)
    cv2.circle(im, (142, 85), 8, (40, 40, 40, 255), -1, cv2.LINE_AA)
    return im


def blit_clipped(dst, src, x, y):
    """BGRA 이미지를 화면 밖으로 나가는 부분은 잘라서 알파 합성"""
    h, w = src.shape[:2]
    H, W = dst.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    s = src[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = dst[y0:y1, x0:x1].astype(np.float32)
    dst[y0:y1, x0:x1] = (roi * (1 - a) + s[:, :, :3].astype(np.float32) * a).astype(np.uint8)


class Character:
    def __init__(self, frame_w, frame_h):
        path = find_character_file()
        img = None
        if path:
            try:
                img = to_bgra(read_image_unicode(path))
            except Exception as e:
                print("캐릭터 그림을 읽지 못했습니다:", e)
        if img is None:
            print("★ character 폴더에서 full.png를 찾지 못해 임시 캐릭터를 씁니다.")
            print("  찾아본 곳:", ", ".join(CHAR_DIRS))
            img = placeholder_character()
        else:
            print("캐릭터 그림 사용:", path)
        ys, xs = np.where(img[:, :, 3] > 10)
        if len(ys):
            img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        target_h = int(frame_h * 0.42)
        scale = target_h / img.shape[0]
        self.img = cv2.resize(img, (max(2, int(img.shape[1] * scale)), target_h), interpolation=cv2.INTER_AREA)
        self.h, self.w = self.img.shape[:2]
        self.cx = frame_w // 2
        self.bottom = frame_h - int(frame_h * 0.04)
        self.fw, self.fh = frame_w, frame_h

    def hand_points(self):
        """캐릭터가 팔을 벌린 그림이라고 보고, 양쪽 손끝 위치(화면 좌표)를 근사한다"""
        top = self.bottom - self.h
        y = top + 0.58 * self.h
        return [np.array([self.cx - self.w * 0.47, y], np.float32),
                np.array([self.cx + self.w * 0.47, y], np.float32)]

    def chest(self):
        return np.array([self.cx, self.bottom - self.h * 0.55], np.float32)

    def draw(self, canvas, bob=0.0, pulse=0.0, shake=0.0):
        sy = 1.0 + pulse * 0.07
        sx = 1.0 - pulse * 0.035
        w, h = max(2, int(self.w * sx)), max(2, int(self.h * sy))
        im = self.img if (w, h) == (self.w, self.h) else cv2.resize(self.img, (w, h), interpolation=cv2.INTER_LINEAR)
        x = int(self.cx - w / 2 + (random.uniform(-shake, shake) if shake else 0))
        y = int(self.bottom - h - bob)
        blit_clipped(canvas, im, x, y)


# ---------------------------------------------------------------- 특수효과 엔진
class Magic:
    def __init__(self, w, h, character):
        self.w, self.h = w, h
        self.ch = character
        self.layer = np.zeros((h, w, 3), np.uint8)
        self.parts = []    # [x, y, vx, vy, life, maxlife, color, size, gravity, drag]
        self.rings = []    # [x, y, r, dr, maxr, color]
        self.shots = []    # [x, y, vx, vy, color]
        self.t = 0.0
        self.gesture = "NONE"
        self._cand, self._cand_n = "NONE", 0
        self.fist_frames = 0
        self.cooldown = 0
        self.pulse = 0.0
        self.shake = 0.0
        self.pts_cache = None
        self.theme = 0
        self.follow = True
        self.trail = np.zeros((h, w, 3), np.uint8)   # 손가락을 따라가는 빛의 꼬리(천천히 사라짐)
        self.trail_tip = None
        self.trail_side = 0
        self.path = []          # 검지+중지 빛이 지나온 길 (부드러운 선으로 다시 그림)
        self.path_anchored = True
        self.big_age = 0        # 두 손 무지개가 나타난 뒤 지난 프레임 수
        self.hearts = []   # [x, y, vx, vy, life, maxlife, size, phase]
        self.stars = []    # [x, y, vx, vy, life, maxlife, size]
        self.orbs = []     # 내려놓은 빛 구슬 [x, y, 나이, 사라지는 중 남은 프레임(None이면 유지)]
        self.holding = False
        self.hold_pos = None
        self.quiet = False   # 집게를 벌려 구슬을 놓을 때 폭죽이 같이 터지지 않게
        self.sparks = []   # 폭죽 불꽃 줄기 [x, y, vx, vy, life, maxlife, color, width]
        self.hlayer = np.zeros((h, w, 3), np.uint8)   # 하트 전용(잔상 없음)

    def main_col(self, off=0.0):
        """현재 테마의 고정 색 (시간이 지나도 바뀌지 않음)"""
        return hue_bgr(THEMES[self.theme][1])

    def rand_col(self):
        return self.main_col()

    def clear(self):
        self.layer[:] = 0
        self.trail[:] = 0
        self.hlayer[:] = 0
        self.trail_tip = None
        self.path = []
        self.big_age = 0
        self.parts.clear()
        self.rings.clear()
        self.shots.clear()
        self.hearts.clear()
        self.stars.clear()
        self.sparks.clear()
        self.orbs.clear()
        self.holding = False
        self.hold_pos = None

    def emit(self, x, y, n, speed=(2, 6), color=None, size=(2, 5), life=(14, 30),
             gravity=0.0, drag=0.96, angle=None, spread=math.tau):
        for _ in range(n):
            a = (angle if angle is not None else random.uniform(0, math.tau)) + random.uniform(-spread / 2, spread / 2)
            sp = random.uniform(*speed)
            lf = random.uniform(*life)
            col = color if color is not None else self.rand_col()
            self.parts.append([x, y, math.cos(a) * sp, math.sin(a) * sp, lf, lf, col,
                               random.uniform(*size), gravity, drag])

    def orb(self, x, y, r, color, core=True):
        x, y = int(x), int(y)
        cv2.circle(self.layer, (x, y), int(r * 1.6), scale_color(color, 0.35), -1, cv2.LINE_AA)
        cv2.circle(self.layer, (x, y), int(r), color, -1, cv2.LINE_AA)
        if core:
            cv2.circle(self.layer, (x, y), max(1, int(r * 0.45)), (255, 255, 255), -1, cv2.LINE_AA)

    def lightning(self, p1, p2, color=(255, 200, 120), depth=5):
        a0, b0 = np.array(p1, float), np.array(p2, float)
        d = np.linalg.norm(b0 - a0)
        if d < 4:
            return
        pts = [a0, b0]
        disp = d * 0.16
        for _ in range(depth):
            new = []
            for a, b in zip(pts[:-1], pts[1:]):
                mid = (a + b) / 2
                n = np.array([-(b - a)[1], (b - a)[0]])
                nn = np.linalg.norm(n)
                if nn > 0:
                    mid = mid + n / nn * random.uniform(-disp, disp)
                new += [a, mid]
            new.append(pts[-1])
            pts = new
            disp *= 0.55
        poly = np.array(pts, np.int32).reshape(-1, 1, 2)
        cv2.polylines(self.layer, [poly], False, color, 5, cv2.LINE_AA)
        cv2.polylines(self.layer, [poly], False, (255, 255, 255), 2, cv2.LINE_AA)

    def heart(self, x, y, size, color):
        t = np.linspace(0, math.tau, 40)
        hx = 16 * np.sin(t) ** 3
        hy = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
        poly = np.stack([x + hx * size / 17, y + hy * size / 17], 1).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(self.hlayer, [poly], color, cv2.LINE_AA)

    def star(self, x, y, size, color):
        a = size
        b = size * 0.28
        pts = [(0, -a), (b, -b), (a, 0), (b, b), (0, a), (-b, b), (-a, 0), (-b, -b)]
        poly = np.array([(x + px, y + py) for px, py in pts], np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(self.layer, [poly], color, cv2.LINE_AA)

    def firework(self, pt):
        """진짜 폭죽처럼: 한 점에서 불꽃 줄기가 사방으로 쭉 뻗고, 끝이 아래로 휘며 바로 사라짐 (원/구슬 없음)"""
        x, y = float(pt[0]), float(pt[1])
        col = self.main_col()
        sc = self.h / 540.0
        n = 26
        base = random.uniform(0, math.tau)
        for i in range(n):
            a = base + i * math.tau / n + random.uniform(-0.05, 0.05)
            sp = (6.0 if i % 2 else 8.5) * random.uniform(0.92, 1.08) * sc
            lf = random.randint(13, 17)
            self.sparks.append([x, y, math.cos(a) * sp, math.sin(a) * sp, lf, lf, col, 2.5])
        for i in range(14):
            a = random.uniform(0, math.tau)
            sp = random.uniform(3, 7) * sc
            self.parts.append([x, y, math.cos(a) * sp, math.sin(a) * sp, 14, 14, col, 2.0, 0.25 * sc, 0.9])

    ORB_MAX = 6

    def _hold(self, pts):
        """집게: 빛 구슬을 잡고 손가락 사이를 따라다님"""
        target = (pts[4] + pts[8]) / 2
        if not self.holding:
            palm = max(1.0, float(np.linalg.norm(pts[0] - pts[9])))
            near = None
            for o in self.orbs:
                if o[3] is None and math.hypot(o[0] - target[0], o[1] - target[1]) < palm * 0.9:
                    if near is None or math.hypot(o[0] - target[0], o[1] - target[1]) < math.hypot(near[0] - target[0], near[1] - target[1]):
                        near = o
            if near is not None:
                self.orbs.remove(near)
                self.hold_pos = np.array([near[0], near[1]], np.float32)
            else:
                hands = self.ch.hand_points()
                side = 0 if np.linalg.norm(hands[0] - target) <= np.linalg.norm(hands[1] - target) else 1
                self.hold_pos = hands[side].copy()
            self.holding = True
            self.pulse = max(self.pulse, 0.4)
        self.hold_pos = self.hold_pos + (target - self.hold_pos) * 0.35

    def _drop(self):
        """집게를 벌리면 그 자리에 구슬을 내려놓음 (작은 링 하나로 알려줌)"""
        self.holding = False
        self.quiet = True
        if self.hold_pos is None:
            return
        x, y = float(self.hold_pos[0]), float(self.hold_pos[1])
        self.orbs.append([x, y, 0, None])
        self.rings.append([x, y, 8, 5, 38, scale_color(self.main_col(), 0.6)])
        live = [o for o in self.orbs if o[3] is None]
        if len(live) > self.ORB_MAX:
            live[0][3] = 14
        self.hold_pos = None

    def _draw_orbs(self):
        """내려놓은 구슬(느리게 숨쉬듯 반짝)과 구슬 사이의 아주 옅은 선(별자리)"""
        col = self.main_col()
        live = [o for o in self.orbs if o[3] is None]
        for a, b in zip(live[:-1], live[1:]):
            cv2.line(self.hlayer, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), scale_color(col, 0.22), 2, cv2.LINE_AA)
        keep = []
        for i, o in enumerate(self.orbs):
            o[2] += 1
            k = 1.0
            if o[3] is not None:
                o[3] -= 1
                if o[3] <= 0:
                    continue
                k = o[3] / 14.0
            breathe = 0.85 + 0.15 * math.sin(self.t * 2.0 + i)
            self._soft_orb(o[0], o[1], 12, scale_color(col, 0.75 * breathe * k))
            keep.append(o)
        self.orbs = keep
        if self.holding and self.hold_pos is not None:
            self._soft_orb(self.hold_pos[0], self.hold_pos[1], 15, scale_color(col, 0.9))

    def _soft_orb(self, x, y, r, color):
        x, y = int(x), int(y)
        cv2.circle(self.hlayer, (x, y), int(r * 1.5), scale_color(color, 0.3), -1, cv2.LINE_AA)
        cv2.circle(self.hlayer, (x, y), int(r), color, -1, cv2.LINE_AA)

    def _draw_rainbow(self):
        """두 손을 모으면 캐릭터 머리 위에 연하고 자연스러운 무지개: 색이 끊기지 않고 이어지며
        가장자리는 번지듯 흐려지고, 왼쪽에서 오른쪽으로 부드럽게 펼쳐진다."""
        ch = self.ch
        t = min(1.0, self.big_age / 18.0)
        sweep = 1 - (1 - t) ** 3
        cx, cy = int(ch.cx), int(ch.bottom - 0.9 * ch.h)
        R = 0.64 * ch.h
        band = 0.30 * ch.h                                         # 무지개 전체 두께
        pad = int(0.1 * ch.h)
        x0, x1 = max(0, cx - int(R) - pad), min(self.w, cx + int(R) + pad)
        y0, y1 = max(0, cy - int(R) - pad), min(self.h, cy + pad)
        if x1 <= x0 or y1 <= y0:
            return
        roi = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        c = (cx - x0, cy - y0)
        N = 40
        shim = 1.0 + 0.04 * math.sin(self.big_age / 30.0 * math.tau)
        for i in range(N):
            u = (i + 0.5) / N                                      # 0 바깥(빨강) -> 1 안쪽(보라)
            r = int(R - u * band)
            hue = 140 * u ** 1.2
            wgt = min(1.0, 0.25 + 3.0 * min(u, 1 - u))                     # 가장자리로 갈수록 투명하게
            col = scale_color(hue_bgr(hue, v=255, s=185), 0.33 * wgt * shim)
            cv2.ellipse(roi, c, (r, r), 0, 180, 180 + 180 * sweep, col,
                        max(2, int(band / N * 2.2)), cv2.LINE_AA)
        k = max(3, int(0.03 * ch.h)) | 1
        roi = cv2.GaussianBlur(roi, (k, k), 0)
        sl = self.hlayer[y0:y1, x0:x1]
        self.hlayer[y0:y1, x0:x1] = cv2.add(sl, roi)

    PATH_LEN = 0.4     # 선 최대 길이 (화면 가로 대비)

    def _trim_path(self):
        pts = np.array(self.path)
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        total = float(seg.sum())
        limit = self.PATH_LEN * self.w
        k = 0
        while total > limit and k < len(seg) - 1:
            total -= float(seg[k])
            k += 1
        if k:
            del self.path[:k]
            self.path_anchored = False

    def _draw_ribbon(self):
        """지나온 길을 곡선으로 부드럽게 다듬어(Chaikin) 2px 간격으로 촘촘히 그림 -> 꺾임/끊김 없는 한 줄.
        꼬리는 가늘고 어둡게, 머리는 굵고 밝게."""
        if len(self.path) < 2 or self.trail_tip is None:
            return
        col = self.main_col()
        pts = np.array(self.path, np.float32)
        pts = np.vstack([pts, self.trail_tip[None, :]])
        for _ in range(3):
            if len(pts) < 3:
                break
            q = pts[:-1] * 0.75 + pts[1:] * 0.25
            r = pts[:-1] * 0.25 + pts[1:] * 0.75
            pts = np.vstack([pts[:1], np.stack([q, r], 1).reshape(-1, 2), pts[-1:]])
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        dist = np.concatenate([[0.0], np.cumsum(seg)])
        if dist[-1] < 2:
            return
        n = max(2, int(dist[-1] / 2))
        u = np.linspace(0, dist[-1], n)
        xs, ys = np.interp(u, dist, pts[:, 0]), np.interp(u, dist, pts[:, 1])
        sc = self.h / 720.0
        for i in range(n):
            t = i / (n - 1)
            r = (1.5 + 5.5 * t ** 1.3) * sc
            cv2.circle(self.hlayer, (int(xs[i]), int(ys[i])), max(1, int(round(r))),
                       scale_color(col, 0.12 + 0.55 * t), -1, cv2.LINE_AA)
        hx, hy = self.trail_tip
        self._soft_orb(hx, hy, 10 * sc, scale_color(col, 0.75))
        if self.path_anchored:        # 시작점(캐릭터 손)에 작은 불빛: 어디서 나오는지 분명하게
            sx, sy = self.path[0]
            self._soft_orb(sx, sy, 7 * sc, scale_color(col, 0.55))

    def _cut(self):
        """손 모양이 바뀌면 이전 효과를 바로 지운다 (한 번에 한 가지 효과만 보이도록).
        내려놓은 구슬(별자리)만은 집게를 풀고 다시 잡는 동안 남아 있어야 해서 따로 다룬다."""
        self.parts.clear()
        self.hearts.clear()
        self.stars.clear()
        self.sparks.clear()
        self.rings.clear()
        self.shots.clear()
        self.trail[:] = 0
        self.layer[:] = 0
        self.trail_tip = None
        self.path = []

    def _clear_orbs(self):
        self.orbs.clear()
        self.holding = False
        self.hold_pos = None

    # ------------------------------------------------------------ 한 프레임 갱신
    def hands_together(self, a, b, was=False):
        """두 손이 가까이 모였는지: 두 손바닥 중심 거리 / 손바닥 크기 (한 번 모이면 조금 더 벌어져도 유지)"""
        idx = [0, 5, 9, 13, 17]
        ca, cb = a[idx].mean(0), b[idx].mean(0)
        palm = max(1.0, (float(np.linalg.norm(a[0] - a[9])) + float(np.linalg.norm(b[0] - b[9]))) / 2)
        return float(np.linalg.norm(ca - cb)) / palm < (2.3 if was else 1.7)

    def update(self, pts, pts2=None):
        """pts: 손 한 개의 (21,2) 픽셀 좌표 배열(없으면 None), pts2: 두 번째 손(없으면 None)"""
        self.t += 1 / 30
        t = self.t
        self.layer = (self.layer * 0.86).astype(np.uint8)
        self.trail = (self.trail * 0.91).astype(np.uint8)
        self.hlayer[:] = 0
        self.pts_cache = pts
        ch = self.ch
        if self.follow and pts is not None:
            target = float(pts[[0, 5, 9, 13, 17], 0].mean())
            if pts2 is not None and self.gesture == "HEART2":
                target = (target + float(pts2[[0, 5, 9, 13, 17], 0].mean())) / 2
            target = min(max(target, ch.w * 0.5), self.w - ch.w * 0.5)
            ch.cx += (target - ch.cx) * 0.22
        chest = ch.chest()
        self.pulse *= 0.88
        self.shake *= 0.85

        # 손 모양은 3프레임 연속 같아야 바뀐 것으로 본다 (깜빡임 방지)
        was_pinch = self.gesture == "PINCH"
        together = pts is not None and pts2 is not None and self.hands_together(pts, pts2, self.gesture == "HEART2")
        if together:
            raw = "HEART2"
        else:
            raw = classify(pts, was_pinch) if pts is not None else "NONE"
        if raw == self._cand:
            self._cand_n += 1
        else:
            self._cand, self._cand_n = raw, 1
        prev = self.gesture
        if self._cand_n >= 3 and raw != self.gesture:
            self.gesture = raw
            self.cooldown = 0
            self.quiet = False
            self._cut()
            if raw == "FIST":
                self.clear()           # 주먹 -> 모든 효과(구슬, 무지개 포함)를 지움
                self.pulse = 0.0
            if raw in ("POINT", "TWO", "PEACE", "THUMB", "HEART2"):
                self._clear_orbs()     # 다른 효과가 시작되면 놓아둔 구슬도 지움
        g = self.gesture if raw == self.gesture else "HOLD"
        if self.holding and self.gesture != "PINCH":
            self._drop()

        col = self.main_col()
        if self.cooldown > 0:
            self.cooldown -= 1

        if g == "PALM":
            if self.cooldown == 0 and not self.quiet:
                self._clear_orbs()
                for pt in ch.hand_points():
                    self.firework(pt)
                self.cooldown = 20
                self.pulse = 0.5

        elif g == "TWO" and pts is not None:
            # 빛은 캐릭터 손에서 출발해 모은 손가락 끝(목표)을 부드럽게 쫓아가고, 지나온 길이 한 줄의 곡선으로 이어진다
            target = (pts[8] + pts[12]) / 2
            hands = ch.hand_points()
            prev_tip = self.trail_tip
            if prev_tip is None:
                self.trail_side = 0 if np.linalg.norm(hands[0] - target) <= np.linalg.norm(hands[1] - target) else 1
                tip = hands[self.trail_side].copy()
                self.path = [tip.copy()]
                self.path_anchored = True
                self.pulse = max(self.pulse, 0.5)
            else:
                tip = prev_tip + (target - prev_tip) * 0.3
                if float(np.linalg.norm(tip - self.path[-1])) > 1.5:
                    self.path.append(tip.copy())
                elif len(self.path) > 2 and int(self.t * 30) % 3 == 0:
                    self.path.pop(0)         # 멈추면 꼬리가 천천히 머리 쪽으로 줄어듦
                    self.path_anchored = False
                self._trim_path()
            if self.path_anchored:
                self.path[0] = hands[self.trail_side].copy()   # 길이가 짧은 동안은 선의 시작이 항상 캐릭터 손
            self.trail_tip = tip
            self._draw_ribbon()

        elif g == "PEACE":
            if self.cooldown == 0:
                sc = self.h / 720.0
                top = ch.bottom - ch.h
                for _ in range(2):
                    a = random.uniform(0, math.tau)
                    rx, ry = ch.w * random.uniform(0.35, 0.65), ch.h * random.uniform(0.3, 0.55)
                    x = ch.cx + math.cos(a) * rx
                    y = top + ch.h * 0.4 + math.sin(a) * ry * 0.8
                    life = random.randint(14, 20)
                    self.stars.append([x, y, random.uniform(-0.4, 0.4) * sc, random.uniform(-0.8, -0.2) * sc,
                                       life, life, random.uniform(10, 17) * sc])
                self.cooldown = 4

        elif g == "PINCH" and pts is not None:
            self._hold(pts)

        elif g == "THUMB":
            if self.cooldown == 0 and len(self.hearts) < 9:
                sc = self.h / 720.0
                top = ch.bottom - ch.h
                size = random.uniform(15, 21) * sc
                for _ in range(14):
                    x = ch.cx + random.uniform(-0.32, 0.32) * ch.w
                    y = top + random.uniform(-0.14, 0.0) * ch.h
                    if all(math.hypot(x - o[0], y - o[1]) > (size + o[6]) * 1.3 for o in self.hearts):
                        life = random.randint(20, 26)
                        self.hearts.append([x, y, 0.0, -random.uniform(3.0, 4.2) * sc, life, life, size, 0.0])
                        self.pulse = max(self.pulse, 0.2)
                        break
                self.cooldown = 2

        elif g == "POINT" and pts is not None:
            v = pts[8] - pts[5]
            nv = float(np.linalg.norm(v))
            if nv > 1e-3 and self.cooldown == 0:
                d = v / nv
                start = chest + d * ch.h * 0.2
                self.shots.append([start[0], start[1], d[0] * 7, d[1] * 7, col])
                self.cooldown = 9
                self.pulse = 0.6

        if self.gesture == "HEART2":
            if self.big_age == 0:
                self.pulse = max(self.pulse, 0.4)
            self.big_age += 1
            self._draw_rainbow()
        else:
            self.big_age = 0
        if self.gesture != "TWO":
            self.trail_tip = None
            self.path = []
        elif g != "TWO":
            self._draw_ribbon()      # 손 모양이 잠깐 흔들려도 선이 깜빡이지 않게 그대로 유지

        self._step()

    def _step(self):
        self._draw_orbs()
        sparks = []
        for sp in self.sparks:
            sp[0] += sp[2]
            sp[1] += sp[3]
            sp[3] += 0.2 * self.h / 540.0
            sp[2] *= 0.9
            sp[3] *= 0.9
            sp[4] -= 1
            if sp[4] <= 0:
                continue
            k = sp[4] / sp[5]
            tail = 3.2 * (0.4 + 0.6 * k)
            cv2.line(self.layer, (int(sp[0] - sp[2] * tail), int(sp[1] - sp[3] * tail)), (int(sp[0]), int(sp[1])),
                     scale_color(sp[6], 0.3 + 0.7 * k), 2, cv2.LINE_AA)
            sparks.append(sp)
        self.sparks = sparks

        stars = []
        for st in self.stars:
            st[0] += st[2]
            st[1] += st[3]
            st[4] -= 1
            if st[4] <= 0:
                continue
            k = st[4] / st[5]
            tw = math.sin(k * math.pi)
            self.star(st[0], st[1], st[6] * (0.3 + 0.7 * tw), scale_color(self.main_col(), 0.35 + 0.5 * tw))
            stars.append(st)
        self.stars = stars

        hearts = []
        pink = hue_bgr(THEMES[4][1], v=255, s=170)
        for h in self.hearts:
            h[1] += h[3]
            h[4] -= 1
            if h[4] <= 0:
                continue
            age = h[5] - h[4]
            t = min(1.0, age / 6.0)
            pop = 1 + 2.7 * (t - 1) ** 3 + 1.7 * (t - 1) ** 2   # 뿅! 살짝 커졌다 자리잡는 곡선
            k = min(1.0, h[4] / (h[5] * 0.35))
            self.heart(h[0], h[1], h[6] * max(0.05, pop), scale_color(pink, 0.35 + 0.5 * k))
            hearts.append(h)
        self.hearts = hearts

        alive = []
        for p in self.parts:
            p[0] += p[2]
            p[1] += p[3]
            p[3] += p[8]
            p[2] *= p[9]
            p[3] *= p[9]
            p[4] -= 1
            if p[4] <= 0 or not (-40 < p[0] < self.w + 40 and -40 < p[1] < self.h + 40):
                continue
            k = p[4] / p[5]
            cv2.circle(self.layer, (int(p[0]), int(p[1])), max(1, int(p[7] * (0.4 + 0.6 * k))),
                       scale_color(p[6], 0.35 + 0.65 * k), -1, cv2.LINE_AA)
            alive.append(p)
        self.parts = alive

        rings = []
        for r in self.rings:
            r[2] += r[3]
            r[3] *= 0.93
            if r[2] < r[4] and r[3] > 0.6:
                k = 1 - r[2] / r[4]
                cv2.circle(self.layer, (int(r[0]), int(r[1])), int(r[2]), scale_color(r[5], 0.3 + 0.7 * k), 4,
                           cv2.LINE_AA)
                rings.append(r)
        self.rings = rings

        shots = []
        for s in self.shots:
            px, py = s[0], s[1]
            if math.hypot(s[2], s[3]) < 22:
                s[2] *= 1.1
                s[3] *= 1.1
            s[0] += s[2]
            s[1] += s[3]
            if -60 < s[0] < self.w + 60 and -60 < s[1] < self.h + 60:
                cv2.line(self.trail, (int(px), int(py)), (int(s[0]), int(s[1])), scale_color(s[4], 0.55), 14,
                         cv2.LINE_AA)
                self.orb(s[0], s[1], 11, scale_color(s[4], 0.9), core=False)
                self.emit(s[0], s[1], 1, speed=(0.2, 1.0), color=scale_color(s[4], 0.7), size=(1.5, 3), life=(8, 14), drag=0.9)
                shots.append(s)
            else:
                self.emit(min(max(s[0], 0), self.w), min(max(s[1], 0), self.h), 14, speed=(2, 6),
                          color=s[4], life=(10, 22))
        self.shots = shots

    # ------------------------------------------------------------ 합성
    def compose(self, frame, bg_mode=0):
        if bg_mode == 0:
            base = (frame * 0.7).astype(np.uint8)
        elif bg_mode == 1:
            base = (frame * 0.28).astype(np.uint8)
        else:
            base = np.zeros_like(frame)
        bob = 3.0 * math.sin(self.t * 3) + self.pulse * 14
        self.ch.draw(base, bob=bob, pulse=self.pulse, shake=self.shake)   # 캐릭터 먼저, 효과는 그 위에 빛으로 얹음
        lay = cv2.add(cv2.add(self.layer, self.trail), self.hlayer)
        glow = cv2.GaussianBlur(lay, (0, 0), 9)
        glow2 = cv2.GaussianBlur(lay, (0, 0), 3)
        out = cv2.add(base, cv2.add(lay, cv2.add(glow, glow2)))
        return out


# ---------------------------------------------------------------- 손 뼈대 (우측 상단 네모칸, 흰색)
def draw_skeleton_box(out, pts, w, h, pts2=None):
    """화면 우측 상단 네모칸 안에만 손 뼈대를 흰색으로 그린다. 칸은 카메라 화면의 축소판이라
    손이 카메라에서 있는 위치 그대로 칸 안에 나타난다. (out: 확대된 출력 화면, w/h: 카메라 해상도)"""
    H, W = out.shape[:2]
    bw = int(W * 0.22)
    bh = int(bw * h / w)
    x0, y0 = W - bw - 14, 14
    roi = out[y0:y0 + bh, x0:x0 + bw]
    roi[:] = (roi * 0.25).astype(np.uint8)                    # 반투명 어두운 바탕
    cv2.rectangle(out, (x0, y0), (x0 + bw, y0 + bh), (255, 255, 255), 2)   # 네모칸 테두리
    if pts is None:
        cv2.putText(out, "no hand", (x0 + 12, y0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1,
                    cv2.LINE_AA)
        return
    for hp in (pts, pts2):
        if hp is None:
            continue
        p = (hp * (bw / w)).astype(int) + np.array([x0, y0])
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(p[a]), tuple(p[b]), (255, 255, 255), 2, cv2.LINE_AA)
        for i in range(21):
            cv2.circle(out, tuple(p[i]), 5 if i in (4, 8, 12, 16, 20) else 3, (255, 255, 255), -1, cv2.LINE_AA)


# ---------------------------------------------------------------- 도움말 문구
def _find_korean_font():
    for c in ("C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc",
              "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
              "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"):
        if os.path.exists(c):
            return c
    return None


def build_help():
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None, HELP_EN
    fp = _find_korean_font()
    lines = HELP_KO if fp else HELP_EN
    try:
        if fp:
            font = ImageFont.truetype(fp, 20, index=3 if (fp.endswith(".ttc") and "Noto" in fp) else 0)
        else:
            font = ImageFont.load_default()
    except Exception:
        return None, HELP_EN
    wmax = max(int(font.getlength(s)) for s in lines) + 28
    hh = len(lines) * 30 + 20
    img = Image.new("RGBA", (wmax, hh), (0, 0, 0, 150))
    d = ImageDraw.Draw(img)
    for i, s in enumerate(lines):
        d.text((14, 10 + i * 30), s, font=font, fill=(255, 235, 150, 255) if i == 0 else (235, 235, 245, 255))
    arr = np.array(img)
    return arr[:, :, [2, 1, 0, 3]], lines


def draw_help_fallback(dst, lines):
    for i, s in enumerate(lines):
        cv2.putText(dst, s, (12, 26 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 245), 1, cv2.LINE_AA)


# ---------------------------------------------------------------- 카메라 / 실행
def open_camera():
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]
    for idx in (0, 1):
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
            except Exception:
                continue
            if cap is not None and cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap
                cap.release()
    return None


def main():
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError:
        sys.exit("mediapipe가 필요합니다 → pip install mediapipe")

    if not os.path.exists(HAND_MODEL_PATH):
        print("손 인식 모델 다운로드 중...")
        try:
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        except Exception as e:
            sys.exit(f"모델 다운로드 실패({e}). 인터넷 연결을 확인하세요.")
    # 경로에 한글이 있어도 되도록 파일 내용을 직접 읽어서 전달한다
    with open(HAND_MODEL_PATH, "rb") as f:
        model_bytes = f.read()
    if len(model_bytes) < 1_000_000:
        os.remove(HAND_MODEL_PATH)
        sys.exit("모델 파일이 손상됐습니다. 삭제했으니 다시 실행해서 새로 다운로드하세요.")
    landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=model_bytes),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.55, min_hand_presence_confidence=0.55, min_tracking_confidence=0.55))

    cap = open_camera()
    if cap is None:
        sys.exit("웹캠을 열 수 없습니다. 다른 프로그램이 카메라를 쓰고 있는지 확인하세요.")
    ok, frame = cap.read()
    h, w = frame.shape[:2]
    scale = 1.0 if w <= 1280 else 1280 / w
    if scale != 1.0:
        w, h = int(w * scale), int(h * scale)

    character = Character(w, h)
    magic = Magic(w, h, character)
    help_img, help_lines = build_help()

    win = "character_magic"
    dw = max(1280, w)
    dh = int(round(h * dw / w))
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, dw, dh)
    fullscreen, show_help, skeleton, bg_mode = False, True, True, 0
    last_ts, fps, t_prev = 0, 0.0, time.monotonic()
    prev_c = None

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        if scale != 1.0:
            frame = cv2.resize(frame, (w, h))
        frame = cv2.flip(frame, 1)
        ts = max(last_ts + 1, int(time.monotonic() * 1000))
        last_ts = ts
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = landmarker.detect_for_video(mp_img, ts)
        hands = [np.array([[q.x * w, q.y * h] for q in lm], np.float32) for lm in (result.hand_landmarks or [])]
        pts = pts2 = None
        if hands:
            # 한 손 효과는 직전에 쓰던 손을 계속 따라간다 (두 손이 보여도 손이 바뀌지 않게)
            if len(hands) > 1 and prev_c is not None:
                hands.sort(key=lambda a: float(np.linalg.norm(a[[0, 5, 9, 13, 17]].mean(0) - prev_c)))
            pts = hands[0]
            pts2 = hands[1] if len(hands) > 1 else None
            prev_c = pts[[0, 5, 9, 13, 17]].mean(0)
        else:
            prev_c = None

        magic.update(pts, pts2)
        out = magic.compose(frame, bg_mode)
        if (dw, dh) != (w, h):
            out = cv2.resize(out, (dw, dh), interpolation=cv2.INTER_CUBIC)
        if skeleton:
            draw_skeleton_box(out, pts, w, h, pts2)

        now = time.monotonic()
        fps = fps * 0.9 + (1 / max(1e-3, now - t_prev)) * 0.1
        t_prev = now
        cv2.putText(out, f"FPS {fps:.0f}  {magic.gesture}  color:{THEME_EN[magic.theme]}  follow:{'ON' if magic.follow else 'OFF'}", (12, dh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 210), 1, cv2.LINE_AA)
        if show_help:
            if help_img is not None:
                blit_clipped(out, help_img, 10, 10)
            else:
                draw_help_fallback(out, help_lines)

        cv2.imshow(win, out)
        k = cv2.waitKey(1) & 0xFF
        if k in (27, ord("q"), ord("Q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        elif k in (ord("b"), ord("B")):
            bg_mode = (bg_mode + 1) % 3
        elif k in (ord("k"), ord("K")):
            skeleton = not skeleton
        elif k in (ord("h"), ord("H")):
            show_help = not show_help
        elif k in (ord("t"), ord("T")):
            magic.theme = (magic.theme + 1) % len(THEMES)
        elif k in (ord("m"), ord("M")):
            magic.follow = not magic.follow
        elif k in (ord("c"), ord("C")):
            magic.clear()
        elif k in (ord("f"), ord("F")):
            fullscreen = not fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL)
        elif k in (ord("s"), ord("S")):
            os.makedirs("screenshots", exist_ok=True)
            name = os.path.join("screenshots", time.strftime("magic_%Y%m%d_%H%M%S.png"))
            cv2.imwrite(name, out)
            print("스크린샷 저장:", name)

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()