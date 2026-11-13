"""
scan and light.py  -  캠으로 그림 스캔 → 새 ARAP 방식으로 캐릭터 만들기 → 캠으로 움직이기

[흐름]
  0) 실행하면 standcharacter.png(팔 내린 마스코트)로 바로 시작 (START_CHARACTER 설정, ESC → 스캐너)
  1) 캠에 그림을 비추면 따질 영역이 초록색으로 보임 → SPACE → 3초 스캔
     (스캔하는 동안 가장 선명한 장면을 골라 저장, [ / ] 로 상자 크기 조절)
  2) 미리보기를 보고 Y → 저장하고 바로 시작 / N → 다시 스캔
     (미리보기에서 팔과 몸 사이 등 남은 흰 부분을 클릭하면 지워짐, Z 되돌리기)
     (미리보기에 자동으로 잡은 관절이 같이 표시됨, J 로 켜기/끄기)
     (G → 스캔 없이 가장 최근에 저장한 그림으로 바로 시작)
  3) 따낸 그림을 새 ARAP 방식으로 그물을 씌워서, 캠 속 내 동작대로 움직임
     - 종이색 테두리는 자동으로 지움
     - 팔과 몸통 사이 틈은 그물에 넣지 않음 (팔이 배를 끌고 가지 않음)
     - 얼굴·손·발은 휘지 않고 통째로 회전, 팔은 몸통 앞에 그림
     - 좌우로 움직이면 캐릭터도 이동 (C 로 기준 위치 다시 잡기), 점프하면 뛰어오름
     - 검지만 펴면 캐릭터 손에서 불빛이 나옴
     - 실행 중 키: C 기준 위치 | D 그물·조종점 보기 | V 캠 화면 | ESC 스캐너로
  4) ESC → 스캐너로 돌아옴 (다른 그림으로 다시 할 수 있음) / 스캐너에서 Q → 종료

[준비]  이 파일과 같은 폴더에 아래 파일이 있어야 합니다.
  - scan and play.py   : 저장/불러오기 기능을 가져다 씀 (스캐너는 이 파일의 강화판 사용)
  - Index light.py     : 새 ARAP 방식 + 캠으로 움직이기 + 검지 불빛
  - arap_dodge_game.py : 위 두 파일이 쓰는 기본 코드
  - pose_landmarker.task / hand_landmarker.task (Index light 와 같음)

실행:  python "scan and light.py"

[그림 팁]
  관절 위치를 그림 모양으로 자동으로 잡기 때문에, "정면을 보고 선" 캐릭터가 가장 잘 움직입니다.
  스캔 미리보기에서 관절 위치를 먼저 확인하고(J), 실행 중에는 D 키로 노란 조종점을 확인하세요.
"""
import importlib.util
import os
import sys
import time
import traceback

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def load_module(name, filename):
    """파일 이름에 띄어쓰기가 있어서 import 문으로는 못 불러오므로 파일 경로로 불러온다."""
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise SystemExit(f"'{filename}' 을(를) 찾지 못했습니다. 이 파일과 같은 폴더에 두세요.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


scanner = load_module("scan_and_play", "scan and play.py")   # 저장/불러오기 기능
light = load_module("index_light", "Index light.py")         # 새 ARAP + 캠 조작 + 검지 불빛

import cv2                                                   # noqa: E402
import numpy as np                                           # noqa: E402

# ------------------------------------------------------------------ 사용할 캐릭터
# 실행하자마자 이 그림으로 시작한다 (ESC → 스캐너로 가서 다른 그림을 스캔할 수 있음)
# None 으로 바꾸면 예전처럼 스캐너부터 시작
START_CHARACTER = "standcharacter.png"       # 동글동글 말 마스코트로 하려면 "character.png"

# ------------------------------------------------------------------ 강화된 스캐너 설정
SCAN_SECONDS = 3             # 스캔 시간(초). 이 동안 가장 선명한 장면을 골라 저장
CAM_SIZE = (1280, 720)       # 캠 해상도 요청 (안 되면 캠 기본값 사용)
BOX_RATIO = 0.85             # 스캔 상자 크기 (화면 세로 대비). [ / ] 키로 조절
CLOSE_SIZE = 7               # 끊긴 선 이어 붙이는 정도. 자꾸 잘리면 키우기(9, 11)
BRIDGE_SIZE = 21             # 크게 끊긴 선(손·소매)을 메울 때 쓰는 크기. 손이 잘리면 키우기
WIDE_ENOUGH = 9              # 메운 부분 중 이 두께(px의 절반)보다 넓은 곳만 그림으로 인정 (좁은 틈은 제외)
KEEP_RATIO = 0.015          # 가장 큰 덩어리의 이 비율보다 큰 조각(귀·꼬리 등)은 같이 남김
GROUP_GAP = 25               # 본체에서 이 거리(px) 안에 있는 조각만 같이 남김
# 미리보기 관절 표시: 선으로 이을 관절 쌍과 색(BGR)
JOINT_LINES = [("head", "neck"), ("neck", "hip"), ("neck", "sh_l"), ("neck", "sh_r"),
               ("sh_l", "el_l"), ("el_l", "hand_l"), ("sh_r", "el_r"), ("el_r", "hand_r"),
               ("hip", "kn_l"), ("kn_l", "foot_l"), ("hip", "kn_r"), ("kn_r", "foot_r")]
JOINT_COLORS = {"he": (90, 90, 255), "ne": (60, 160, 255), "hi": (0, 230, 255), "sh": (90, 220, 60),
                "el": (255, 210, 0), "ha": (255, 80, 255), "kn": (0, 140, 255), "fo": (255, 140, 120)}
PREVIEW_HEIGHT = 520         # 미리보기 창 높이(px)
ERASE_TOL = 22               # 클릭해서 지울 때 같은 색으로 볼 범위 (넓게 지워지면 줄이기)
EDGE_WARN = 12               # 그림이 상자 가장자리에 이만큼 붙으면 '잘릴 수 있음' 경고

YELLOW, GREEN, RED, ORANGE = (0, 255, 255), (0, 255, 0), (0, 0, 255), (0, 140, 255)


# ------------------------------------------------------------------ 강화된 캐릭터 따기
def detect_character(img):
    """상자 안 그림에서 캐릭터 영역(mask)을 찾는다. 못 찾으면 None.

    원래 스캐너: 검은 펜 선 → 가장 큰 외곽선 하나만 사용
      → 선이 조금만 끊겨도 팔·귀·꼬리가 잘림, 색칠만 된 부분(선 없는 곳)도 빠짐
    강화판:
      1) 펜 선 + 색칠된 부분 + 종이보다 어두운 부분을 모두 '그림'으로 봄
      2) 끊긴 선을 이어 붙임(닫기 연산)
      3) 가장 큰 덩어리 + 근처에 있는 작은 조각(귀·꼬리)까지 같이 남김
      4) 선으로 둘러싸인 안쪽(얼굴·손의 흰 부분)은 채움"""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    border = np.concatenate([gray[:6].ravel(), gray[-6:].ravel(), gray[:, :6].ravel(), gray[:, -6:].ravel()])
    paper = float(np.median(border))                      # 종이 밝기
    if paper < 80:                                        # 어두운 배경(천장, 방)은 무시
        return None

    block = max(15, (min(h, w) // 25) | 1)                # 상자 크기에 맞춘 이진화 범위
    lines = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                  cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    colored = ((hsv[:, :, 1] > 55) & (hsv[:, :, 2] > 40)).astype(np.uint8) * 255
    dark = (gray < paper - 45).astype(np.uint8) * 255
    fg = cv2.bitwise_or(cv2.bitwise_or(lines, colored), dark)

    m = 8                                                 # 상자 테두리선 간섭 제거
    fg[:m], fg[-m:], fg[:, :m], fg[:, -m:] = 0, 0, 0, 0
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # 잡티 제거
    # 실제 그림 픽셀: 아주 작은 틈(1~2px)만 이어 붙인 것
    fine = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # 덩어리 고르기용: 크게 이어 붙인 것 (이건 '어느 조각이 캐릭터인지' 고를 때만 쓰고
    # 그림 픽셀로는 쓰지 않음 → 팔과 몸통 사이 좁은 틈이 메워지지 않게)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_SIZE, CLOSE_SIZE))
    coarse = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(coarse, connectivity=8)
    if count < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    main = 1 + int(np.argmax(areas))
    if areas.max() < 1500:
        return None
    # 본체 근처(GROUP_GAP 이내)의 큰 조각만 같이 남기기
    near = cv2.dilate((labels == main).astype(np.uint8),
                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * GROUP_GAP + 1,) * 2))
    region = np.zeros_like(fg)
    for i in range(1, count):
        if i == main or (areas[i - 1] >= KEEP_RATIO * areas.max() and near[labels == i].any()):
            region[labels == i] = 255
    keep = cv2.bitwise_and(fine, region)

    # 안쪽 구멍 채우기: 바깥에서 물을 부어 닿지 않는 곳 = 안쪽
    mask = fill_holes(keep)

    # 선이 끊긴 곳(손·소매처럼 흰 부분)은 위에서 물이 새어 들어가 빠진다.
    # → 선을 점점 크게 이어 붙인 상태로 다시 채워 보고, 새로 생긴 부분 중
    #   '넓은' 부분(손·소매 안쪽)만 추가한다. 팔과 몸통 사이 같은 '좁은 틈'은 추가하지 않음.
    for size in (CLOSE_SIZE, BRIDGE_SIZE):
        kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        # 선을 두껍게 해서 끊긴 곳을 잇고 → 안쪽을 채우고 → 다시 원래 두께로
        filled = cv2.erode(fill_holes(cv2.dilate(keep, kk)), kk)
        extra = cv2.bitwise_and(filled, cv2.bitwise_not(mask))
        n, lab = cv2.connectedComponents(extra, connectivity=4)
        if n > 1:
            width = cv2.distanceTransform(extra, cv2.DIST_L2, 3)
            for i in range(1, n):
                part = lab == i
                if width[part].max() >= WIDE_ENOUGH:
                    mask[part] = 255
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))     # 바깥 선 끝까지 포함
    return mask


def fill_holes(m):
    flood = m.copy()
    h, w = m.shape
    cv2.floodFill(flood, np.zeros((h + 2, w + 2), np.uint8), (0, 0), 255)
    return cv2.bitwise_or(m, cv2.bitwise_not(flood))


def near_edge(mask):
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (xs.min() < EDGE_WARN or ys.min() < EDGE_WARN or
            xs.max() > w - 1 - EDGE_WARN or ys.max() > h - 1 - EDGE_WARN)


def sharpness(img, mask):
    """초점이 잘 맞은 장면일수록 큰 값"""
    lap = cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    return float(lap[mask > 0].var()) if mask.any() else 0.0


def cut_out(img, mask, pad=15):
    """mask 부분만 남긴 투명 배경 그림(texture)과 mask를, 여백 pad 를 두고 잘라서 돌려준다"""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    y1, y2 = max(0, ys.min() - pad), min(h, ys.max() + pad + 1)
    x1, x2 = max(0, xs.min() - pad), min(w, xs.max() + pad + 1)
    alpha = cv2.GaussianBlur(mask, (3, 3), 0)               # 가장자리 계단 현상 완화
    texture = cv2.merge([*cv2.split(img), alpha])[y1:y2, x1:x2]
    return texture, mask[y1:y2, x1:x2]


class PreviewEditor:
    """스캔 결과 미리보기 창. 남은 흰 부분(팔과 몸 사이 등)을 마우스로 클릭해서 지울 수 있다.
      왼쪽 클릭: 클릭한 곳과 같은 색으로 이어진 부분을 투명하게
      Z: 되돌리기
      J: 관절(조종점) 표시 켜기/끄기 — 실제 실행 때와 같은 방법으로 자동 추정한 위치.
         흰 부분을 지우면 관절 위치도 다시 계산된다."""

    def __init__(self, texture, mask):
        self.texture = texture.copy()
        self.mask = mask.copy()
        self.history = []
        # 펜 선(흐린 선 포함)을 '벽'으로 써서, 클릭한 흰 부분이 선 너머로 번지지 않게 한다
        gray = cv2.cvtColor(np.ascontiguousarray(texture[:, :, :3]), cv2.COLOR_BGR2GRAY)
        block = max(11, (min(gray.shape) // 20) | 1)
        lines = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (3, 3), 0), 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4)
        self.wall = cv2.dilate(lines, np.ones((3, 3), np.uint8)) > 0
        h = texture.shape[0]
        self.scale = max(1.0, PREVIEW_HEIGHT / h)
        self.show_joints = True
        self.update_joints()
        cv2.namedWindow("Preview")
        cv2.setMouseCallback("Preview", self.on_mouse)
        self.show()

    def show(self):
        h, w = self.texture.shape[:2]
        bg = np.full((h, w, 3), 40, np.uint8)                 # 체크무늬 배경 → 투명한 곳이 잘 보임
        yy, xx = np.mgrid[0:h, 0:w]
        bg[((yy // 10 + xx // 10) % 2) == 0] = 70
        a = self.texture[:, :, 3:4] / 255.0
        view = (self.texture[:, :, :3] * a + bg * (1 - a)).astype(np.uint8)
        view = cv2.resize(view, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_NEAREST)
        if self.show_joints and self.joints:
            self.draw_joints(view)
        cv2.putText(view, "click: erase  Z: undo  J: joints  Y/N", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, YELLOW, 2)
        cv2.imshow("Preview", view)

    def update_joints(self):
        """Index light 와 같은 순서(배경 정리 → 키 맞추기 → 자동 추정)로 관절 위치를 계산"""
        try:
            rgba = light.remove_background(cv2.cvtColor(self.texture, cv2.COLOR_BGRA2RGBA))
            alpha = rgba[:, :, 3]
            f = min(1.0, light.CHAR_HEIGHT / alpha.shape[0])   # 실행 때와 같은 크기로 줄여서 계산
            small = cv2.resize(alpha, (max(1, int(alpha.shape[1] * f)), max(1, int(alpha.shape[0] * f))),
                               interpolation=cv2.INTER_AREA)
            pts = light.auto_points(small)
            self.joints = {k: np.asarray(v, float) / f for k, v in pts.items()}
        except Exception as e:                                  # 그림이 너무 작거나 이상하면 표시만 생략
            print("--> 관절 위치를 계산하지 못했습니다:", e)
            self.joints = None

    def draw_joints(self, view):
        p = {k: tuple(int(c) for c in v * self.scale) for k, v in self.joints.items()}
        for a, b in JOINT_LINES:
            if a in p and b in p:
                cv2.line(view, p[a], p[b], (255, 255, 255), 2, cv2.LINE_AA)
        for k, c in p.items():
            cv2.circle(view, c, 7, JOINT_COLORS.get(k[:2], (0, 255, 255)), -1, cv2.LINE_AA)
            cv2.circle(view, c, 7, (255, 255, 255), 1, cv2.LINE_AA)

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        x, y = int(x / self.scale), int(y / self.scale)
        h, w = self.mask.shape
        if not (0 <= x < w and 0 <= y < h) or self.mask[y, x] == 0:
            return
        bgr = np.ascontiguousarray(self.texture[:, :, :3])
        ff = np.zeros((h + 2, w + 2), np.uint8)
        ff[1:-1, 1:-1][(self.mask == 0) | self.wall] = 1       # 투명한 곳과 펜 선은 건너가지 않음
        if self.wall[y, x]:                                     # 선 위를 클릭했으면 근처 선 아닌 곳으로
            ys, xs = np.nonzero(~self.wall & (self.mask > 0))
            k = np.argmin((ys - y) ** 2 + (xs - x) ** 2)
            y, x = int(ys[k]), int(xs[k])
        cv2.floodFill(bgr.copy(), ff, (x, y), 0, (ERASE_TOL,) * 3, (ERASE_TOL,) * 3,
                      4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        region = ff[1:-1, 1:-1] == 255
        if region.sum() > 0.35 * (self.mask > 0).sum():      # 너무 넓으면(몸통 전체 등) 실수로 보고 무시
            print("--> 지울 부분이 너무 넓어서 무시했습니다. 더 좁은 곳을 클릭하세요.")
            return
        region = cv2.dilate(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        region &= self.mask > 0
        self.history.append((self.texture.copy(), self.mask.copy()))
        self.texture[region, 3] = 0
        self.mask[region] = 0
        self.update_joints()
        self.show()

    def undo(self):
        if self.history:
            self.texture, self.mask = self.history.pop()
            self.update_joints()
            self.show()


def scan_character():
    """강화된 스캐너. 게임에 쓸 texture.png 경로를 돌려준다 (Q로 끝내면 None)

    원래 스캐너와 달라진 점
      - 캠 해상도를 높이고 스캔 상자를 키움 ([ / ] 키로 크기 조절)
      - 스캔 전에도 따질 영역을 초록색으로 미리 보여줌 → 잘리는 부분을 미리 확인
      - 그림이 상자 가장자리에 붙으면 '잘릴 수 있음' 경고
      - 스캔하는 동안 가장 선명한(흔들리지 않은) 장면을 골라 저장"""
    camera = light.game.open_camera()
    if camera is None:
        print("웹캠을 열 수 없습니다 (다른 프로그램이 사용 중이거나 권한이 없을 수 있음)")
        return None
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_SIZE[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_SIZE[1])

    box_ratio = BOX_RATIO
    start_time = None
    best = None                    # (선명도, 상자 그림, mask)
    texture = mask_out = editor = None
    warn_until = 0.0

    def label(frame, text, color, x, y):
        cv2.putText(frame, text, (max(5, x), max(25, y - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    try:
        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                print("캠에서 화면을 읽지 못했습니다.")
                return None
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            box_h = int(min(H - 50, H * box_ratio))
            box_w = int(min(W - 20, box_h * 0.8))           # 세로로 긴 상자 (서 있는 캐릭터용)
            x1, y1 = (W - box_w) // 2, (H - box_h) // 2 + 15
            x2, y2 = x1 + box_w, y1 + box_h
            inside = frame[y1:y2, x1:x2].copy()

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                return None
            if key == ord('['):
                box_ratio = max(0.4, box_ratio - 0.05)
            if key == ord(']'):
                box_ratio = min(0.95, box_ratio + 0.05)

            # 스캔 완료 → Y / N
            if texture is not None:
                color, text = YELLOW, "PLAY? [Y] / [N]   (Preview: click to erase)"
                if key in (ord('y'), ord('Y')):
                    return scanner.save_character(editor.texture, editor.mask)
                if key in (ord('z'), ord('Z')):
                    editor.undo()
                if key in (ord('j'), ord('J')):
                    editor.show_joints = not editor.show_joints
                    editor.show()
                if key in (ord('n'), ord('N')):
                    texture = mask_out = editor = None
                    cv2.destroyWindow("Preview")
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label(frame, text, color, x1, y1)
                cv2.imshow("Character Scanner", frame)
                continue

            mask = detect_character(inside)

            # 미리보기: 따질 영역을 초록색으로 살짝 칠함
            if mask is not None:
                overlay = frame[y1:y2, x1:x2]
                tint = overlay.copy()
                tint[mask > 0] = (0.55 * tint[mask > 0] + 0.45 * np.array(GREEN)).astype(np.uint8)
                frame[y1:y2, x1:x2] = tint
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame[y1:y2, x1:x2], contours, -1, GREEN, 2)

            if key == 32:                                    # SPACE
                start_time, best = time.time(), None
            elif key in (ord('c'), ord('C')) and start_character_path():
                return start_character_path()              # C: START_CHARACTER 그림으로 시작
            elif key in (ord('g'), ord('G')):
                last = scanner.latest_saved_character()
                if last is not None:
                    return last
                print("--> 저장된 캐릭터가 없습니다. 먼저 스캔하세요.")

            edge = mask is not None and near_edge(mask)
            if start_time is not None:
                if mask is None:
                    start_time = None
                    warn_until = time.time() + 1.5
                else:
                    s = sharpness(inside, mask)
                    if best is None or s > best[0]:
                        best = (s, inside, mask)
                    if time.time() - start_time >= SCAN_SECONDS:
                        texture, mask_out = cut_out(best[1], best[2])
                        if scanner.UNMIRROR_SAVED:
                            texture, mask_out = cv2.flip(texture, 1), cv2.flip(mask_out, 1)
                        editor = PreviewEditor(texture, mask_out)
                        start_time = None

            if start_time is not None:
                left = max(0, SCAN_SECONDS - int(time.time() - start_time))
                color, text = GREEN, f"Scanning... {left}s  (hold still)"
            elif time.time() < warn_until:
                color, text = RED, "No Drawing Detected!"
            elif edge:
                color, text = RED, "Too close to edge - move back or ]"
            else:
                color, text = ORANGE, "[SPACE] Scan  [C] Default  [G] Last  [ ] Box  [Q] Quit"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label(frame, text, color, x1, y1)
            cv2.imshow("Character Scanner", frame)
    finally:
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def start_character_path():
    """START_CHARACTER 그림의 전체 경로 (없으면 None)"""
    if not START_CHARACTER:
        return None
    path = os.path.join(BASE_DIR, START_CHARACTER)
    if not os.path.exists(path):
        print(f"--> {START_CHARACTER} 파일을 찾지 못했습니다. 이 파일과 같은 폴더에 두세요.")
        return None
    return path


def play(texture_path):
    """스캔한 그림으로 Index light 실행 (ESC 누르면 돌아옴)"""
    light.USE_LATEST_SCAN = False                         # 지정한 그림 경로를 그대로 쓰도록
    light.find_character_path = lambda: texture_path
    try:
        light.main()
    except SystemExit as e:
        print(e)
    except Exception:
        traceback.print_exc()
        print("--> 이 그림으로 시작하지 못했습니다. 다시 스캔해 보세요.")
    finally:
        try:
            light.pygame.quit()
        except Exception:
            pass


def main():
    print("=== 그림 스캔 → 새 ARAP 캐릭터 → 캠으로 움직이기 ===")
    print("[SPACE] 3초 스캔   [Y] 이 그림으로 시작   [N] 다시 스캔   [ / ] 상자 크기")
    print("[G] 최근 저장한 그림으로 바로 시작   [Q] 종료")
    print(f"[C] {START_CHARACTER} 로 다시 시작")
    print("실행 중 [ESC] 를 누르면 스캐너로 돌아옵니다.  (D 조종점 보기, V 캠 화면)")
    first = start_character_path()
    if first:                          # 처음엔 START_CHARACTER 그림으로 바로 시작
        print("시작:", first)
        play(first)
        time.sleep(0.3)
    while True:
        texture_path = scan_character()
        if texture_path is None:
            break
        time.sleep(0.3)                # 캠이 완전히 풀릴 때까지 잠깐 대기
        print("시작:", texture_path)
        play(texture_path)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
