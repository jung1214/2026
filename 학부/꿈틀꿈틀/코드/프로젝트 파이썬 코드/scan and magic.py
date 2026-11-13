"""
scan and magic.py  -  그림 스캔 → 캐릭터가 캠 속 나를 따라 움직임 → 손가락으로 특수효과

[흐름]
  0) 실행하면 START_CHARACTER(scan and light.py 설정, 기본 standcharacter.png)로 바로 시작
     ESC → 스캐너로 가서 새 그림을 스캔할 수 있음
  1) 스캐너: 캠에 그림을 비추고 SPACE → 3초 스캔 → 미리보기에서 Y (G: 최근 그림, Q: 종료)
  2) 따낸 그림이 ARAP 방식으로 캠 속 내 몸 동작(팔·다리·좌우 이동·앉기·점프)을 따라 함
  3) 손 모양에 따라 캐릭터 손에서 특수효과가 나옴
       검지만 펴기          : 캐릭터의 같은 쪽 손에서 불빛 (Index light 와 같음, 양손 따로)
       손바닥 활짝          : 캐릭터 양손에서 폭죽
       브이 (검지+중지 벌림) : 캐릭터 주위에 별이 반짝
       엄지 척              : 캐릭터 머리 위로 분홍 하트
       엄지+검지 집게        : 빛 구슬을 잡아서 옮기고, 벌리면 내려놓기
       검지+중지 모으기      : 캐릭터 손에서 나온 빛이 내 손가락을 따라감
       두 손 모으기          : 캐릭터 머리 위에 무지개
       주먹                 : 모든 효과 지우기

[조작]  T 효과 색 | X 효과 지우기 | H 도움말 | C 지금 자리를 좌우 이동 기준으로 | D 그물·조종점 | V 캠 화면
        ESC 스캐너로 (스캐너에서 Q → 종료)

[준비]  이 파일과 같은 폴더에 아래 파일이 있어야 합니다.
  - scan and light.py  : 스캐너 (+ scan and play.py, Index light.py, arap_dodge_game.py 를 불러옴)
  - Index light.py     : ARAP 그물 · 캠으로 움직이기 · 검지 불빛
  - finger_effects.py  : 손 모양 판정 · 특수효과 엔진
  - pose_landmarker.task / hand_landmarker.task

실행:  python "scan and magic.py"
[개인정보] 캠 영상은 화면에만 표시하고 저장하지 않습니다.
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


sl = load_module("scan_and_light", "scan and light.py")   # 스캐너 (안에서 Index light 도 불러옴)
light = sl.light                                         # Index light: ARAP + 캠 + 검지 불빛
game = light.game                                        # arap_dodge_game: 기본 코드
fe = load_module("finger_effects", "finger_effects.py")  # 손 모양 판정 + 특수효과

import cv2                                               # noqa: E402
import numpy as np                                       # noqa: E402
import pygame                                            # noqa: E402

# ------------------------------------------------------------------ 설정값
SCREEN_W, SCREEN_H = 1280, 720
FPS = 30                     # 특수효과가 30프레임 기준으로 만들어져 있어서 30으로 맞춤
GROUND_GAP = 40              # 캐릭터 발과 화면 아래 사이 여백(px)
GLOW = True                  # 특수효과 번짐(빛 퍼짐) 효과. 느리면 False
SHOW_HELP = True             # 시작할 때 도움말 보이기

HELP_LINES = [
    "내 캐릭터 특수효과",
    "검지만 펴기 : 같은 쪽 손에서 불빛",
    "손바닥 활짝 : 양손에서 폭죽",
    "브이 : 주위에 별 반짝",
    "엄지 척 : 분홍 하트",
    "집게(엄지+검지) : 빛 구슬 잡아 옮기기",
    "검지+중지 모으기 : 손에서 나온 빛이 손가락을 따라감",
    "두 손 모으기 : 머리 위에 무지개",
    "주먹 : 효과 모두 지우기",
    "T 색  X 지우기  H 도움말  C 기준 위치  D 그물  V 캠  ESC 스캐너",
]


# ------------------------------------------------------------------ 캠: 몸 + 손 (손 좌표도 같이 넘김)
class MagicTracker(light.BodyHandTracker):
    """Index light 의 BodyHandTracker 와 같고, 손 21개 점을 화면 좌표로 바꿔서 함께 넘긴다
    (손 모양 판정·손가락을 따라가는 효과에 씀)."""

    def __init__(self):
        super().__init__()
        self.hands = {}          # {"l"/"r": (21,2) 화면 좌표}

    def to_screen(self, h, fw, fh):
        """캠 좌표 → 화면 좌표 (비율을 지키면서 화면을 꽉 채우도록, 가운데 기준)"""
        s = max(SCREEN_W / fw, SCREEN_H / fh)
        out = np.empty((len(h), 2), np.float32)
        out[:, 0] = (h[:, 0] - fw / 2) * s + SCREEN_W / 2
        out[:, 1] = (h[:, 1] - fh / 2) * s + SCREEN_H / 2
        return out

    def _loop(self):
        last_ts = 0
        self._filter = light.OneEuro(light.EURO_MIN_CUTOFF, light.EURO_BETA)
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
                image = light.mp.Image(image_format=light.mp.ImageFormat.SRGB, data=rgb)
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
                if light.SMOOTH_FILTER:
                    pts[:, :2] = self._filter(pts[:, :2], time.monotonic())
                if min(pts[11, 2], pts[12, 2]) >= 0.3:
                    pose = {"time": time.monotonic(), "pts": pts, "fh": fh,
                            "cx": float((pts[11, 0] + pts[12, 0]) / 2 / fw)}

            # 손 → 캐릭터의 어느 손인지 짝짓기
            hands = [np.array([[p.x * fw, p.y * fh] for p in h]) for h in (hr.hand_landmarks or [])]
            by_side = {}
            for h in hands:
                s = light.hand_side(h, pts, fw)
                if s in by_side:
                    s = "r" if s == "l" else "l"
                by_side[s] = h

            # 검지 판정 (연속 프레임 + 잠깐 놓쳐도 유지) - Index light 와 같음
            pointing_now = {}
            for s in "lr":
                h = by_side.get(s)
                pointing_now[s] = h is not None and light.is_pointing(h)
                if pointing_now[s]:
                    self._count[s] += 1
                    self._miss[s] = 0
                    v = h[8] - h[5]
                    n = np.linalg.norm(v)
                    if n > 1e-6:
                        self._dir[s] = v / n
                else:
                    self._miss[s] += 1
                    if self._miss[s] > light.MISS_TOLERANCE:
                        self._count[s] = 0

            # 미리보기: 손 점 + 검지 끝 표시
            view = rgb.copy()
            for s, h in by_side.items():
                for p in h:
                    cv2.circle(view, (int(p[0]), int(p[1])), 3, (200, 200, 200), -1)
                color = (255, 205, 90) if pointing_now[s] else (120, 120, 255)
                cv2.circle(view, (int(h[8][0]), int(h[8][1])), 10, color, -1)
            small = cv2.resize(view, light.PREVIEW_SIZE)

            screen_hands = {s: self.to_screen(h, fw, fh) for s, h in by_side.items()}
            with self.lock:
                if pose is not None:
                    self.pose = pose
                self.light_on = {s: self._count[s] >= light.HOLD_FRAMES for s in "lr"}
                self.point_dir = dict(self._dir)
                self.preview = small.tobytes()
                self.hands = screen_hands

    def get_hands(self):
        with self.lock:
            return dict(self.hands)


# ------------------------------------------------------------------ 특수효과 엔진이 볼 캐릭터 정보
class ArapCharacter:
    """finger_effects 의 Magic 이 쓰는 캐릭터 정보(cx, bottom, w, h, 손 위치, 가슴 위치)를
    매 프레임 움직이는 ARAP 캐릭터에서 채워 넣는다 → 효과가 실제로 움직이는 손에서 나옴."""

    def __init__(self, mesh, rest):
        self.w, self.h = mesh.w, mesh.h
        self.idx = {k: mesh.handle_idx[light.HANDLES.index(k)] for k in ("head", "neck", "hip", "hand_l", "hand_r")}
        self.cx, self.bottom = SCREEN_W / 2, SCREEN_H - GROUND_GAP
        self._hands = [np.zeros(2, np.float32), np.zeros(2, np.float32)]
        self._chest = np.zeros(2, np.float32)

    def update(self, points, origin):
        P = points + origin
        self.cx = float(P[self.idx["head"], 0])          # 무지개·하트·별은 머리 위/둘레 기준
        self.bottom = float(P[:, 1].max())                # 점프하면 효과도 같이 올라감
        self._hands = [P[self.idx["hand_l"]].astype(np.float32), P[self.idx["hand_r"]].astype(np.float32)]
        self._chest = ((P[self.idx["neck"]] + P[self.idx["hip"]]) / 2).astype(np.float32)

    def hand_points(self):
        return [self._hands[0].copy(), self._hands[1].copy()]

    def chest(self):
        return self._chest.copy()


class ArapMagic(fe.Magic):
    POINT_SHOT = False       # 검지만 펴기는 구슬 발사 대신 Index light 의 검지 불빛을 씀


def magic_surface(magic):
    """Magic 의 빛 레이어(BGR)를 pygame 화면에 더하기 합성할 수 있는 Surface 로"""
    lay = cv2.add(cv2.add(magic.layer, magic.trail), magic.hlayer)
    if not lay.any():
        return None
    if GLOW:                                              # 반 크기에서 번지게 해서 빠르게
        small = cv2.resize(lay, (SCREEN_W // 2, SCREEN_H // 2), interpolation=cv2.INTER_AREA)
        g1 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 4.5), (SCREEN_W, SCREEN_H))
        g2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 1.5), (SCREEN_W, SCREEN_H))
        lay = cv2.add(lay, cv2.add(g1, g2))
    rgb = cv2.cvtColor(lay, cv2.COLOR_BGR2RGB)
    return pygame.image.frombuffer(rgb.tobytes(), (SCREEN_W, SCREEN_H), "RGB")


def theme_rgb(theme):
    """검지 불빛 색: 노랑(첫 번째 색)은 Index light 원래 색, 나머지는 효과 색과 같게"""
    if theme == 0:
        return ORIGINAL_LIGHT
    b, g, r = fe.hue_bgr(fe.THEMES[theme][1])
    return (r, g, b)


ORIGINAL_LIGHT = tuple(light.LIGHT_COLOR)


def make_lights(theme):
    light.LIGHT_COLOR = theme_rgb(theme)                  # HandLight 가 이 색으로 빛을 만듦
    return {s: light.HandLight((SCREEN_W, SCREEN_H)) for s in "lr"}


def help_surface(font):
    lines = [font.render(s, True, (255, 235, 150) if i == 0 else (235, 235, 245)) for i, s in enumerate(HELP_LINES)]
    w = max(t.get_width() for t in lines) + 24
    h = len(lines) * 26 + 16
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 150))
    for i, t in enumerate(lines):
        surf.blit(t, (12, 8 + i * 26))
    return surf


# ------------------------------------------------------------------ 실행
def run(path):
    """path 그림으로 캐릭터를 만들고 캠으로 움직이며 특수효과 (ESC → 돌아감)"""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("내 캐릭터 + 손가락 특수효과 (scan and magic)")
    clock = pygame.time.Clock()
    font = game.load_font(18)

    print("캐릭터 그림 사용:", path)
    rgba = light.load_character_rgba(path)
    ratios = light.CUSTOM_POINTS.get(os.path.basename(path))
    if ratios is not None:                                   # 조종점을 직접 지정한 그림 (standcharacter.png)
        pts = light.complete_points(light.custom_points(ratios, rgba[:, :, 3]))
    else:                                                    # 스캔한 그림 등: 자동으로
        pts = light.auto_points(rgba[:, :, 3])
    game.find_character_points = lambda alpha: pts
    mesh = light.OutlineMesh(rgba, step=light.MESH_STEP)
    if light.FIX_HEAD or light.FIX_HANDS_FEET:
        mesh.fix_parts(head=light.FIX_HEAD, hands_feet=light.FIX_HANDS_FEET)
    mesh.arms_on_top()
    rest = mesh.rest_pts
    hand_vertex = {s: mesh.handle_idx[light.HANDLES.index("hand_" + s)] for s in "lr"}

    tracker = MagicTracker()
    if tracker.ok:
        tracker.start()
        print("캠 인식 시작: 캠 앞에 서서 손 모양을 바꿔 보세요.")
    else:
        print("캠을 사용할 수 없습니다:", tracker.error)

    origin = np.array([(SCREEN_W - mesh.w) / 2, SCREEN_H - mesh.h - GROUND_GAP], float)
    center_x = origin[0]
    half_range = (SCREEN_W - mesh.w) / 2
    ref_cx = light.MOVE_REF
    targets = light.pose_to_targets(None, rest)
    jump = light.JumpTracker()
    torso_char = np.linalg.norm(rest["neck"] - rest["hip"])
    leg_char = max(rest["foot_l"][1], rest["foot_r"][1]) - rest["hip"][1]
    lift_now = 0.0

    ch = ArapCharacter(mesh, rest)
    magic = ArapMagic(SCREEN_W, SCREEN_H, ch)
    magic.follow = False                                     # 좌우 이동은 몸(ARAP)이 맡음
    lights = make_lights(magic.theme)
    help_img = help_surface(font)
    show_mesh, show_camera, show_help = False, True, SHOW_HELP
    prev_c = None

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
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_x:
                    magic.clear()
                elif event.key == pygame.K_t:
                    magic.theme = (magic.theme + 1) % len(fe.THEMES)
                    lights = make_lights(magic.theme)
                elif event.key == pygame.K_c and tracker.get()[0] is not None:
                    ref_cx = tracker.get()[0]["cx"]
                    print(f"기준 위치를 다시 잡았습니다: {ref_cx:.2f}")

        pose, light_on, point_dir, preview = tracker.get()
        hands = tracker.get_hands()
        tracking = pose is not None and time.monotonic() - pose["time"] < game.LOST_TIME

        # 좌우 이동
        if light.FOLLOW_X and tracking:
            t = game.clamp((pose["cx"] - ref_cx) / light.MOVE_RANGE, -1.0, 1.0)
            origin[0] += (center_x + t * half_range - origin[0]) * light.MOVE_SMOOTH

        # 캐릭터 동작 (ARAP) - Index light 와 같음
        lift_now = jump.update(pose["pts"] if tracking else None, pose.get("fh") if tracking else None)
        if not light.JUMP:
            lift_now = 0.0
        sink = jump.squat * leg_char * light.SQUAT_GAIN if light.SQUAT else 0.0
        goal = light.pose_to_targets(pose["pts"] if tracking else None, rest, lift_now * torso_char * light.JUMP_GAIN,
                                     jump.ground, min(sink, leg_char * light.SQUAT_MAX))
        targets += (goal - targets) * (light.TARGET_SMOOTH if light.SMOOTH_FILTER else game.POSE_SMOOTH)
        points = mesh.solve(targets)
        char_surface = mesh.render(points)
        ch.update(points, origin)

        # 손 모양 특수효과: 한 손 효과는 직전에 쓰던 손을 계속 따라감 (finger_effects 와 같음)
        hl = list(hands.values())
        if len(hl) > 1 and prev_c is not None:
            hl.sort(key=lambda a: float(np.linalg.norm(a[[0, 5, 9, 13, 17]].mean(0) - prev_c)))
        hp = hl[0] if hl else None
        hp2 = hl[1] if len(hl) > 1 else None
        prev_c = hp[[0, 5, 9, 13, 17]].mean(0) if hp is not None else None
        magic.update(hp, hp2)

        # 검지 불빛: Index light 와 같음 (캐릭터 같은 쪽 손 + 검지 방향으로 살짝)
        #   집게(엄지+검지)나 두 손 모으기 중에는 켜지 않음
        for s in "lr":
            h = hands.get(s)
            on = light_on[s] and h is not None and fe.classify(h) == "POINT" and magic.gesture != "HEART2"
            hand = points[hand_vertex[s]] + origin
            d = point_dir[s]
            target = hand + d * light.REACH * light.CHAR_HEIGHT if d is not None else hand
            lights[s].update(on, target)

        # 그리기: 배경 → 캐릭터 → 특수효과 → 검지 불빛
        screen.fill(light.BG_COLOR)
        screen.blit(char_surface, (int(origin[0]) - mesh.pad, int(origin[1]) - mesh.pad))
        fx = magic_surface(magic)
        if fx is not None:
            screen.blit(fx, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        for s in "lr":
            lights[s].draw(screen)

        if show_mesh:
            sp = points + origin
            for a, b in mesh.edges:
                pygame.draw.line(screen, (0, 220, 255), sp[a], sp[b], 1)
            for p in targets:
                pygame.draw.circle(screen, (255, 230, 0), (p + origin).astype(int), 5)

        if show_camera and preview is not None:
            cam = pygame.image.frombuffer(preview, light.PREVIEW_SIZE, "RGB")
            cam_x = SCREEN_W - light.PREVIEW_SIZE[0] - 10
            screen.blit(cam, (cam_x, 10))
            pygame.draw.rect(screen, (255, 255, 255), (cam_x, 10, *light.PREVIEW_SIZE), 1)
            if light.FOLLOW_X:
                for v, colr, wd in ((ref_cx - light.MOVE_RANGE, (120, 120, 120), 1),
                                    (ref_cx + light.MOVE_RANGE, (120, 120, 120), 1),
                                    (ref_cx, (255, 220, 0), 2)):
                    x = cam_x + int(game.clamp(v, 0, 1) * light.PREVIEW_SIZE[0])
                    pygame.draw.line(screen, colr, (x, 10), (x, 10 + light.PREVIEW_SIZE[1]), wd)
                if tracking:
                    x = cam_x + int(pose["cx"] * light.PREVIEW_SIZE[0])
                    pygame.draw.line(screen, (0, 255, 0), (x, 10), (x, 10 + light.PREVIEW_SIZE[1]), 2)

        if show_help:
            screen.blit(help_img, (10, 10))

        if not tracker.ok:
            status = "캠 없음: " + tracker.error
        elif not tracking:
            status = "몸이 안 보여요 - 캠 앞에 서 주세요"
        else:
            status = f"손 모양: {magic.gesture}   색: {fe.THEMES[magic.theme][0]}"
            if lift_now > 0:
                status = "점프!   " + status
        screen.blit(font.render(status + "   |   H 도움말  ESC 스캐너로", True, (170, 170, 170)),
                    (10, SCREEN_H - 28))

        pygame.display.flip()
        clock.tick(FPS)

    tracker.close()
    pygame.quit()


def play(path):
    try:
        run(path)
    except SystemExit as e:
        print(e)
    except Exception:
        traceback.print_exc()
        print("--> 이 그림으로 시작하지 못했습니다. 다시 스캔해 보세요.")
    finally:
        try:
            pygame.quit()
        except Exception:
            pass


def main():
    print("=== 그림 스캔 → 내 동작 따라하기 → 손가락 특수효과 ===")
    print("[SPACE] 3초 스캔   [Y] 이 그림으로 시작   [N] 다시 스캔   [ / ] 상자 크기")
    print("[G] 최근 저장한 그림으로 바로 시작   [Q] 종료")
    print("실행 중 [ESC] 를 누르면 스캐너로 돌아옵니다.  (H 도움말, T 색, X 지우기)")
    first = sl.start_character_path()
    if first:
        print("시작:", first)
        play(first)
        time.sleep(0.3)
    while True:
        path = sl.scan_character()
        if path is None:
            break
        time.sleep(0.3)
        print("시작:", path)
        play(path)
        time.sleep(0.3)


if __name__ == "__main__":
    main()
