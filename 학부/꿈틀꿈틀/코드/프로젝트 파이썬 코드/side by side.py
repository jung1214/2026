"""
side by side.py  -  화면 반반: 왼쪽은 내 캐릭터, 오른쪽은 캠 속 나 → 둘이 나란히 같은 동작

[구도]
   ┌──────────────┬──────────────┐
   │   내 캐릭터    ┆    캠 속 나     │
   │ (나를 따라 함) ┆  (거울처럼 보임) │
   └──────────────┴──────────────┘
  - 왼쪽: 스캔한 그림(또는 마스코트)이 ARAP 방식으로 내 동작을 그대로 따라 함
  - 오른쪽: 캠 화면 (거울처럼 좌우 반전) - 가운데 점선으로 나눔
  - 손가락 특수효과는 scan and magic.py 와 같음 (캐릭터 손에서 나옴)
    검지만 펴면 캐릭터 손과 캠 속 내 검지 끝에서 같이 불빛이 남

[흐름]  scan and magic.py 와 같음
  0) START_CHARACTER(scan and light.py 설정)로 바로 시작 → ESC 누르면 스캐너
  1) 스캐너: SPACE 3초 스캔 → Y 시작 (G 최근 그림, Q 종료)

[조작]  T 효과 색 | X 효과 지우기 | H 도움말 | C 지금 자리를 좌우 이동 기준으로 | D 그물·조종점
        L 화면 좌우 바꾸기(캐릭터 ↔ 나) | F 또는 F11 전체화면 켜기/끄기 | ESC 스캐너로

[준비]  scan and magic.py 가 돌아가는 폴더에 이 파일을 같이 두면 됨
        (scan and magic.py, scan and light.py, scan and play.py, Index light.py,
         arap_dodge_game.py, finger_effects.py, 캐릭터 png, *.task 모델)

실행:  python "side by side.py"
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
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise SystemExit(f"'{filename}' 을(를) 찾지 못했습니다. 이 파일과 같은 폴더에 두세요.")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


sam = load_module("scan_and_magic", "scan and magic.py")   # 캐릭터 + 특수효과 (스캐너·ARAP 포함)
sl, light, game, fe = sam.sl, sam.light, sam.game, sam.fe

import cv2                                               # noqa: E402
import numpy as np                                       # noqa: E402
import pygame                                            # noqa: E402

# ------------------------------------------------------------------ 설정값
SCREEN_W, SCREEN_H = 1280, 720
PANEL_W = SCREEN_W // 2      # 한쪽 칸 너비
CHAR_ON_LEFT = True          # True: 왼쪽 캐릭터 / 오른쪽 나 (그림과 같은 구도). L 키로 바꿈
CHAR_HEIGHT = 380            # 캐릭터 키(픽셀). 칸이 좁아져서 Index light(340)보다 살짝 키움
GROUND_GAP = 40              # 캐릭터 발과 화면 아래 사이 여백(px)
CAM_DIM = 0.85               # 캠 화면 밝기 (1이면 그대로, 낮추면 효과가 더 잘 보임)
DIVIDER_COLOR = (235, 235, 235)
LABELS = True                # 칸 위에 '캐릭터' / '나' 글씨
MY_FINGER_LIGHT = True       # 검지를 펴면 캠 속 내 검지 끝에서도 불빛
FULLSCREEN = False           # True: 전체화면으로 시작. 실행 중 F 또는 F11 로 켜고 끔 (다음 그림에도 유지)
SMOOTH_SCALE = True          # 전체화면에서 부드럽게 확대 (느리면 False)

sam.SCREEN_W, sam.SCREEN_H = SCREEN_W, SCREEN_H           # 효과 레이어 크기를 같게
light.CHAR_HEIGHT = CHAR_HEIGHT


def panels(char_left):
    """(캐릭터 칸 x, 캠 칸 x)"""
    return (0, PANEL_W) if char_left else (PANEL_W, 0)


# ------------------------------------------------------------------ 캠: 몸 + 손 + 큰 캠 화면
class CapTap:
    """캠에서 읽은 장면을 그대로 넘겨주면서, 오른쪽 칸에 띄울 크기로 한 장 복사해 둔다."""

    def __init__(self, cap, owner):
        self.cap, self.owner = cap, owner

    def read(self):
        ok, frame = self.cap.read()
        if ok and frame is not None:
            fh, fw = frame.shape[:2]
            s = max(PANEL_W / fw, SCREEN_H / fh)                   # 칸을 꽉 채우게(넘치는 곳은 잘라냄)
            img = cv2.resize(cv2.flip(frame, 1), (int(round(fw * s)), int(round(fh * s))))
            y0 = (img.shape[0] - SCREEN_H) // 2
            x0 = (img.shape[1] - PANEL_W) // 2
            img = img[y0:y0 + SCREEN_H, x0:x0 + PANEL_W]
            if CAM_DIM < 1:
                img = (img * CAM_DIM).astype(np.uint8)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            with self.owner.lock:
                self.owner.cam = rgb.tobytes()
        return ok, frame

    def release(self):
        self.cap.release()

    def __getattr__(self, name):
        return getattr(self.cap, name)


class SideTracker(sam.MagicTracker):
    def __init__(self):
        super().__init__()
        self.cam = None
        self.cam_x = PANEL_W
        if self.cap is not None:
            self.cap = CapTap(self.cap, self)

    def to_screen(self, h, fw, fh):
        """캠 좌표 → 오른쪽(캠) 칸 안의 화면 좌표 → 손가락 효과가 화면 속 내 손 위에 나옴"""
        s = max(PANEL_W / fw, SCREEN_H / fh)
        out = np.empty((len(h), 2), np.float32)
        out[:, 0] = (h[:, 0] - fw / 2) * s + PANEL_W / 2 + self.cam_x
        out[:, 1] = (h[:, 1] - fh / 2) * s + SCREEN_H / 2
        return out

    def get_cam(self):
        with self.lock:
            return self.cam


# ------------------------------------------------------------------ 전체화면
def make_display(full):
    """실제 창(또는 전체화면)을 만든다. 그림은 항상 1280x720 캔버스에 그린 뒤 맞춰서 키운다."""
    if full:
        display = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)   # 모니터 해상도 그대로
    else:
        display = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.mouse.set_visible(not full)
    return display


def present(display, canvas):
    """캔버스를 화면 크기에 맞게(비율 유지, 남는 곳은 검은 띠) 키워서 보여준다."""
    dw, dh = display.get_size()
    if (dw, dh) == (SCREEN_W, SCREEN_H):
        display.blit(canvas, (0, 0))
    else:
        s = min(dw / SCREEN_W, dh / SCREEN_H)
        w, h = int(SCREEN_W * s), int(SCREEN_H * s)
        scale = pygame.transform.smoothscale if SMOOTH_SCALE else pygame.transform.scale
        display.fill((0, 0, 0))
        display.blit(scale(canvas, (w, h)), ((dw - w) // 2, (dh - h) // 2))
    pygame.display.flip()


def draw_divider(screen):
    """가운데 세로 점선"""
    x = PANEL_W
    for y in range(0, SCREEN_H, 28):
        pygame.draw.line(screen, DIVIDER_COLOR, (x, y), (x, min(SCREEN_H, y + 16)), 3)


# ------------------------------------------------------------------ 실행
def run(path):
    global CHAR_ON_LEFT, FULLSCREEN
    pygame.init()
    display = make_display(FULLSCREEN)
    screen = pygame.Surface((SCREEN_W, SCREEN_H))         # 모든 그림은 여기에 그림 (좌표는 항상 1280x720 기준)
    pygame.display.set_caption("캐릭터와 나 - 나란히 같은 동작 (side by side)")
    clock = pygame.time.Clock()
    font = game.load_font(18)
    big = game.load_font(26)

    print("캐릭터 그림 사용:", path)
    rgba = light.load_character_rgba(path)
    ratios = light.CUSTOM_POINTS.get(os.path.basename(path))
    if ratios is not None:
        pts = light.complete_points(light.custom_points(ratios, rgba[:, :, 3]))
    else:
        pts = light.auto_points(rgba[:, :, 3])
    game.find_character_points = lambda alpha: pts
    mesh = light.OutlineMesh(rgba, step=light.MESH_STEP)
    if light.FIX_HEAD or light.FIX_HANDS_FEET:
        mesh.fix_parts(head=light.FIX_HEAD, hands_feet=light.FIX_HANDS_FEET)
    mesh.arms_on_top()
    rest = mesh.rest_pts
    hand_vertex = {s: mesh.handle_idx[light.HANDLES.index("hand_" + s)] for s in "lr"}

    tracker = SideTracker()
    char_x, cam_x = panels(CHAR_ON_LEFT)
    tracker.cam_x = cam_x
    if tracker.ok:
        tracker.start()
        print("캠 인식 시작: 캠 앞에 서서 움직여 보세요.")
    else:
        print("캠을 사용할 수 없습니다:", tracker.error)

    half_range = max(0.0, (PANEL_W - mesh.w) / 2)           # 캐릭터 칸 안에서만 좌우 이동
    offset = 0.0                                             # 칸 가운데에서 얼마나 옆으로 갔는지
    origin = np.array([char_x + (PANEL_W - mesh.w) / 2, SCREEN_H - mesh.h - GROUND_GAP], float)
    ref_cx = light.MOVE_REF
    targets = light.pose_to_targets(None, rest)
    jump = light.JumpTracker()
    torso_char = np.linalg.norm(rest["neck"] - rest["hip"])
    leg_char = max(rest["foot_l"][1], rest["foot_r"][1]) - rest["hip"][1]
    lift_now = 0.0

    ch = sam.ArapCharacter(mesh, rest)
    magic = sam.ArapMagic(SCREEN_W, SCREEN_H, ch)
    magic.follow = False
    lights = sam.make_lights(magic.theme)
    my_lights = sam.make_lights(magic.theme)               # 캠 속 내 검지 끝 불빛
    help_img = sam.help_surface(font)
    show_mesh, show_help = False, sam.SHOW_HELP
    prev_c = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key in (pygame.K_f, pygame.K_F11):  # 전체화면 켜기/끄기
                    FULLSCREEN = not FULLSCREEN
                    display = make_display(FULLSCREEN)
                elif event.key == pygame.K_d:
                    show_mesh = not show_mesh
                elif event.key == pygame.K_h:
                    show_help = not show_help
                elif event.key == pygame.K_x:
                    magic.clear()
                elif event.key == pygame.K_t:
                    magic.theme = (magic.theme + 1) % len(fe.THEMES)
                    lights = sam.make_lights(magic.theme)
                    my_lights = sam.make_lights(magic.theme)
                elif event.key == pygame.K_l:                  # 캐릭터 ↔ 나 자리 바꾸기
                    CHAR_ON_LEFT = not CHAR_ON_LEFT
                    char_x, cam_x = panels(CHAR_ON_LEFT)
                    tracker.cam_x = cam_x
                    magic.clear()
                elif event.key == pygame.K_c and tracker.get()[0] is not None:
                    ref_cx = tracker.get()[0]["cx"]
                    print(f"기준 위치를 다시 잡았습니다: {ref_cx:.2f}")

        pose, light_on, point_dir, _ = tracker.get()
        hands = tracker.get_hands()
        cam = tracker.get_cam()
        tracking = pose is not None and time.monotonic() - pose["time"] < game.LOST_TIME

        # 좌우 이동: 캐릭터 칸 안에서만
        if light.FOLLOW_X and tracking:
            t = game.clamp((pose["cx"] - ref_cx) / light.MOVE_RANGE, -1.0, 1.0)
            offset += (t * half_range - offset) * light.MOVE_SMOOTH
        origin[0] = char_x + (PANEL_W - mesh.w) / 2 + offset

        # 캐릭터 동작 (ARAP)
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

        # 손 모양 특수효과
        hl = list(hands.values())
        if len(hl) > 1 and prev_c is not None:
            hl.sort(key=lambda a: float(np.linalg.norm(a[[0, 5, 9, 13, 17]].mean(0) - prev_c)))
        hp = hl[0] if hl else None
        hp2 = hl[1] if len(hl) > 1 else None
        prev_c = hp[[0, 5, 9, 13, 17]].mean(0) if hp is not None else None
        magic.update(hp, hp2)

        # 검지 불빛: 캐릭터 손 + (켜 두면) 캠 속 내 검지 끝
        for s in "lr":
            h = hands.get(s)
            on = light_on[s] and h is not None and fe.classify(h) == "POINT" and magic.gesture != "HEART2"
            hand = points[hand_vertex[s]] + origin
            d = point_dir[s]
            lights[s].update(on, hand + d * light.REACH * light.CHAR_HEIGHT if d is not None else hand)
            if MY_FINGER_LIGHT:
                my_lights[s].update(on, h[8] if h is not None else None)

        # 그리기: 캠 칸 → 캐릭터 칸 → 특수효과 → 불빛 → 점선
        screen.fill(light.BG_COLOR)
        if cam is not None:
            screen.blit(pygame.image.frombuffer(cam, (PANEL_W, SCREEN_H), "RGB"), (cam_x, 0))
        else:
            msg = big.render("캠 화면 준비 중...", True, (150, 150, 150))
            screen.blit(msg, (cam_x + (PANEL_W - msg.get_width()) // 2, SCREEN_H // 2))
        screen.set_clip(pygame.Rect(char_x, 0, PANEL_W, SCREEN_H))      # 캐릭터가 캠 칸으로 넘어가지 않게
        screen.blit(char_surface, (int(origin[0]) - mesh.pad, int(origin[1]) - mesh.pad))
        screen.set_clip(None)
        fx = sam.magic_surface(magic)
        if fx is not None:
            screen.blit(fx, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        for s in "lr":
            lights[s].draw(screen)
            if MY_FINGER_LIGHT:
                my_lights[s].draw(screen)
        draw_divider(screen)

        if show_mesh:
            sp = points + origin
            for a, b in mesh.edges:
                pygame.draw.line(screen, (0, 220, 255), sp[a], sp[b], 1)
            for p in targets:
                pygame.draw.circle(screen, (255, 230, 0), (p + origin).astype(int), 5)

        if LABELS:
            for text, x in (("캐릭터", char_x), ("나", cam_x)):
                t = big.render(text, True, (240, 240, 240))
                screen.blit(t, (x + PANEL_W - t.get_width() - 16, 12))
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
        screen.blit(font.render(status + "   |   L 좌우 바꾸기  F 전체화면  H 도움말  ESC 스캐너로", True, (190, 190, 190)),
                    (10, SCREEN_H - 28))

        present(display, screen)
        clock.tick(sam.FPS)

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
    print("=== 캐릭터와 나, 나란히 같은 동작 ===")
    print("[SPACE] 3초 스캔   [Y] 이 그림으로 시작   [N] 다시 스캔   [G] 최근 그림   [Q] 종료")
    print("실행 중 [ESC] 를 누르면 스캐너로 돌아옵니다.  (L 좌우 바꾸기, F/F11 전체화면, H 도움말)")
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