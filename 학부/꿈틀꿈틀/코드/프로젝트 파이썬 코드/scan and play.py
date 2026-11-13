"""
scan_and_play.py  -  그림 스캔 → 바로 공 피하기 게임

[흐름]
  1) 캠에 그림을 비추고 SPACE → 5초 스캔
  2) 미리보기를 보고 Y → 저장하고 바로 게임 시작 / N → 다시 스캔
  3) 게임에서 ESC → 스캐너로 돌아옴 (다른 그림으로 다시 할 수 있음)

[준비]
  이 파일을 arap_dodge_game.py 와 같은 폴더에 두고 실행합니다.
  설치:  pip install pygame numpy opencv-python mediapipe
  실행:  python scan_and_play.py

[스캐너 조작]
  SPACE  5초 스캔 시작
  Y / N  (스캔 후) 이 그림으로 게임 시작 / 다시 스캔
  G      스캔 없이, 가장 최근에 저장한 캐릭터로 바로 게임
  Q      프로그램 종료

[그림 팁]
  게임이 그림 모양만 보고 머리/손/발 위치를 잡기 때문에,
  "정면을 보고 팔을 양옆으로 벌리고 선" 캐릭터가 가장 잘 움직입니다.
"""
import os
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
try:
    import arap_dodge_game as game
except ImportError as e:
    raise SystemExit(f"arap_dodge_game.py 를 불러오지 못했습니다: {e}\n"
                     "이 파일과 같은 폴더에 arap_dodge_game.py 가 있는지, "
                     "pygame 이 설치되어 있는지 확인하세요.")

# ------------------------------------------------------------------ 설정값
SAVE_DIR = os.path.join(BASE_DIR, "saved_characters")
SCAN_SECONDS = 5
BOX_SIZE = 400
# 화면은 거울처럼 좌우 반전해서 보여주므로, 그대로 저장하면 그림이 뒤집혀 저장된다.
# True면 저장할 때 다시 뒤집어서 종이에 그린 방향 그대로 저장한다.
UNMIRROR_SAVED = True

YELLOW, GREEN, RED, ORANGE = (0, 255, 255), (0, 255, 0), (0, 0, 255), (0, 140, 255)


# =========================================================================
# 1. 캐릭터 외곽선 실시간 감지 함수  (기존 스캐너 코드 그대로)
# =========================================================================
def find_valid_character(img):
    """박스 안에 진짜 캐릭터 그림(선으로 둘러싸인 면적)이 있는지 검사."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) 천장이나 어두운 방 배경 차단 (종이는 기본적으로 밝음)
    if np.mean(gray) < 80:
        return None, None

    # 2) 이진화 (검은 펜 선 추출)
    black_white = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 19, 9)

    # 3) 박스 가장자리 10px 차단 (상자 테두리선 간섭 제거)
    h, w = black_white.shape
    black_white[0:10, :] = 0
    black_white[h-10:h, :] = 0
    black_white[:, 0:10] = 0
    black_white[:, w-10:w] = 0

    # 4) 선 팽창 (끊긴 펜선 연결)
    connect_tool = np.ones((3, 3), np.uint8)
    black_white = cv2.dilate(black_white, connect_tool, iterations=2)

    # 5) 외곽선 탐색
    all_outlines, _ = cv2.findContours(black_white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not all_outlines:
        return None, None

    main_contour = max(all_outlines, key=cv2.contourArea)

    # 면적이 1200px 이상인 '진짜 그림' 형태일 때만 감지 성공으로 인정
    if cv2.contourArea(main_contour) >= 1200:
        return main_contour, gray

    return None, None


# =========================================================================
# 2. 캐릭터 추출 및 15px 안전 패딩 크롭 함수  (기존 스캐너 코드 그대로)
# =========================================================================
def extract_character(img, main_contour, gray):
    """15px 여유(Padding)를 주어 펜선 잘림을 막고, texture와 mask 동시 생성"""
    alpha_mask = np.zeros_like(gray)
    cv2.drawContours(alpha_mask, [main_contour], -1, 255, thickness=cv2.FILLED)

    bx, by, bw, bh = cv2.boundingRect(main_contour)

    pad = 15
    h, w = img.shape[:2]
    x1 = max(0, bx - pad)
    y1 = max(0, by - pad)
    x2 = min(w, bx + bw + pad)
    y2 = min(h, by + bh + pad)

    blue, green, red = cv2.split(img)
    result_png = cv2.merge([blue, green, red, alpha_mask])

    cropped_png = result_png[y1:y2, x1:x2]
    cropped_mask = alpha_mask[y1:y2, x1:x2]

    return cropped_png, cropped_mask


# =========================================================================
# 3. 저장 / 불러오기
# =========================================================================
def save_png(path, image):
    """폴더 경로에 한글이 있어도 저장되도록 cv2.imwrite 대신 직접 파일로 쓴다."""
    ok, data = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 변환 실패: {path}")
    data.tofile(path)


def save_character(texture, mask):
    """saved_characters/character_날짜_시간/ 에 texture.png, mask.png 저장 후 texture 경로 반환"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(SAVE_DIR, f"character_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    texture_path = os.path.join(folder, "texture.png")
    save_png(texture_path, texture)                       # 투명 배경 컬러 그림
    save_png(os.path.join(folder, "mask.png"), mask)      # 흑백 실루엣
    print(f"--> [저장 완료] '{folder}'")
    return texture_path


def latest_saved_character():
    """가장 최근에 저장한 texture.png 경로 (없으면 None)"""
    if not os.path.isdir(SAVE_DIR):
        return None
    for name in sorted(os.listdir(SAVE_DIR), reverse=True):
        path = os.path.join(SAVE_DIR, name, "texture.png")
        if os.path.exists(path):
            return path
    return None


# =========================================================================
# 4. 스캐너: 게임에 쓸 texture.png 경로를 돌려준다 (Q로 끝내면 None)
# =========================================================================
def scan_character():
    camera = game.open_camera()
    if camera is None:
        print("웹캠을 열 수 없습니다 (다른 프로그램이 사용 중이거나 권한이 없을 수 있음)")
        return None

    start_time = None      # 스캔을 시작한 시각 (None이면 스캔 중이 아님)
    texture = mask = None  # 스캔 결과 (None이 아니면 Y/N 선택 대기 중)
    preview = None
    warn_until = 0.0       # 이 시각까지 "그림 없음" 경고를 보여줌

    def label(frame, text, color, x, y):
        cv2.putText(frame, text, (max(5, x - 20), y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    try:
        while True:
            is_success, frame = camera.read()
            if not is_success or frame is None:
                print("캠에서 화면을 읽지 못했습니다.")
                return None

            frame = cv2.flip(frame, 1)  # 좌우 반전(거울처럼 보이게)
            height, width = frame.shape[:2]
            box = min(BOX_SIZE, height - 60, width - 20)
            x1, y1 = (width - box) // 2, (height - box) // 2
            x2, y2 = x1 + box, y1 + box
            inside_the_box = frame[y1:y2, x1:x2]

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q')):
                return None

            # ---------------------------------------------------------
            # [상태 1] 스캔 완료 → 이 그림으로 게임할지 선택 (노란색 상자)
            # ---------------------------------------------------------
            if texture is not None:
                color, text = YELLOW, "PLAY? [Y] / [N]"
                if key in (ord('y'), ord('Y')):
                    return save_character(texture, mask)
                if key in (ord('n'), ord('N')):
                    print("--> [취소] 저장하지 않고 다시 스캔합니다.")
                    texture = mask = preview = None
                    cv2.destroyWindow("Preview")

            # ---------------------------------------------------------
            # [상태 2] 스캔 대기 / 타이머 진행
            # ---------------------------------------------------------
            else:
                if key == 32:                              # SPACE: 스캔 시작
                    start_time = time.time()
                    warn_until = 0.0
                elif key in (ord('g'), ord('G')):          # G: 최근 저장한 캐릭터로 바로 게임
                    last = latest_saved_character()
                    if last is not None:
                        return last
                    print("--> 저장된 캐릭터가 없습니다. 먼저 스캔하세요.")

                if start_time is not None:
                    contour, gray = find_valid_character(inside_the_box)
                    if contour is None:                    # 그림이 없거나 박스를 벗어남
                        start_time = None
                        warn_until = time.time() + 1.5
                    else:
                        passed = time.time() - start_time
                        if passed >= SCAN_SECONDS:         # 5초 경과 → 추출
                            texture, mask = extract_character(inside_the_box, contour, gray)
                            if UNMIRROR_SAVED:
                                texture, mask = cv2.flip(texture, 1), cv2.flip(mask, 1)
                            preview = np.full(texture.shape[:2] + (3,), 255, np.uint8)
                            shown = texture[:, :, 3] > 0
                            preview[shown] = texture[:, :, :3][shown]
                            start_time = None

                if texture is not None:
                    color, text = YELLOW, "PLAY? [Y] / [N]"
                elif start_time is not None:
                    left = max(0, SCAN_SECONDS - int(time.time() - start_time))
                    color, text = GREEN, f"Scanning... {left}s"
                elif time.time() < warn_until:
                    color, text = RED, "No Drawing Detected!"
                else:
                    color, text = ORANGE, "[SPACE] Scan  [G] Last  [Q] Quit"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label(frame, text, color, x1, y1)
            cv2.imshow("Character Scanner", frame)
            if preview is not None:
                cv2.imshow("Preview", preview)
    finally:
        # 게임이 캠을 쓸 수 있도록 반드시 캠을 놓아준다
        camera.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)


# =========================================================================
# 5. 게임 실행: arap_dodge_game 이 방금 스캔한 그림을 쓰도록 연결
# =========================================================================
def play(texture_path):
    game.find_character_png = lambda: texture_path   # character.png 대신 스캔한 그림 사용
    try:
        game.main()
    except SystemExit as e:
        print(e)
    except Exception:
        traceback.print_exc()
        print("--> 이 그림으로 게임을 시작하지 못했습니다. 다시 스캔해 보세요.")
        game.pygame.quit()


def main():
    print("=== 그림 스캔 → 공 피하기 게임 ===")
    print("[SPACE] 5초 스캔   [Y] 이 그림으로 게임 시작   [N] 다시 스캔")
    print("[G] 최근 저장한 캐릭터로 바로 게임   [Q] 종료")
    print("게임 중 [ESC] 를 누르면 스캐너로 돌아옵니다.")
    while True:
        texture_path = scan_character()
        if texture_path is None:
            break
        time.sleep(0.3)                # 캠이 완전히 풀릴 때까지 잠깐 대기
        print("게임 시작:", texture_path)
        play(texture_path)
        time.sleep(0.3)


if __name__ == "__main__":
    main()