# importing the necessary libraries
import fitz  # PyMuPDF
import numpy as np
import cv2
import mediapipe as mp
import time
import warnings
import speech_recognition as sr
import threading
import re
from collections import deque

warnings.filterwarnings("ignore")  # Ignore warnings

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.7)
face_mesh = mp_face.FaceMesh(min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# PDF setup
doc = fitz.open(r"D:\PROJECTS BY ME\Gesture pdf\Tutorial_EDIT.pdf")
current_page = 0
total_pages = len(doc)

# Camera setup
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit()

# State variables
last_scroll_time = time.time()
prev_index_y = None  # Track finger movement
prev_nose_y = None  # Track nose movement
prev_pinch_dis = None  # Track pinch distance

zoom_level = 1.0  # Initial zoom factor
min_zoom = 0.5
max_zoom = 5.0
zoom_threshold = 0.005

# swipe gesture
swipe_history = deque(maxlen=5)
swipe_threshold_ = 0.2
swipe_cooldown = 1.0
last_swipe = 0

# hint overlay
hint_text = "👋 Swipe, Nod, or Speak to control pages"
alpha = 0  # Fade value (0 invisible - 255 fully visible)
fade_speed = 10
show_hint = True
last_action_time = time.time()
hint_timeout = 5  # seconds of inactivity before showing the hint again

_guidance_lines = [
    "Gestures:",
    "Index Up: Scroll",
    "Swipe Left / Right: Prev / Next page",
    "Pinch (thumb+index): Zoom in/out",
    "Nod head: Scroll up/down",
    "",
    "Voice Commands:",
    "'next', 'previous', 'go to page 3'",
    "'zoom in', 'zoom out', 'reset zoom'"
]

# transient (action) hint
transient_text = ""
transient_text_timer = 0
transient_duration = 2.0  # seconds
transient_start = 0  # ✅ Added initialization

# font and styling
_overlay_font = cv2.FONT_HERSHEY_SIMPLEX
_overlay_line_height = 22


def _trigger_action_hint(msg):
    """Show transient action hint (bottom-center) and hide guidance box"""
    global transient_text, transient_start, show_hint, last_action_time, hint_text, alpha
    transient_text = msg
    transient_start = time.time()
    show_hint = False
    last_action_time = time.time()
    hint_text = msg


def _draw_overlays(img):
    """ Draws guidance box (top-left) with fade and transient hints (bottom-center). """
    global alpha, show_hint, last_action_time, hint_timeout, fade_speed
    h, w = img.shape[:2]

    # Auto fade in/out for guidance
    if time.time() - last_action_time > hint_timeout:
        show_hint = True
    if show_hint:
        alpha = min(255, alpha + fade_speed)
    else:
        alpha = max(0, alpha - fade_speed)

    # Draw guidance box
    if alpha > 5:
        box_w = 480
        box_x, box_y = 20, 18
        lines_count = len(_guidance_lines)
        box_h = 22 * lines_count + 20
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y),
                      (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha/255.0, img,
                        1 - alpha/255.0, 0, img)
        text_x = box_x + 12
        for i, line in enumerate(_guidance_lines):
            y = box_y + 20 + i * _overlay_line_height
            cv2.putText(img, line, (text_x, y), _overlay_font,
                        0.6, (230, 230, 230), 1, cv2.LINE_AA)

    # Transient action hint
    global transient_text, transient_start, transient_duration
    if transient_text:
        elapsed = time.time() - transient_start
        if elapsed < transient_duration:
            t = 1.0 - (elapsed / transient_duration)
            t = max(0.0, min(1.0, t))
            overlay = img.copy()
            txt = transient_text
            (tw, th), _ = cv2.getTextSize(
                txt, _overlay_font, 1.0, 2)
            txt_x = int((w - tw) / 2)
            txt_y = h - 60
            pad_x, pad_y = 18, 14
            cv2.rectangle(overlay, (txt_x - pad_x, txt_y - th - pad_y),
                          (txt_x + tw + pad_x, txt_y + pad_y // 2),
                          (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6 * t, img,
                            1 - 0.6 * t, 0, img)
            cv2.putText(img, txt, (txt_x, txt_y),
                        _overlay_font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, txt, (txt_x, txt_y),
                        _overlay_font, 1.0, (180, 255, 180), 2, cv2.LINE_AA)
        else:
            transient_text = ""
    return img


def swipe_detect(current_x, current_time):
    global last_swipe
    swipe_history.append((current_x, current_time))
    if len(swipe_history) < swipe_history.maxlen:
        return None
    x0, t0 = swipe_history[0]
    x1, t1 = swipe_history[-1]
    if (t1-last_swipe) < swipe_cooldown:
        return None
    delta_x = x1 - x0
    delta_t = t1 - t0
    velocity = delta_x/delta_t if delta_t != 0 else 0
    if abs(velocity) > swipe_threshold_:
        last_swipe = t1
        return "right" if velocity > 0 else "left"
    return None


def render_pdf_page(doc, page_number, zoom=1.0):
    page = doc.load_page(page_number)
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
        pix.height, pix.width, 3)
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def listen_for_voice():
    global current_page, zoom_level, last_scroll_time
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    while True:
        with mic as source:
            print("🎤 Listening for voice command...")
            audio = recognizer.listen(source, phrase_time_limit=8)
        try:
            command = recognizer.recognize_google(audio).lower().strip()
            print(f"🗣️ Command received: {command}")
            now = time.time()

            # Page navigation
            page_match = re.search(
                r"(?:go to|page number|open page|page)\s*(\d+)", command)
            if page_match:
                page_num = int(page_match.group(1)) - 1
                if 0 <= page_num < total_pages:
                    current_page = page_num
                    print(f"📄 Navigating to page {page_num + 1}")
                    continue
                else:
                    print(
                        f"⚠️ Invalid page number: {page_num + 1}. Valid range is 1 to {total_pages}.")

            # Keyword-based commands
            if any(k in command for k in ["next", "next page", "scroll down", "down"]):
                if now - last_scroll_time > 1:
                    current_page = min(total_pages-1, current_page + 1)
                    last_scroll_time = now
            elif any(k in command for k in ["previous", "previous page", "scroll up", "up", "back"]):
                if now - last_scroll_time > 1:
                    current_page = max(0, current_page - 1)
                    last_scroll_time = now
            elif any(k in command for k in ["bigger", "zoom in"]):
                zoom_level = min(max_zoom, zoom_level + 0.5)
            elif any(k in command for k in ["smaller", "zoom out"]):
                zoom_level = max(min_zoom, zoom_level - 0.5)
            elif "reset zoom" in command:
                zoom_level = 1.0
        except sr.UnknownValueError:
            print("❓ Could not understand the command.")
        except sr.RequestError as e:
            print(f"⚠️ Voice recognition error {e}")


# Run voice in background
voice_thread = threading.Thread(target=listen_for_voice, daemon=True)
voice_thread.start()

_prev_page_for_overlay = current_page
_prev_zoom_for_overlay = zoom_level

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hand_result = hands.process(img_rgb)
        face_result = face_mesh.process(img_rgb)
        now = time.time()

        # Head nod detection
        if face_result.multi_face_landmarks:
            for face_landmarks in face_result.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_landmarks, mp_face.FACEMESH_TESSELATION)
                nose_y = face_landmarks.landmark[1].y
                if prev_nose_y is not None and now - last_scroll_time > 1:
                    delta_nose = prev_nose_y - nose_y
                    if delta_nose > 0.03:  # nod up
                        current_page = max(0, current_page - 1)
                        last_scroll_time = now
                    elif delta_nose < -0.03:  # nod down
                        current_page = min(total_pages - 1, current_page + 1)
                        last_scroll_time = now
                prev_nose_y = nose_y

        # Hand gesture detection
        if hand_result.multi_hand_landmarks:
            for hand_landmarks in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Finger positions
                index_tip_y = hand_landmarks.landmark[8].y
                index_pip_y = hand_landmarks.landmark[6].y
                middle_tip_y = hand_landmarks.landmark[12].y
                middle_pip_y = hand_landmarks.landmark[10].y
                ring_tip_y = hand_landmarks.landmark[16].y
                ring_pip_y = hand_landmarks.landmark[14].y
                pinky_tip_y = hand_landmarks.landmark[20].y
                pinky_pip_y = hand_landmarks.landmark[18].y

                # Gesture conditions
                index_up = index_tip_y < index_pip_y - 0.05
                middle_down = middle_tip_y > middle_pip_y
                ring_down = ring_tip_y > ring_pip_y
                pinky_down = pinky_tip_y > pinky_pip_y

                if index_up and middle_down and ring_down and pinky_down:
                    if prev_index_y is not None and now - last_scroll_time > 1:
                        delta = prev_index_y - index_tip_y
                        if delta > 0.05:
                            current_page = max(0, current_page - 1)
                            last_scroll_time = now
                        elif delta < -0.05:
                            current_page = min(total_pages -
                                               1, current_page + 1)
                            last_scroll_time = now
                    prev_index_y = index_tip_y
                else:
                    prev_index_y = None

                # Pinch zoom
                thumb_tip = hand_landmarks.landmark[4]
                index_tip = hand_landmarks.landmark[8]
                pinch_dist = ((thumb_tip.x - index_tip.x)**2 +
                              (thumb_tip.y - index_tip.y)**2)**0.5
                if prev_pinch_dis is not None:
                    delta_pinch = pinch_dist - prev_pinch_dis
                    if abs(delta_pinch) > zoom_threshold:
                        if delta_pinch > 0:
                            zoom_level = min(max_zoom, zoom_level + 0.1)
                        else:
                            zoom_level = max(min_zoom, zoom_level - 0.1)
                prev_pinch_dis = pinch_dist

                # Swipe
                index_tip_x = hand_landmarks.landmark[8].x
                swipe_direction = swipe_detect(index_tip_x, now)
                if swipe_direction == "right":
                    current_page = min(total_pages - 1, current_page + 1)
                    print("👉 Swipe Right → Next Page")
                elif swipe_direction == "left":
                    current_page = max(0, current_page - 1)
                    print("👈 Swipe Left → Previous Page")

        # Render PDF + Webcam
        pdf_img = render_pdf_page(doc, current_page, zoom=zoom_level)
        pdf_img = cv2.resize(pdf_img, (1000, 1000))
        frame = cv2.resize(frame, (1000, 1000))
        combined = np.hstack((frame, pdf_img))

        # Overlay text
        cv2.putText(combined, f"Page: {current_page+1}/{total_pages}",
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (111, 180, 24), 2)
        cv2.putText(combined, f"Zoom: {zoom_level:.1f}x",
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (190, 210, 250), 2)

        # Overlay hints
        if current_page != _prev_page_for_overlay:
            if current_page > _prev_page_for_overlay:
                _trigger_action_hint(f"Swiped / Page → {current_page+1}")
            else:
                _trigger_action_hint(f"Page → {current_page+1}")
            _prev_page_for_overlay = current_page
        if abs(zoom_level - _prev_zoom_for_overlay) > 1e-3:
            _trigger_action_hint(f"Zoom: {zoom_level:.1f}x")
            _prev_zoom_for_overlay = zoom_level

        combined = _draw_overlays(combined)

        cv2.imshow("PDF Scroller using Gesture ", combined)
        if cv2.waitKey(1) & 0xFF == 27:
            break

except Exception as e:
    print(f"❌ Unhandled error in main loop: {e}")

cap.release()
cv2.destroyAllWindows()
