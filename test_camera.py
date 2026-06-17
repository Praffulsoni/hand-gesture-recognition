import cv2
import mediapipe as mp
import joblib
import math
import time
import pygame
from collections import deque, Counter
import numpy as np

# ==========================================
# PYGAME UI INITIALIZATION (NATIVE FULLSCREEN)
# ==========================================
pygame.init()
pygame.font.init()

# Fetch your monitor's exact current pixel resolution
infoObject = pygame.display.Info()
WIDTH, HEIGHT = infoObject.current_w, infoObject.current_h

# Open an edge-to-edge, hardware-accelerated, double-buffered fullscreen window
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN | pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Aura - Advanced Gesture Interface")
clock = pygame.time.Clock()

# Premium Color Palette (RGBA/RGB)
BG_CARD = (25, 28, 36, 180)       # Translucent deep slate dark
ACCENT_CYAN = (78, 205, 196)     # Soft mint/cyan
ACCENT_PURPLE = (142, 68, 173)   # Elegant dashboard purple
TEXT_MAIN = (245, 246, 250)      # Clean off-white
TEXT_MUTED = (149, 165, 166)     # Soft charcoal gray
COLOR_LIVE = (46, 204, 113)      # Vibrant emerald green

# Dynamic Skeleton Interface Colors
NODE_ORANGE = (255, 140, 0)      # High-visibility structural orange
BONE_BLACK = (15, 15, 15)        # Sharp solid black for lines

# Anti-Aliased System Typography
FONT_MAIN = pygame.font.SysFont("Helvetica", 42, bold=True)
FONT_SUB = pygame.font.SysFont("Helvetica", 18)
FONT_TITLE = pygame.font.SysFont("Helvetica", 14, bold=True)

# ==========================================
# MACHINE LEARNING & MEDIAPIPE CONFIG
# ==========================================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.7
)

# Load your high-accuracy champion model
model = joblib.load("champion_model.pkl")

# Temporal Filtering / Prediction Buffers
BUFFER_SIZE = 12
prediction_buffer = deque(maxlen=BUFFER_SIZE)
history_log = deque(maxlen=3)
last_added_gesture = None

# ==========================================
# PROCESSING HELPER FUNCTIONS
# ==========================================
def normalize_live_landmarks(landmarks_list):
    """
    Translates raw frame coordinates to match training dimensions.
    Zero-centers around the wrist and scales by the maximum extension distance.
    """
    raw_coords = [[lm.x, lm.y] for lm in landmarks_list]
    wrist_x, wrist_y = raw_coords[0]
    
    centered = []
    for x, y in raw_coords:
        centered.append([x - wrist_x, y - wrist_y])
        
    max_dist = 0
    for x, y in centered:
        dist = math.sqrt(x**2 + y**2)
        if dist > max_dist:
            max_dist = dist
            
    if max_dist == 0: 
        max_dist = 1
        
    normalized_features = []
    for x, y in centered:
        normalized_features.append(x / max_dist)
        normalized_features.append(y / max_dist)
        
    return normalized_features

def draw_blur_card(surface, x, y, w, h, radius=16, color=BG_CARD):
    """Draws a premium rounded UI block container with clean alpha opacity."""
    rect_surface = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(rect_surface, color, (0, 0, w, h), border_radius=radius)
    surface.blit(rect_surface, (x, y))

# ==========================================
# HD HARDWARE ENGINE INITIALIZATION
# ==========================================
chosen_index = 1 
print(f"\n🚀 Forcing initialization of Camo/Iriun via MSMF Engine at Index {chosen_index}...")
cap = cv2.VideoCapture(chosen_index, cv2.CAP_MSMF)

# Enforce hardware query settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

success, test_frame = cap.read()
if not success:
    print("⚠️ External feed index unavailable. Dropping to base device hardware Index 0...")
    cap.release()
    chosen_index = 0
    cap = cv2.VideoCapture(chosen_index, cv2.CAP_MSMF)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    print("✔️ HD Stream hooked and broadcast stabilized successfully!")

# ==========================================
# MAIN REAL-TIME PROCESSING LOOP
# ==========================================
running = True

while running:
    # Safely handle OS window events (Press ESC key to exit fullscreen smoothly)
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            running = False

    success, frame = cap.read()
    if not success: 
        print("⚠️ Camera matrix frame drop detected.")
        break

    # 1. Process clean, raw coordinates through MediaPipe first
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # 2. Mirror video frame horizontally and cleanly scale it up to true monitor fullscreen
    flipped_rgb = cv2.flip(rgb, 1)
    raw_surface = pygame.image.frombuffer(flipped_rgb.tobytes(), flipped_rgb.shape[1::-1], "RGB")
    frame_surface = pygame.transform.smoothscale(raw_surface, (WIDTH, HEIGHT))

    smoothed_prediction = "STANDBY"
    buffer_strength_percent = 0
    hand_pts = {} 

    # 3. Extract and project tracking positions onto fullscreen space
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for idx, lm in enumerate(hand_landmarks.landmark):
                # Invert screen x mapping ratio to account for the display canvas mirroring
                cx = int((1.0 - lm.x) * WIDTH)
                cy = int(lm.y * HEIGHT)
                hand_pts[idx] = (cx, cy)

            # Inference pipeline
            features = normalize_live_landmarks(hand_landmarks.landmark)
            raw_prediction = model.predict([features])[0]
            prediction_buffer.append(raw_prediction)

            # Temporal Smoothing Majority Vote
            buffer_counts = Counter(prediction_buffer)
            smoothed_prediction, confidence_count = buffer_counts.most_common(1)[0]
            buffer_strength_percent = int((confidence_count / BUFFER_SIZE) * 100)

            # Update timeline history array log
            if smoothed_prediction != last_added_gesture:
                history_log.appendleft(smoothed_prediction)
                last_added_gesture = smoothed_prediction
    else:
        if len(prediction_buffer) > 0:
            prediction_buffer.popleft()

    # ==========================================
    # CORE INTERFACE RENDERING ENGINE
    # ==========================================
    # Draw high-definition camera stream background surface
    screen.blit(frame_surface, (0, 0))

    # Render hand structural skeleton tracking overlays
    if hand_pts:
        # Draw connections first so they sit cleanly behind joint node layers
        for connection in mp_hands.HAND_CONNECTIONS:
            start_idx, end_idx = connection
            if start_idx in hand_pts and end_idx in hand_pts:
                pygame.draw.line(screen, BONE_BLACK, hand_pts[start_idx], hand_pts[end_idx], width=4)

        # Draw structural joint nodes
        for idx, pt in hand_pts.items():
            pygame.draw.circle(screen, NODE_ORANGE, pt, 6)           
            pygame.draw.circle(screen, (255, 255, 255), pt, 2)       

    # CARD A: SYSTEM METRICS LAYER (Top Left)
    draw_blur_card(screen, 30, 30, 260, 90, radius=12)
    screen.blit(FONT_TITLE.render("SYSTEM CORE", True, ACCENT_CYAN), (45, 42))
    fps = int(clock.get_fps())
    screen.blit(FONT_SUB.render(f"ENGINE FPS: {fps}", True, TEXT_MAIN), (45, 65))
    
    status_color = COLOR_LIVE if results.multi_hand_landmarks else TEXT_MUTED
    status_text = "TRACKING" if results.multi_hand_landmarks else "IDLE"
    pygame.draw.circle(screen, status_color, (245, 48), 6)
    screen.blit(FONT_SUB.render(status_text, True, status_color), (180, 38))

    # CARD B: SIGNAL STABILITY DISPLAY METER (Top Right)
    draw_blur_card(screen, WIDTH - 290, 30, 260, 90, radius=12)
    screen.blit(FONT_TITLE.render("SIGNAL STABILITY", True, ACCENT_CYAN), (WIDTH - 275, 42))
    screen.blit(FONT_SUB.render(f"{buffer_strength_percent}%", True, TEXT_MAIN), (WIDTH - 275, 65))
    pygame.draw.rect(screen, (60, 64, 73), (WIDTH - 190, 70, 140, 10), border_radius=5)
    if buffer_strength_percent > 0:
        fill_w = int((buffer_strength_percent / 100) * 140)
        pygame.draw.rect(screen, ACCENT_CYAN, (WIDTH - 190, 70, fill_w, 10), border_radius=5)

    # CARD C: RECENT DETECTIONS LOG CHANNELS (Bottom Left)
    draw_blur_card(screen, 30, HEIGHT - 180, 260, 150, radius=14)
    screen.blit(FONT_TITLE.render("RECENT DETECTIONS", True, TEXT_MUTED), (45, HEIGHT - 162))
    for i, entry in enumerate(history_log):
        log_surface = FONT_SUB.render(f"› {entry.lower()}", True, TEXT_MAIN if i == 0 else TEXT_MUTED)
        screen.blit(log_surface, (45, (HEIGHT - 130) + (i * 26)))

    # CARD D: MAIN CENTERED PREDICTION BANNER (Bottom Center)
    banner_w, banner_h = 560, 95
    banner_x = (WIDTH - banner_w) // 2
    banner_y = HEIGHT - banner_h - 35
    
    draw_blur_card(screen, banner_x, banner_y, banner_w, banner_h, radius=20, color=(15, 18, 26, 210))
    pygame.draw.rect(screen, ACCENT_PURPLE, (banner_x, banner_y, banner_w, banner_h), width=2, border_radius=20)
    
    pred_string = smoothed_prediction.upper()
    pred_render = FONT_MAIN.render(pred_string, True, TEXT_MAIN if results.multi_hand_landmarks else ACCENT_PURPLE)
    pred_rect = pred_render.get_rect(center=(WIDTH // 2, banner_y + 48))
    screen.blit(pred_render, pred_rect)

    # Refresh Render Framework Display Layers
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()