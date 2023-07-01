import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pygame

# Load the pre-trained TensorFlow model
model = tf.keras.models.load_model('gesture_model.h5')

# Initialize MediaPipe's Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Gesture labels
gesture_labels = {
    0: 'Left',
    1: 'Right',
}

# Initialize Pygame
pygame.init()
window_width = 800
window_height = 600
window = pygame.display.set_mode((window_width, window_height))
pygame.display.set_caption("Gesture-Based Game")

# Set up the player
player_width = 50
player_height = 50
player_x = window_width // 2 - player_width // 2
player_y = window_height - player_height - 10
player_speed = 5

# Set up the obstacle
obstacle_width = 50
obstacle_height = 50
obstacle_x = np.random.randint(0, window_width - obstacle_width)
obstacle_y = 0
obstacle_speed = 3

# Game over flag and text
game_over = False
game_over_text = pygame.font.Font('freesansbold.ttf', 64).render('Game Over', True, (255, 255, 255))

# Restart button
restart_button_text = pygame.font.Font('freesansbold.ttf', 32).render('Restart', True, (255, 255, 255))
restart_button_rect = restart_button_text.get_rect()
restart_button_rect.center = (window_width // 2, window_height // 2 + 50)

# Game loop
running = True
clock = pygame.time.Clock()

# Main loop
cap = cv2.VideoCapture(0)
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and game_over:
            mouse_pos = pygame.mouse.get_pos()
            if restart_button_rect.collidepoint(mouse_pos):
                # Restart the game
                game_over = False
                player_x = window_width // 2 - player_width // 2
                obstacle_x = np.random.randint(0, window_width - obstacle_width)
                obstacle_y = 0

    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Detect hand landmarks
    results = hands.process(frame)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the index finger landmarks
            index_finger_landmarks = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_finger_x = int(index_finger_landmarks.x * frame.shape[1])
            index_finger_y = int(index_finger_landmarks.y * frame.shape[0])

            # Recognize gesture based on index finger position
            if index_finger_x < frame.shape[1] // 3:
                gesture = 'Left'
                player_x -= player_speed
            elif index_finger_x > frame.shape[1] * 2 // 3:
                gesture = 'Right'
                player_x += player_speed
            else:
                gesture = 'Neutral'

            # Draw a circle at the index finger position
            cv2.circle(frame, (index_finger_x, index_finger_y), 10, (0, 255, 0), -1)

            # Draw the recognized gesture on the frame
            cv2.putText(frame, gesture, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Gesture Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if not game_over:
        # Update the player's position
        if player_x < 0:
            player_x = 0
        elif player_x > window_width - player_width:
            player_x = window_width - player_width

        # Update the obstacle's position
        obstacle_y += obstacle_speed

        # Check for collision between the player and the obstacle
        if player_x < obstacle_x + obstacle_width and \
                player_x + player_width > obstacle_x and \
                player_y < obstacle_y + obstacle_height and \
                player_y + player_height > obstacle_y:
            # Game over
            game_over = True

        # Reset the obstacle if it goes off the screen
        if obstacle_y > window_height:
            obstacle_x = np.random.randint(0, window_width - obstacle_width)
            obstacle_y = 0

        # Clear the window
        window.fill((0, 0, 0))

        # Draw the player
        pygame.draw.rect(window, (255, 255, 255), (player_x, player_y, player_width, player_height))

        # Draw the obstacle
        pygame.draw.rect(window, (255, 0, 0), (obstacle_x, obstacle_y, obstacle_width, obstacle_height))

    # Display "Game Over" text if game over
    if game_over:
        window.blit(game_over_text, (window_width // 2 - game_over_text.get_width() // 2, window_height // 2 - game_over_text.get_height() // 2))
        pygame.draw.rect(window, (0, 0, 255), restart_button_rect)
        window.blit(restart_button_text, restart_button_rect)

    # Update the display
    pygame.display.update()

    # Limit the frame rate
    clock.tick(60)

cap.release()
cv2.destroyAllWindows()
pygame.quit()