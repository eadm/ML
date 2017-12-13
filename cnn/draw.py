import cv2
import numpy as np

pressed = False


# mouse callback function
def draw_circle(event, x, y, flags, param):
    global pressed

    if event == cv2.EVENT_LBUTTONDOWN:
        pressed = True

    if event == cv2.EVENT_LBUTTONUP:
        pressed = False

    if pressed:
        cv2.circle(img, (x, y), 3, (255, 255, 255), -1)


# Create a black image, a window and bind the function to window
img = np.zeros((84, 84, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while 1:
    cv2.imshow('image', img)
    if cv2.waitKey(20) & 0xFF == 27:
        break
cv2.destroyAllWindows()
