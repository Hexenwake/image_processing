import cv2
import numpy as np

image = cv2.imread("path to image")

image = cv2.resize(image, (600, 600))

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

line_intersects_contour = False
# Define the lower and upper threshold for human skin color
lower_color = np.array([0, 20, 70], dtype=np.uint8)
upper_color = np.array([20, 175, 255], dtype=np.uint8)


lower_porcelain = np.array([0, 0, 170], dtype=np.uint8)
upper_porcelain = np.array([180, 30, 255], dtype=np.uint8)

# Define the lower and upper bounds for lip color
lower_lips = np.array([150, 23, 70], dtype=np.uint8)
upper_lips = np.array([180, 68, 255], dtype=np.uint8)

# Create a binary mask of the skin pixels based on the color threshold
skin_mask = cv2.inRange(hsv, lower_color, upper_color)
mask_lips = cv2.inRange(hsv, lower_lips, upper_lips)

skin_mask_dis = cv2.bitwise_and(image, image, mask=skin_mask)
mask_lips_dis = cv2.bitwise_and(image, image, mask=mask_lips)

cv2.imshow('skin_mask_dis', skin_mask_dis)

# cv2.imshow('final_mask_skin', final_mask_skin)

# Perform morphological operations to clean up the mask
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=2)
skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

mask_lips = cv2.morphologyEx(mask_lips, cv2.MORPH_OPEN, kernel, iterations=2)
mask_lips = cv2.morphologyEx(mask_lips, cv2.MORPH_CLOSE, kernel, iterations=2)

final_mask = cv2.bitwise_not(mask_lips)
final_mask_skin = cv2.bitwise_and(skin_mask, final_mask)

# Apply the skin mask to extract the skin region
skin_with_morph = cv2.bitwise_and(image, image, mask=skin_mask)
cv2.imshow('skin_with_morph', skin_with_morph)

contours, hierarchy = cv2.findContours(final_mask_skin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the largest area
largest_contour = max(contours, key=cv2.contourArea)
other_contour = []
for contour in contours:
    if contour is not largest_contour:
        other_contour.append(contour)

frame = np.zeros_like(image)
frame2 = np.zeros_like(image)
cv2.drawContours(frame, largest_contour, -1, (255, 255, 255), 2)
cv2.drawContours(frame2, other_contour, -1, (255, 255, 255), 2)

frame_all = np.zeros_like(image)
cv2.drawContours(frame_all, contours, -1, (255, 255, 255), 2)

cv2.imshow('frame', frame)
cv2.imshow('frame2', frame2)


# Find the extreme points of the largest contour
topmost = tuple(largest_contour[largest_contour[:, :, 1].argmin()][0])
bottommost = tuple(largest_contour[largest_contour[:, :, 1].argmax()][0])

# Draw circles at the extreme points
radius = 5
color = (0, 0, 255)  # Red color (BGR format)
thickness = -1  # Fill the circle
cv2.circle(frame_all, topmost, radius, color, thickness)
cv2.circle(frame_all, bottommost, radius, color, thickness)

line_mask = np.zeros_like(image)
cv2.line(line_mask, topmost, bottommost, (255, 255, 255), 2)
cv2.line(frame_all, topmost, bottommost, (255, 255, 255), 2)
cv2.imshow('line_mask', line_mask)
cv2.imshow('frame_all', frame_all)

result = cv2.bitwise_and(line_mask, frame2)
result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
cv2.imshow('result', result)
res_contour, _ = cv2.findContours(result, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

if len(res_contour) != 0:
    # Compute the bounding rectangle for the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)
    # Draw a rectangle around the largest skin region
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('Largest Skin Region', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
