import matplotlib.pylab as plt
import sys
import cv2
import numpy as np

offset_width = 100

def region(image):
    """This function isolates certain the region that contains the lane lines
    it takes the image we want (We will pass the canny results) to isolate and
    produces the masked image based on our defined region of interest"""

    height, width = image.shape
    region_of_interest = np.array([[(offset_width, height), (550, 350), (1200, height)]])
    mask = np.zeros_like(image)  # Mask now is black image
    mask = cv2.fillPoly(mask, region_of_interest, 255)  # Fill region with white
    mask = cv2.bitwise_and(image, mask)  # Isolate edges that correspond with lane lines

    return mask


def average(image, lines):
    """This function averages hough lines result. it finds m and c of line
    and outputs 2 solid lines instead (one on each side)"""

    left = []
    right = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        m = parameters[0]
        c = parameters[1]
        if m < 0:
            left.append((m, c))
        else:
            right.append((m, c))

    right_avg = np.average(right, axis=0)
    left_avg = np.average(left, axis=0)
    left_line = make_points(image, left_avg)
    right_line = make_points(image, right_avg)
    return np.array([left_line, right_line])


def edge_detection(img, gray_img, blurred_img):
    """
    # Canny Edge Detection
    Note: Regular canny detection will not work if the lanes has yellow color.
    You can use the following canny() function to reduce overhead **IF** the lanes are only white.
    However, the yellow detection code also detects white lanes anyways.

    edges = cv2.Canny(blurred_img, 50, 150)
    """

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_yellow = np.array([20, 100, 100], dtype="uint8")  # Yellow Ranges
    upper_yellow = np.array([100, 255, 255], dtype="uint8")  # Yellow Ranges
    yellow_mask = cv2.inRange(hsv_img, lower_yellow, upper_yellow)
    white_mask = cv2.inRange(gray_img, 200, 255)
    yw_mask = cv2.bitwise_or(white_mask, yellow_mask)
    edges = cv2.bitwise_and(gray_img, yw_mask)
    return edges


def make_points(image, average):
    try:
        m, c = average
    except TypeError:
        m, c = 0.001, 0
    y1 = int(image.shape[0])
    y2 = int(y1 * (3.3 / 5))
    x1 = int((y1 - c) // m)
    x2 = int((y2 - c) // m)
    return np.array([x1, y1, x2, y2])


def display_lines(image, lines):
    lines_image = np.zeros_like(image, 'uint8')
    poly_image = np.zeros_like(image, 'uint8')
    points = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line
            cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)

            # Fill space between lines
            if x1 < x2:
                points.append([x1, y1])
                points.append([x2, y2])
            else:
                points.append([x2, y2])
                points.append([x1, y1])

    # Measure distance from center
    actual_center_x_axis = image.shape[1] / 2
    car_center_x_axis = points[0][0] + (points[-1][0] - points[0][0]) / 2
    center_diff = actual_center_x_axis - car_center_x_axis  # Scale is: 10 px = 1 cm

    if center_diff > 0:
        text = f"Vehicle is {abs(round(float(center_diff / 1000), 2))}m left of the center of the lane"
    elif center_diff < 0:
        text = f"Vehicle is {abs(round(float(center_diff / 1000), 2))}m right of the center of the lane"
    else:
        text = f"Vehicle is at the center of the lane"

    # Highlight the lanes
    points = np.array(points)
    cv2.fillPoly(poly_image, [points], (0, 255, 0))
    cv2.addWeighted(poly_image, 0.3, lines_image, 1, 0, lines_image)
    lines_image = cv2.putText(lines_image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    #  Crop car engine hood
    lines_image = cv2.rectangle(lines_image, (0, image.shape[0] - 75), (image.shape[1], image.shape[0]), (0, 0, 0), -1)

    return lines_image


def frame_process(img):
    # Read Image
    copy = np.copy(img)

    # Convert to Grayscale
    gray_img = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)

    # Add Gaussian blur to improve canny results
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)

    # Edge Detection with support of yellow lanes
    edges = edge_detection(img, gray_img, blurred_img)

    # Isolated lane lines edges
    isolated_lanes = region(edges)

    # Hough line transform
    lines = cv2.HoughLinesP(isolated_lanes, 1, np.pi / 180, 35, np.array([]), minLineLength=25, maxLineGap=2)
    averaged_lines = average(copy, lines)
    black_lines = display_lines(copy, averaged_lines)
    lanes = cv2.addWeighted(copy, 1, black_lines, 1, 1)
    return lanes

# python3 -mode path

path = sys.argv[2]

#  user_input = input("Type 'i' for Image mode, 'v' for Video mode:\n> ").lower()
if sys.argv[1][-1] == 'i':
    #  path = input("Enter the path for the image to process (example: test_images/test6.jpg):\n> ")
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imshow('Result', img)
    plt.imshow(frame_process(img))
    plt.show()

elif sys.argv[1][-1] == "v":
    video = cv2.VideoCapture(path)
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    save_name = f"Result_{path}.mp4"
    result = cv2.VideoWriter(save_name,cv2.VideoWriter_fourcc(*'MP4V'), 25, (frame_width, frame_height))

    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = frame_process(frame)
            result.write(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    video.release()
    cv2.destroyAllWindows()
else:
    print("Incorrect mode entered. Please try again.")
