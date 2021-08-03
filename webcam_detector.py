import yolov5
import cv2

"""
A simple webcam object detector using yolov5.
"""

def detect_objects():
    model = yolov5.load('yolov5s.pt')

    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, frame = cap.read()
        # frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

        detected = model(frame)
        detected.save("results")

        cv2.imshow('input', detected.imgs[0])

        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_objects()
