import cv2 as cv
import numpy as np

drawing = False
ix, iy = -1, -1
rectangles = []


def draw_rectangle(event, x, y, flags, param):
    global drawing, ix, iy, rectangles
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        temp = param.copy()
        cv.rectangle(temp, (ix, iy), (x, y), (0, 255, 0), 2)
        cv.imshow("Select ROI", temp)
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        x0, y0 = min(ix, x), min(iy, y)
        x1, y1 = max(ix, x), max(iy, y)
        rectangles.append((x0, y0, x1 - x0, y1 - y0))
        cv.rectangle(param, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.imshow("Select ROI", param)


def detect_motion(cap, count=3):
    fgbg = cv.createBackgroundSubtractorMOG2(
        history=150, varThreshold=60, detectShadows=True
    )
    detections = []
    frame_out = None

    for _ in range(150):
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = fgbg.apply(frame)
        fgmask[fgmask == 127] = 0  # remove shadow mask
        fgmask = cv.GaussianBlur(fgmask, (5, 5), 0)
        _, fgmask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
        fgmask = cv.dilate(fgmask, None, iterations=2)

        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if cv.contourArea(cnt) < 800 or w < 10 or h < 10:
                continue

            roi = hsv[y : y + h, x : x + w]
            mean_s = np.mean(roi[:, :, 1])
            mean_v = np.mean(roi[:, :, 2])
            if mean_s < 15 or mean_v < 15:
                continue  # too gray or too dark

            detections.append((x, y, w, h))

        if len(detections) >= count:
            frame_out = frame
            break

    return detections[:count], frame_out


def calculate_histograms(rects, frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    templates = []

    for x, y, w, h in rects:
        pad_x = int(w * 0.2)
        pad_y = int(h * 0.2)
        roi = hsv[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

        mean_s = np.mean(roi[:, :, 1])

        if mean_s < 40:
            mask = cv.inRange(roi, (0, 10, 10), (180, 255, 255))
            hist = cv.calcHist([roi], [0], mask, [180], [0, 180])
        else:
            mask = cv.inRange(roi, (0, 30, 30), (180, 255, 255))
            hist = cv.calcHist([roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])

        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        templates.append(hist)

    return templates


def camshift(hsv, hist, window):
    x, y, w, h = window
    for _ in range(10):
        if len(hist.shape) == 2:  # 2D histogram
            backproj = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
        else:  # 1D hue only
            backproj = cv.calcBackProject([hsv], [0], hist, [0, 180], 1)

        roi = backproj[y : y + h, x : x + w]
        m = cv.moments(roi)
        if m["m00"] < 1:
            return window, False

        cx = int(m["m10"] / m["m00"]) + x
        cy = int(m["m01"] / m["m00"]) + y
        new_x = max(0, min(hsv.shape[1] - w, cx - w // 2))
        new_y = max(0, min(hsv.shape[0] - h, cy - h // 2))
        if abs(new_x - x) < 2 and abs(new_y - y) < 2:
            break
        x, y = new_x, new_y

    return (x, y, w, h), True


if __name__ == "__main__":
    cap = cv.VideoCapture(".videos/zaznaj_gibanje_nato_sledi.mp4")
    if not cap.isOpened():
        exit("Napaka: video ni naložen.")

    ret, first = cap.read()
    if not ret:
        exit("Napaka: ne morem prebrati prve slike.")

    mode = "auto"
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(
        "tracking_output.avi",
        fourcc,
        cap.get(cv.CAP_PROP_FPS),
        (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))),
    )

    if mode == "manual":
        cv.imshow("Select ROI", first)
        cv.setMouseCallback("Select ROI", draw_rectangle, first)
        print("Izberi ROIs, pritisni 'q' ko končaš.")
        while True:
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        templates = calculate_histograms(rectangles, first)
    else:
        rectangles, snapshot = detect_motion(cap, count=3)
        if not rectangles:
            exit("Napaka: objekti niso zaznani.")
        templates = calculate_histograms(rectangles, snapshot)

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        for i in range(len(rectangles)):
            x, y, w, h = rectangles[i]
            new_box, ok = camshift(hsv, templates[i], (x, y, w, h))
            rectangles[i] = new_box
            x, y, w, h = new_box

            if ok:
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv.putText(
                    frame,
                    f"Obj {i}",
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
            else:
                cv.putText(
                    frame,
                    f"Obj {i} LOST",
                    (x, y - 5),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                )

        cv.imshow("Tracking", frame)
        out.write(frame)
        if cv.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
