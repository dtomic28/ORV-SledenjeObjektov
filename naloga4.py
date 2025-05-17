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
    bg_sub = cv.createBackgroundSubtractorMOG2(
        history=200, varThreshold=50, detectShadows=True
    )
    valid_detections = []
    frame_out = None

    for _ in range(150):
        ret, frame = cap.read()
        if not ret:
            break
        fgmask = bg_sub.apply(frame)
        fgmask[fgmask == 127] = 0
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
            mask = cv.inRange(roi, (0, 40, 40), (180, 255, 255))
            nonzero = cv.countNonZero(mask)
            if nonzero < (w * h * 0.2):
                continue

            valid_detections.append((x, y, w, h))
        if len(valid_detections) >= count:
            frame_out = frame
            break

    return valid_detections[:count], frame_out


def calculate_histograms(rects, frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    templates = []
    for i, (x, y, w, h) in enumerate(rects):
        pad_x = int(w * 0.3)
        pad_y = int(h * 0.3)
        roi = hsv[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

        # Check average saturation to choose histogram type
        avg_saturation = np.mean(roi[:, :, 1])
        if avg_saturation < 50:
            use_1d = True
        else:
            use_1d = False

        if use_1d:
            mask = cv.inRange(roi, (0, 10, 10), (180, 255, 255))
            hist = cv.calcHist([roi], [0], mask, [180], [0, 180])
        else:
            mask = cv.inRange(roi, (0, 20, 20), (180, 255, 255))
            hist = cv.calcHist([roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])

        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        templates.append(hist)

    return templates


def camshift(hsv, hist, window):
    x, y, w, h = window
    for _ in range(10):
        backproj = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)
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

    # refine size
    roi = backproj[y : y + h, x : x + w]
    proj_x = np.sum(roi, axis=0)
    proj_y = np.sum(roi, axis=1)
    w_new = np.count_nonzero(proj_x)
    h_new = np.count_nonzero(proj_y)
    if 10 < w_new < hsv.shape[1] and 10 < h_new < hsv.shape[0]:
        w = int(0.8 * w + 0.2 * w_new)
        h = int(0.8 * h + 0.2 * h_new)

    return (x, y, w, h), True


if __name__ == "__main__":
    cap = cv.VideoCapture(".videos/sledi_gibanju.mp4")
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    out = cv.VideoWriter("tracking_output.avi", fourcc, fps, (width, height))

    ret, first_frame = cap.read()
    if not ret:
        exit("Napaka: ne morem prebrati prve slike.")

    mode = "manual"  # "manual"

    if mode == "manual":
        cv.imshow("Select ROI", first_frame)
        cv.setMouseCallback("Select ROI", draw_rectangle, first_frame)
        print("Draw ROIs, press 'q' when done.")
        while True:
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        templates = calculate_histograms(rectangles, first_frame)
    else:
        rectangles, detection_frame = detect_motion(cap, 3)
        if not rectangles:
            exit("Napaka: ne najdem dovolj objektov.")
        templates = calculate_histograms(rectangles, detection_frame)
        for i, hist in enumerate(templates):
            hist_img = cv.normalize(hist, None, 0, 255, cv.NORM_MINMAX)
            hist_img = cv.resize(hist_img, (256, 180))  # resize for view
            cv.imshow(f"Hist {i}", hist_img.astype(np.uint8))

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

            # Debug backproj view
            backproj = cv.calcBackProject(
                [hsv], [0, 1], templates[i], [0, 180, 0, 256], 1
            )
            debug_roi = backproj[y : y + h, x : x + w]
            color_map = cv.applyColorMap(debug_roi, cv.COLORMAP_JET)
            if color_map.shape[0] > 0 and color_map.shape[1] > 0:
                resized = cv.resize(color_map, (60, 60))
                frame[10 + i * 65 : 70 + i * 65, 10:70] = resized

        cv.imshow("Tracking", frame)
        out.write(frame)
        if cv.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv.destroyAllWindows()
