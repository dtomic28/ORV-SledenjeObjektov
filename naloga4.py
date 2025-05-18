import cv2 as cv
import numpy as np

# Global variables for manual ROI selection
drawing = False
ix, iy = -1, -1
rectangles = []


def draw_rectangle(event, x, y, flags, param):
    """
    Mouse callback to allow manual selection of objects in the video.
    Saves the rectangle coordinates into `rectangles`.
    """
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


def zaznaj_gibanje(cap, objekti=3):
    """
    Detects motion using background subtraction and extracts bounding boxes
    around moving objects. Waits until 3 strong detections appear consistently.
    """

    def je_nov_bbox(x, y, w, h, boxes, min_dist=30):
        """Check if a bounding box is far enough from previous ones (to avoid duplicates)."""
        cx, cy = x + w // 2, y + h // 2
        for px, py, pw, ph in boxes:
            pcx, pcy = px + pw // 2, py + ph // 2
            if np.hypot(cx - pcx, cy - pcy) < min_dist:
                return False
        return True

    def hist_strength(x, y, w, h, hsv):
        """Returns the strength of color distribution inside the box (used for ranking)."""
        roi = hsv[y : y + h, x : x + w]
        mask = cv.inRange(roi, (0, 40, 40), (180, 255, 255))
        hist = cv.calcHist([roi], [0, 1], mask, [180, 256], [0, 180, 0, 256])
        return cv.norm(hist)

    fgbg = cv.createBackgroundSubtractorMOG2(
        history=150, varThreshold=60, detectShadows=True
    )
    frame_out = None
    lokacija_ok_count = 0
    best_boxes = []

    for _ in range(150):
        ret, frame = cap.read()
        if not ret:
            break

        fgmask = fgbg.apply(frame)
        fgmask[fgmask == 127] = 0  # remove shadow mask
        fgmask = cv.GaussianBlur(fgmask, (5, 5), 0)
        _, fgmask = cv.threshold(fgmask, 200, 255, cv.THRESH_BINARY)
        fgmask = cv.dilate(fgmask, None, iterations=2)

        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            x, y, w, h = cv.boundingRect(cnt)
            if cv.contourArea(cnt) < 800 or w < 10 or h < 10:
                continue

            roi = hsv[y : y + h, x : x + w]
            mean_s = np.mean(roi[:, :, 1])
            mean_v = np.mean(roi[:, :, 2])
            if mean_s < 15 or mean_v < 15:
                continue  # skip gray/dark boxes

            if je_nov_bbox(x, y, w, h, detections):
                detections.append((x, y, w, h))

        if len(detections) >= objekti:
            detections.sort(key=lambda box: hist_strength(*box, hsv), reverse=True)
            best_boxes = detections[:objekti]
            lokacija_ok_count += 1
            if lokacija_ok_count >= 3:
                frame_out = frame
                break

    return best_boxes, frame_out


def izracunaj_znacilnice(lokacije_oken, frame):
    """
    Extracts color histograms (templates) for each bounding box.
    These are later used to track objects via histogram backprojection.
    """
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    sablone = []

    for x, y, w, h in lokacije_oken:
        pad_x = int(w * 0.16)
        pad_y = int(h * 0.16)
        roi = hsv[y + pad_y : y + h - pad_y, x + pad_x : x + w - pad_x]

        # Use a 1D histogram on hue for general robustness
        mask = cv.inRange(roi, (0, 10, 10), (180, 255, 255))
        hist = cv.calcHist([roi], [0], mask, [180], [0, 180])
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
        sablone.append(hist)

    return sablone


def camshift(slika, sablona, lokacija_okna, iteracije=10, napaka=1):
    """
    A simplified CamShift implementation using histogram backprojection.
    It shifts the search window to the centroid of the object based on histogram response.
    """
    hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)
    x, y, w, h = lokacija_okna

    for _ in range(iteracije):
        if len(sablona.shape) == 2:
            backproj = cv.calcBackProject([hsv], [0, 1], sablona, [0, 180, 0, 256], 1)
        else:
            backproj = cv.calcBackProject([hsv], [0], sablona, [0, 180], 1)

        roi = backproj[y : y + h, x : x + w]
        m = cv.moments(roi)

        if m["m00"] < 1:
            return lokacija_okna, False  # Object likely lost

        cx = int(m["m10"] / m["m00"]) + x
        cy = int(m["m01"] / m["m00"]) + y
        new_x = max(0, min(slika.shape[1] - w, cx - w // 2))
        new_y = max(0, min(slika.shape[0] - h, cy - h // 2))

        if abs(new_x - x) < napaka and abs(new_y - y) < napaka:
            break

        x, y = new_x, new_y

    return (x, y, w, h), True


# ========================= MAIN =========================

if __name__ == "__main__":
    # cap = cv.VideoCapture(".videos/zaznaj_gibanje_nato_sledi.mp4")
    cap = cv.VideoCapture(".videos/sledi_gibanju.mp4")
    if not cap.isOpened():
        exit("Napaka: video ni naložen.")

    ret, first_frame = cap.read()
    if not ret:
        exit("Napaka: ne morem prebrati prve slike.")

    # Prepare video output
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter(
        "tracking_output.avi",
        fourcc,
        cap.get(cv.CAP_PROP_FPS),
        (int(cap.get(cv.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))),
    )

    zaznaj_mode = "auto"  # can also be "manual"

    if zaznaj_mode == "manual":
        cv.imshow("Select ROI", first_frame)
        cv.setMouseCallback("Select ROI", draw_rectangle, first_frame)
        print("Izberi ROIs z miško. Pritisni 'q', ko končaš.")
        while True:
            if cv.waitKey(1) & 0xFF == ord("q"):
                break
        sablone = izracunaj_znacilnice(rectangles, first_frame)
    else:
        rectangles, snapshot = zaznaj_gibanje(cap, objekti=3)
        if not rectangles:
            exit("Napaka: objekti niso zaznani.")
        sablone = izracunaj_znacilnice(rectangles, snapshot)

    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for i in range(len(rectangles)):
            x, y, w, h = rectangles[i]
            new_box, ok = camshift(frame, sablone[i], (x, y, w, h))
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
