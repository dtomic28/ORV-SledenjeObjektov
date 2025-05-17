import cv2 as cv
import numpy as np

# Global variables for the rectangle drawing
drawing = False
ix, iy = -1, -1
rectangles = []


def draw_rectangle(event, x, y, flags, param):
    """Callback function to draw a rectangle on the image"""
    global ix, iy, drawing, rectangles

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            img_copy = param.copy()  # Copy the image to avoid drawing on the original
            cv.rectangle(img_copy, (ix, iy), (x, y), (0, 255, 0), 2)
            cv.imshow("Select ROI", img_copy)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # Ensure correct coordinates after mouse release (handle cases when coordinates are inverted)
        x0, y0 = ix, iy
        x1, y1 = x, y
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        rectangles.append((x0, y0, x1 - x0, y1 - y0))  # Store the rectangle coordinates
        cv.rectangle(param, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv.imshow("Select ROI", param)
        print(f"Selected ROI: {(x0, y0, x1 - x0, y1 - y0)}")


def zaznaj_gibanje(cap, st_objektov=1):
    """Funkcija za detekcijo gibanja, ki zazna premikajoče objekte"""
    # Ustvarimo MOG2 objekt za odštevanje ozadja
    fgbg = cv.createBackgroundSubtractorMOG2()

    detected_objects = 0  # Število zaznanih objektov
    lokacije_oken = []  # Seznam vseh pravokotnikov (lokacij objektov)

    while True:
        # Preberi naslednji okvir
        ret, frame = cap.read()
        if not ret:
            print("No more frames to read.")
            break  # Ko ni več okvirjev

        # Uporabi MOG2 za izračun gibanja
        fgmask = fgbg.apply(frame)

        # Poišči konture v maski
        contours, _ = cv.findContours(fgmask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Za vsako konturo, ki presega določen prag površine, narišemo pravokotnik
        for cnt in contours:
            if cv.contourArea(cnt) > 500:  # Minimalna površina za zaznavanje objekta
                x, y, w, h = cv.boundingRect(cnt)
                lokacije_oken.append((x, y, w, h))
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                detected_objects += 1

        # Prekini, ko dosežemo zahtevano število objektov
        if detected_objects >= st_objektov:
            print(f"Detected {detected_objects} objects.")
            break  # Zaznali smo dovolj objektov, prekinemo iskanje

        # Prikaz slike z detekcijami gibanja
        cv.imshow("Motion Detected", frame)

        # Pritisnite 'q' za prekinitev
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    return lokacije_oken


def izracunaj_znacilnice(lokacije_oken, frame):
    """Calculate object templates (histograms) for tracking."""
    sablone = []
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Debug: Show the first frame with rectangles for ROI selection
    print("Calculating histograms for detected objects...")

    for x, y, w, h in lokacije_oken:
        roi = hsv[y : y + h, x : x + w]

        # Optional: Visualize the selected ROI
        cv.rectangle(
            frame, (x, y), (x + w, y + h), (255, 0, 0), 2
        )  # Draw rectangle on the original image
        cv.imwrite("Object.png", frame)
        cv.waitKey(1)  # Display the image with the ROI for debugging purposes

        # Calculate the histogram for the ROI
        hist = cv.calcHist([roi], [0, 1], None, [256, 256], [0, 180, 0, 256])

        # Normalize the histogram to ensure values are between 0 and 255
        cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)

        # Append the histogram to the list of templates
        sablone.append(hist)

    return sablone


if __name__ == "__main__":
    # Load video
    cap = cv.VideoCapture(".videos/zaznaj_gibanje_nato_sledi.mp4")

    if not cap.isOpened():
        print("Error: Video not found or couldn't be loaded.")
        exit(1)

    # Set parameters for Camshift
    iteracije = 10
    napaka = 1

    # Initialize rectangles (empty)
    rectangles = []

    # Read the first frame
    ret, prva_slika = cap.read()

    # Prompt to select motion detection or manual selection
    zaznaj_gibanje_mode = "avtomatsko"  # Choose "rocno" for manual or "avtomatsko" for automatic motion detection

    if zaznaj_gibanje_mode == "rocno":
        # Display the frame and allow the user to select ROI manually
        cv.imshow("Select ROI", prva_slika)
        cv.setMouseCallback("Select ROI", draw_rectangle, param=prva_slika)

        # Wait for the user to draw a rectangle and click on the window
        print("Please select a region to track by dragging the mouse.")
        while True:
            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):  # Press 'q' to finish selection
                break

        # If no rectangle was drawn, exit
        if not rectangles:
            print("No object selected for tracking.")
            exit(1)

    elif zaznaj_gibanje_mode == "avtomatsko":
        # Automatic motion detection
        rectangles = zaznaj_gibanje(cap, st_objektov=3)

    ret, detected_frame = cap.read()
    if not ret:
        print("Error: No more frames to read.")
        exit(1)

    # Calculate object templates for tracking using the frame where objects were detected
    sablone = izracunaj_znacilnice(rectangles, detected_frame)

    # Start tracking and showing the video
    while True:
        ret, slika = cap.read()
        if not ret:
            print("crkne")
            break  # Break the loop if no more frames are available

        # Convert frame to HSV
        hsv = cv.cvtColor(slika, cv.COLOR_BGR2HSV)

        # Apply Camshift tracking for each selected object (or detected)
        for rect in rectangles:
            x, y, w, h = rect
            roi = hsv[y : y + h, x : x + w]
            hist = cv.calcHist([roi], [0, 1], None, [256, 256], [0, 180, 0, 256])
            cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            backproj = cv.calcBackProject([hsv], [0, 1], hist, [0, 180, 0, 256], 1)

            # Apply Camshift to find the new location of the object
            ret, track_window = cv.CamShift(
                backproj,
                (x, y, w, h),
                (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, iteracije, napaka),
            )

            # Draw the tracked object
            pts = cv.boxPoints(ret)
            pts = np.int32(pts)  # Convert to int32
            cv.polylines(slika, [pts], True, (0, 255, 0), 2)

        # Display the result in the window
        cv.imshow("Tracking Result", slika)

        # Check for keypress (wait indefinitely for keypress)
        key = cv.waitKey(1)  # Wait for 1 millisecond and capture key press
        if key == ord("q"):  # If 'q' is pressed, break the loop
            break

    # Release the capture and close the window when done
    cap.release()
    cv.destroyAllWindows()
