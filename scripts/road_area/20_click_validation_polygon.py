from pathlib import Path
import argparse
import cv2
import pandas as pd

points = []

def click_event(event, x, y, flags, param):
    global points, display

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: x={x}, y={y}")

        redraw()

def redraw():
    global display, img, points

    display = img.copy()

    for i, (x, y) in enumerate(points):
        cv2.circle(display, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(
            display,
            f"P{i+1}",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            cv2.line(display, points[i], points[i+1], (0, 255, 255), 2)

    if len(points) == 4:
        cv2.line(display, points[3], points[0], (0, 255, 255), 2)

    cv2.imshow("Click 4 chalk corners: P1, P2, P3, P4. Press s to save, r reset, q quit.", display)

def main():
    global img, display, points

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    img_path = Path(args.image)
    img = cv2.imread(str(img_path))

    if img is None:
        raise SystemExit(f"Could not read image: {img_path}")

    max_w = 1400
    scale = 1.0

    if img.shape[1] > max_w:
        scale = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)

    points = []
    redraw()

    cv2.setMouseCallback(
        "Click 4 chalk corners: P1, P2, P3, P4. Press s to save, r reset, q quit.",
        click_event
    )

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            points = []
            redraw()

        elif key == ord("s"):
            if len(points) != 4:
                print("Need exactly 4 points before saving.")
                continue

            rows = []
            for i, (x, y) in enumerate(points):
                # convert back to original image coordinates if resized for display
                rows.append({
                    "point": f"P{i+1}",
                    "x_display": x,
                    "y_display": y,
                    "x_original": x / scale,
                    "y_original": y / scale,
                    "display_scale": scale,
                    "image_path": str(img_path),
                })

            out = Path(args.output_csv)
            out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"Saved: {out}")
            break

        elif key == ord("q"):
            print("Quit without saving.")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
