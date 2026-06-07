from pathlib import Path
import argparse
import json

import cv2
import numpy as np


clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append([x, y])
        print(f"Point {len(clicked_points)}: ({x}, {y})")


def draw_points(image, points):
    out = image.copy()

    labels = [
        "1 near-left",
        "2 near-right",
        "3 far-right",
        "4 far-left",
    ]

    for i, (x, y) in enumerate(points):
        cv2.circle(out, (int(x), int(y)), 6, (0, 0, 255), -1)
        cv2.putText(
            out,
            labels[i] if i < len(labels) else str(i + 1),
            (int(x) + 8, int(y) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    if len(points) >= 2:
        for i in range(len(points) - 1):
            p1 = tuple(map(int, points[i]))
            p2 = tuple(map(int, points[i + 1]))
            cv2.line(out, p1, p2, (0, 255, 255), 2)

    if len(points) == 4:
        cv2.line(out, tuple(map(int, points[3])), tuple(map(int, points[0])), (0, 255, 255), 2)

    return out


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True)
    parser.add_argument("--lens-id", type=int, required=True)
    parser.add_argument("--real-width-m", type=float, required=True)
    parser.add_argument("--real-length-m", type=float, required=True)
    parser.add_argument("--resolution-m-per-px", type=float, default=0.05)
    parser.add_argument("--output-json", required=True)

    args = parser.parse_args()

    image_path = Path(args.image)
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    print("\nClick 4 road points in this order:")
    print("1. near-left")
    print("2. near-right")
    print("3. far-right")
    print("4. far-left")
    print("\nImportant:")
    print("- All 4 points must lie on the same flat road plane.")
    print("- The clicked quadrilateral should correspond to the real-width and real-length you provide.")
    print("- Press 'r' to reset points.")
    print("- Press 'q' after selecting 4 points.")

    window_name = "Click road calibration points"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    global clicked_points

    while True:
        display = draw_points(image, clicked_points)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        if key == ord("r"):
            clicked_points = []
            print("Reset points.")

        elif key == ord("q"):
            if len(clicked_points) != 4:
                print("Need exactly 4 points before quitting.")
                continue
            break

    cv2.destroyAllWindows()

    src_pts = np.array(clicked_points, dtype=np.float32)

    output_width_px = int(round(args.real_width_m / args.resolution_m_per_px))
    output_height_px = int(round(args.real_length_m / args.resolution_m_per_px))

    # Destination bird's-eye rectangle:
    # near-left  -> bottom-left
    # near-right -> bottom-right
    # far-right  -> top-right
    # far-left   -> top-left
    dst_pts = np.array(
        [
            [0, output_height_px - 1],
            [output_width_px - 1, output_height_px - 1],
            [output_width_px - 1, 0],
            [0, 0],
        ],
        dtype=np.float32,
    )

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)

    calibration = {
        "lens_id": args.lens_id,
        "image_path": str(image_path),
        "image_points_order": [
            "near_left",
            "near_right",
            "far_right",
            "far_left",
        ],
        "image_points_xy": clicked_points,
        "real_width_m": args.real_width_m,
        "real_length_m": args.real_length_m,
        "resolution_m_per_px": args.resolution_m_per_px,
        "output_width_px": output_width_px,
        "output_height_px": output_height_px,
        "homography_image_to_bev": H.tolist(),
        "area_per_pixel_m2": args.resolution_m_per_px ** 2,
    }

    with open(output_json, "w") as f:
        json.dump(calibration, f, indent=2)

    print("\nSaved calibration:", output_json)
    print("Output BEV size:", output_width_px, "x", output_height_px)
    print("Area per BEV pixel:", args.resolution_m_per_px ** 2, "m²")


if __name__ == "__main__":
    main()