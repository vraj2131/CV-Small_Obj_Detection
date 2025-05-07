import glob, os, shutil

# set these to match your setup
IMG_W, IMG_H = 512, 512    # VEDAI tile size
MODALITIES = ("co","ir")

for modality in MODALITIES:
    img_dir = f"data/vedai/images/{modality}"
    lbl_dir = f"data/vedai/labels/{modality}"
    print(f"Processing {lbl_dir} …")

    for img_path in glob.glob(f"{img_dir}/*.png"):
        base = os.path.splitext(os.path.basename(img_path))[0]    # e.g. "00001270_co"
        lbl_path = os.path.join(lbl_dir, base + ".txt")
        if not os.path.exists(lbl_path):
            # if no label exists, create an empty one
            open(lbl_path, "w").close()
            continue

        yolo_lines = []
        for line in open(lbl_path):
            parts = line.strip().split()
            # detect polygon‐style (≥8 coords) or bad lines
            if len(parts) < 8:
                continue
            # last 8 numbers are polygon xy pairs
            coords = list(map(float, parts[-8:]))
            xs = coords[0::2]
            ys = coords[1::2]
            xmin, xmax = min(xs), max(xs)
            ymin, ymax = min(ys), max(ys)
            x_c = (xmin + xmax) / 2.0 / IMG_W
            y_c = (ymin + ymax) / 2.0 / IMG_H
            w   = (xmax - xmin) / IMG_W
            h   = (ymax - ymin) / IMG_H

            # force class_id = 0 for "vehicle"
            yolo_lines.append(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}")

        # overwrite the label file
        with open(lbl_path, "w") as f:
            f.write("\n".join(yolo_lines))

    # remove old cache so YOLOv5 will rescan
    cache_file = os.path.join(lbl_dir, f"{modality}.cache")
    if os.path.exists(cache_file):
        os.remove(cache_file)
        print(f"  • removed cache {cache_file}")

print("Done converting all labels to single-class YOLO format.")
