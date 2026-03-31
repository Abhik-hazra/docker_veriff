import os
import csv
import sys
from pathlib import Path

import cv2
import torch
import pandas as pd
from ultralytics import YOLO


class Tee:
    """Mirror stdout to terminal and a log file."""
    def __init__(self, filepath: str):
        log_dir = os.path.dirname(filepath)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self._terminal = sys.stdout
        self._log = open(filepath, "w", buffering=1)

    def write(self, message):
        self._terminal.write(message)
        self._log.write(message)

    def flush(self):
        self._terminal.flush()
        self._log.flush()

    def close(self):
        self._log.close()


# =========================
# Config (env overridable)
# =========================
VIDEO_FOLDER = os.getenv("VIDEO_FOLDER", "videos")
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "output/results.csv")
EVALUATION_CSV = os.getenv("EVALUATION_CSV", "output/evaluation.csv")
ANNOTATED_DIR = os.getenv("ANNOTATED_DIR", "output/annotated_frames")
LABELS_FILE = os.getenv("LABELS_FILE", "labels.txt")
LOG_FILE = os.getenv("LOG_FILE", "output/run.log")

CONFIDENCE = float(os.getenv("CONFIDENCE", "0.6"))
PERSON_CLASS_ID = int(os.getenv("PERSON_CLASS_ID", "0"))
SAMPLE_FPS = int(os.getenv("SAMPLE_FPS", "1"))
MIN_FRAMES_FOR_MULTI = int(os.getenv("MIN_FRAMES_FOR_MULTI", "2"))
SAVE_ANNOTATED = os.getenv("SAVE_ANNOTATED", "true").lower() == "true"
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", "yolov10m.pt")


def ensure_parent_dir(file_path: str):
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_model(weights: str = MODEL_WEIGHTS) -> YOLO:
    print(f"\n[INFO] Loading YOLO model weights: {weights}")
    model = YOLO(weights)

    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)
    print(f"[INFO] Running on device: {device}")
    return model


def detect_people(model: YOLO, frame) -> tuple[int, list[tuple[int, int, int, int]]]:
    results = model.predict(
        source=frame,
        classes=[PERSON_CLASS_ID],
        conf=CONFIDENCE,
        verbose=False,
    )

    boxes = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == PERSON_CLASS_ID:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                boxes.append((x1, y1, x2, y2))

    return len(boxes), boxes


def annotate_frame(frame, boxes: list, frame_idx: int, person_count: int):
    annotated = frame.copy()

    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            "Person",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    label = f"Frame {frame_idx} | Persons: {person_count}"
    cv2.putText(
        annotated,
        label,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
    )

    return annotated


def process_video(model: YOLO, video_path: str) -> dict | None:
    video_name = Path(video_path).name
    video_stem = Path(video_path).stem

    print("\n" + "=" * 55)
    print(f"[PROCESSING] {video_name}")
    print("=" * 55)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = round(total_frames / fps, 2) if fps and fps > 0 else 0

    print(f"FPS           : {fps:.2f}" if fps else "FPS           : 0.00")
    print(f"Total Frames  : {total_frames}")
    print(f"Duration      : {duration_sec} seconds")

    frame_interval = max(1, int(fps / SAMPLE_FPS)) if fps and fps > 0 else 1
    print(f"Sampling      : every {frame_interval} frame(s) ({SAMPLE_FPS} frame/sec)\n")

    if SAVE_ANNOTATED:
        video_annotate_dir = os.path.join(ANNOTATED_DIR, video_stem)
        os.makedirs(video_annotate_dir, exist_ok=True)

    max_person_count = 0
    frames_with_multiple = 0
    frames_processed = 0
    current_frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx % frame_interval == 0:
            person_count, boxes = detect_people(model, frame)
            frames_processed += 1

            max_person_count = max(max_person_count, person_count)

            if person_count >= 2:
                frames_with_multiple += 1

            if SAVE_ANNOTATED:
                annotated = annotate_frame(frame, boxes, current_frame_idx, person_count)
                save_path = os.path.join(
                    ANNOTATED_DIR,
                    video_stem,
                    f"frame_{current_frame_idx:06d}_persons_{person_count}.jpg"
                )
                cv2.imwrite(save_path, annotated)

            print(
                f"[Frame {current_frame_idx:5d}] "
                f"Sample #{frames_processed:3d} | "
                f"Persons detected: {person_count}"
            )

        current_frame_idx += 1

    cap.release()

    classification = (
        "Multiple People"
        if frames_with_multiple >= MIN_FRAMES_FOR_MULTI
        else "Single Person"
    )

    print("\n── RESULT ─────────────────────────────")
    print(f"Frames processed         : {frames_processed}")
    print(f"Max persons in one frame : {max_person_count}")
    print(f"Frames with ≥2 persons   : {frames_with_multiple}")
    print(f"Classification           : {classification}")

    return {
        "video_name": video_name,
        "duration_sec": duration_sec,
        "frames_processed": frames_processed,
        "max_person_count": max_person_count,
        "frames_with_multiple_people": frames_with_multiple,
        "classification": classification,
    }


def save_results(results: list[dict], output_path: str):
    ensure_parent_dir(output_path)

    fieldnames = [
        "video_name",
        "duration_sec",
        "frames_processed",
        "max_person_count",
        "frames_with_multiple_people",
        "classification",
    ]

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\n[SAVED] Detection results → {output_path}")


def evaluate_results(results_csv: str, labels_file: str):
    if not os.path.exists(labels_file):
        print(f"\n[WARNING] Labels file not found: {labels_file} — skipping evaluation.")
        return

    results_df = pd.read_csv(results_csv)
    results_df["predicted"] = results_df["classification"].apply(
        lambda x: 1 if x == "Multiple People" else 0
    )

    labels_df = pd.read_csv(labels_file, sep="\t", header=None)
    labels_df.columns = ["video_stem", "actual"]
    labels_df["video_stem"] = labels_df["video_stem"].astype(str).str.strip()
    labels_df["video_name"] = labels_df["video_stem"] + ".mp4"

    merged = results_df.merge(labels_df[["video_name", "actual"]], on="video_name", how="inner")
    merged["correct"] = merged["predicted"] == merged["actual"]

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(merged[["video_name", "actual", "predicted", "correct"]].to_string(index=False))

    accuracy = merged["correct"].mean() * 100 if len(merged) else 0
    total = len(merged)
    correct = int(merged["correct"].sum()) if total else 0

    print(f"\nAccuracy : {accuracy:.1f}% ({correct}/{total} videos correct)")

    wrong = merged[~merged["correct"]]
    if not wrong.empty:
        print(f"\nMisclassified ({len(wrong)} video(s)):")
        print(wrong[["video_name", "actual", "predicted"]].to_string(index=False))
    else:
        print("\nAll videos classified correctly!")

    ensure_parent_dir(EVALUATION_CSV)
    merged.to_csv(EVALUATION_CSV, index=False)
    print(f"\n[SAVED] Evaluation details → {EVALUATION_CSV}")


def main():
    ensure_parent_dir(LOG_FILE)
    tee = Tee(LOG_FILE)
    sys.stdout = tee

    try:
        if not os.path.isdir(VIDEO_FOLDER):
            print(f"[ERROR] Videos folder not found: '{VIDEO_FOLDER}'")
            return

        video_files = sorted(Path(VIDEO_FOLDER).glob("*.mp4"))
        if not video_files:
            print(f"[ERROR] No .mp4 files found in '{VIDEO_FOLDER}' folder.")
            return

        print(f"[INFO] Found {len(video_files)} video(s) to process.")
        for v in video_files:
            print(f" → {v.name}")

        model = load_model()

        all_results = []
        for video_path in video_files:
            result = process_video(model, str(video_path))
            if result:
                all_results.append(result)

        if all_results:
            save_results(all_results, OUTPUT_CSV)
            evaluate_results(OUTPUT_CSV, LABELS_FILE)

        print("\n[DONE] All videos processed. Check the output/ folder for results.")
    finally:
        tee.close()
        sys.stdout = tee._terminal


if __name__ == "__main__":
    main()
