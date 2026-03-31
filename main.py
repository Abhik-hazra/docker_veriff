import os
import subprocess
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.get("/health")
def health():
    return jsonify({"status": "healthy"}), 200


@app.post("/process-video")
def process_video():
    try:
        data = request.get_json(silent=True) or {}

        video_folder = data.get("video_folder", os.getenv("VIDEO_FOLDER", "videos"))
        output_csv = data.get("output_csv", os.getenv("OUTPUT_CSV", "output/results.csv"))
        evaluation_csv = data.get("evaluation_csv", os.getenv("EVALUATION_CSV", "output/evaluation.csv"))
        annotated_dir = data.get("annotated_dir", os.getenv("ANNOTATED_DIR", "output/annotated_frames"))
        labels_file = data.get("labels_file", os.getenv("LABELS_FILE", "labels.txt"))
        log_file = data.get("log_file", os.getenv("LOG_FILE", "output/run.log"))

        env = os.environ.copy()
        env["VIDEO_FOLDER"] = video_folder
        env["OUTPUT_CSV"] = output_csv
        env["EVALUATION_CSV"] = evaluation_csv
        env["ANNOTATED_DIR"] = annotated_dir
        env["LABELS_FILE"] = labels_file
        env["LOG_FILE"] = log_file

        result = subprocess.run(
            ["python", "veriff_submission.py"],
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "stdout": result.stdout,
                "stderr": result.stderr
            }), 500

        return jsonify({
            "status": "success",
            "stdout": result.stdout,
            "stderr": result.stderr
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
