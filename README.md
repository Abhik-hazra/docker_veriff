# Veriff — Person Detection Challenge

Classifies each video as `Single Person` or `Multiple People` using YOLO.

## Project Structure

```text
veriff-submission/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── Dockerfile.cloudrun
├── README.md
├── main.py
├── requirements.txt
├── veriff_submission.py
├── labels.txt                # optional
├── videos/
│   └── .gitkeep
└── output/
    └── .gitkeep
```

## Local Python Run

1. Create and activate a virtual environment
2. Install dependencies
3. Put `.mp4` files into `videos/`
4. Run the script

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python veriff_submission.py
```

## Local Docker Run

Build the image:

```bash
docker build -t veriff-local .
```

Run the container:

```bash
docker run --rm \
  -v "$(pwd)/videos:/app/videos" \
  -v "$(pwd)/output:/app/output" \
  -v "$(pwd)/labels.txt:/app/labels.txt" \
  veriff-local
```

If you do not have a `labels.txt` file, remove that third `-v` mount.

## Output

- `output/results.csv`
- `output/evaluation.csv` (if labels exist)
- `output/run.log`
- `output/annotated_frames/` (if annotation saving is enabled)
