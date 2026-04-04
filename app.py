"""
MicroAudio — MVP Localhost
Upload MP4 + Phase Inversion (right channel) + Download
Story 1.1
"""

import os
import uuid
import shutil
import subprocess
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    jsonify,
    after_this_request,
)

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
PROCESSED_DIR = BASE_DIR / "processed"
ALLOWED_EXTENSIONS = {".mp4"}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    """Return True if ffmpeg is reachable on PATH."""
    return shutil.which("ffmpeg") is not None


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _process_video(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Run phase-inversion via FFmpeg. Returns (success, message)."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-af", "pan=stereo|c0=c0|c1=-1*c1",
        "-c:v", "copy",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min max
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            # Provide a human-friendly excerpt
            lines = [l for l in stderr.splitlines() if l.strip()]
            tail = "\n".join(lines[-5:]) if lines else "Unknown error"
            return False, f"FFmpeg failed (code {result.returncode}):\n{tail}"
        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, "Processing timed out after 10 minutes."
    except Exception as exc:
        return False, f"Unexpected error: {exc}"


def _cleanup(*paths: Path) -> None:
    for p in paths:
        try:
            p.unlink(missing_ok=True)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    ffmpeg_ok = _ffmpeg_available()
    return render_template("index.html", ffmpeg_ok=ffmpeg_ok)


@app.route("/upload", methods=["POST"])
def upload():
    # --- preflight: FFmpeg ---
    if not _ffmpeg_available():
        return jsonify({"error": "FFmpeg nao encontrado no sistema. Instale o FFmpeg e adicione ao PATH."}), 500

    # --- file validation ---
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "Nome de arquivo vazio."}), 400

    if not _is_allowed(file.filename):
        return jsonify({"error": "Formato invalido. Apenas arquivos MP4 sao aceitos."}), 400

    # --- save upload ---
    job_id = uuid.uuid4().hex
    original_ext = Path(file.filename).suffix
    input_path = UPLOAD_DIR / f"{job_id}{original_ext}"
    output_path = PROCESSED_DIR / f"{job_id}_processed{original_ext}"

    try:
        file.save(str(input_path))
    except Exception as exc:
        return jsonify({"error": f"Falha ao salvar arquivo: {exc}"}), 500

    # --- process ---
    success, message = _process_video(input_path, output_path)

    if not success:
        _cleanup(input_path, output_path)
        return jsonify({"error": message}), 422

    # Cleanup input immediately; output cleaned after download
    _cleanup(input_path)

    return jsonify({
        "success": True,
        "job_id": job_id,
        "filename": f"microaudio_{Path(file.filename).stem}{original_ext}",
    })


@app.route("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    # Sanitize job_id to prevent path traversal
    safe_id = "".join(c for c in job_id if c.isalnum())
    output_path = PROCESSED_DIR / f"{safe_id}_processed.mp4"

    if not output_path.exists():
        return jsonify({"error": "Arquivo nao encontrado. Pode ter expirado."}), 404

    @after_this_request
    def cleanup(response):
        _cleanup(output_path)
        return response

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=filename,
        mimetype="video/mp4",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  MicroAudio — MVP")
    print(f"  FFmpeg: {'OK' if _ffmpeg_available() else 'NAO ENCONTRADO'}")
    print(f"  Porta: {port}")
    print("=" * 50)
    app.run(host="0.0.0.0", port=port, debug=False)
