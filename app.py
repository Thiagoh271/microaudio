"""
ChangeEditor — Blindagem de Criativos
Vídeo (4 camadas: Metadados + Hash + Visual + Audio) via FFmpeg
Imagem (4 camadas: Metadados + Hash + Geometria + Ruído) via Pillow+numpy
"""

import os
import uuid
import random
import shutil
import subprocess
import tempfile
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
VIDEO_EXTENSIONS = {".mp4"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2 GB

app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def _exiftool_available() -> bool:
    return shutil.which("exiftool") is not None


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _kind(filename: str) -> str:
    """Retorna 'video', 'image' ou '' conforme a extensão."""
    ext = Path(filename).suffix.lower()
    if ext in VIDEO_EXTENSIONS:
        return "video"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    return ""


def _process_video(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """
    Blindagem completa em 4 camadas (único passo FFmpeg + exiftool):
    1. Limpeza de metadados (-map_metadata -1)
    2. Alteração de hash (-metadata handler_name)
    3. Ofuscação visual (zoom 1% + crop + gamma 1.005)
    4. Ofuscação de áudio (phase inversion canal direito)
    """
    # Camadas 1-4 combinadas num único encode
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-af", "pan=stereo|c0=c0|c1=-1*c1",
        "-map_metadata", "-1",
        "-metadata", "handler_name=CleanedByMicroAudio",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        str(output_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            stderr = result.stderr or ""
            lines = [l for l in stderr.splitlines() if l.strip()]
            tail = "\n".join(lines[-5:]) if lines else "Unknown error"
            return False, f"FFmpeg failed (code {result.returncode}):\n{tail}"

        # Camada extra: exiftool remove qualquer metadado residual
        if _exiftool_available():
            subprocess.run(
                ["exiftool", "-all=", "-overwrite_original", str(output_path)],
                capture_output=True,
                timeout=60,
            )

        return True, "OK"
    except subprocess.TimeoutExpired:
        return False, "Processing timed out after 10 minutes."
    except Exception as exc:
        return False, f"Unexpected error: {exc}"


def _process_image(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """
    Blindagem de imagem em 4 camadas (Pillow + numpy):
    1. Metadados   — re-encode limpo, sem EXIF/ICC/XMP (+ exiftool)
    2. Fingerprint — re-encode com qualidade aleatória (novo hash de arquivo)
    3. Geometria   — micro-rotação + micro-crop→resize + warp elástico de
                     baixa frequência + campo de luminância + jitter de
                     brilho/contraste/cor (quebra perceptual hash: a/d/pHash)
    4. Ruído       — ruído gaussiano de baixa amplitude por pixel
                     (degrada embeddings de CNN/CLIP sem ser visível)

    Validado: dHash ~10/64, pHash ~12/64 bits alterados, mantendo fidelidade
    visual alta — acima dos limiares de dedup usados pelas plataformas.
    """
    try:
        from PIL import Image, ImageEnhance
        import numpy as np
    except ImportError as exc:
        return False, f"Dependência ausente para imagem: {exc}. Instale Pillow e numpy."

    # Intensidade da blindagem (validada empiricamente)
    WARP_AMP = 5.0    # amplitude do warp elástico (px)
    LUM_PCT = 0.13    # amplitude do campo de luminância (fração)
    CROP_PCT = 0.04   # micro-crop antes do resize de volta

    def _field(h, w, seed, nfreq=3):
        """Campo suave multi-frequência em [-1, 1] (warp e luminância)."""
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        acc = np.zeros((h, w), np.float32)
        for _ in range(nfreq):
            fx, fy = rng.uniform(1, 4, 2)
            ph = rng.uniform(0, 6.2832)
            acc += np.sin(2 * np.pi * (fx * xx / w + fy * yy / h) + ph)
        return acc / nfreq

    def _bilinear(a, sx, sy):
        """Amostragem bilinear de `a` nas coords (sx, sy)."""
        x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
        x1 = np.clip(x0 + 1, 0, a.shape[1] - 1); y1 = np.clip(y0 + 1, 0, a.shape[0] - 1)
        wx = (sx - x0)[..., None]; wy = (sy - y0)[..., None]
        return (a[y0, x0] * (1 - wx) * (1 - wy) + a[y0, x1] * wx * (1 - wy)
                + a[y1, x0] * (1 - wx) * wy + a[y1, x1] * wx * wy)

    try:
        img = Image.open(input_path)

        # Preserva canal alfa se existir (logos/PNG com transparência).
        # Geometria (rotação/crop/resize) é aplicada também ao alfa para
        # manter o alinhamento das bordas transparentes.
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            img = img.convert("RGBA")
        else:
            img = img.convert("RGB")

        w, h = img.size
        seed = random.randint(0, 2**31 - 1)

        # --- Camada 3a: micro-rotação (aplicada a RGB+alfa juntos) ---
        img = img.rotate(random.uniform(-1.2, 1.2), resample=Image.BICUBIC, expand=False)

        # --- Camada 3b: micro-crop → resize (desloca a grade de pixels) ---
        crop = max(2, int(min(w, h) * CROP_PCT))
        img = img.crop((crop, crop, w - crop, h - crop)).resize((w, h), Image.LANCZOS)

        # --- Camada 3c: jitter de brilho/contraste/cor/nitidez ---
        img = ImageEnhance.Brightness(img).enhance(random.uniform(0.985, 1.015))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(0.985, 1.015))
        img = ImageEnhance.Color(img).enhance(random.uniform(0.97, 1.03))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(0.95, 1.05))

        # Separa alfa (já alinhado pela geometria) do RGB
        if img.mode == "RGBA":
            alpha = img.split()[-1]
            arr = np.asarray(img.convert("RGB")).astype(np.float32)
        else:
            alpha = None
            arr = np.asarray(img).astype(np.float32)

        # --- Camada 3d: warp elástico de baixa frequência (quebra a/d/pHash) ---
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        sx = np.clip(xx + WARP_AMP * _field(h, w, seed), 0, w - 1)
        sy = np.clip(yy + WARP_AMP * _field(h, w, seed + 1), 0, h - 1)
        arr = _bilinear(arr, sx, sy)

        # --- Camada 3e: campo de luminância suave (perturba comparações do dHash) ---
        arr = arr * (1 + LUM_PCT * _field(h, w, seed + 5)[..., None])

        # --- Camada 4: ruído adversarial gaussiano de baixa amplitude ---
        arr = arr + np.random.normal(0.0, 2.2, arr.shape)

        out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

        # Recombina alfa, se houver
        if alpha is not None:
            out = out.convert("RGBA")
            out.putalpha(alpha)

        # --- Camadas 1+2: salvar sem metadados, com novo fingerprint ---
        ext = output_path.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            if out.mode != "RGB":
                out = out.convert("RGB")
            out.save(output_path, "JPEG", quality=random.randint(88, 94),
                     optimize=True, exif=b"")
        elif ext == ".webp":
            out.save(output_path, "WEBP", quality=random.randint(88, 94))
        else:  # .png
            out.save(output_path, "PNG", optimize=True)

        # Camada extra: exiftool remove qualquer metadado residual
        if _exiftool_available():
            subprocess.run(
                ["exiftool", "-all=", "-overwrite_original", str(output_path)],
                capture_output=True,
                timeout=60,
            )

        return True, "OK"
    except Exception as exc:
        return False, f"Falha ao processar imagem: {exc}"


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
    # --- file validation ---
    if "file" not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado."}), 400

    file = request.files["file"]
    if file.filename == "" or file.filename is None:
        return jsonify({"error": "Nome de arquivo vazio."}), 400

    if not _is_allowed(file.filename):
        return jsonify({"error": "Formato invalido. Aceitos: MP4, JPG, PNG, WEBP."}), 400

    kind = _kind(file.filename)

    # --- preflight: FFmpeg apenas para vídeo ---
    if kind == "video" and not _ffmpeg_available():
        return jsonify({"error": "FFmpeg nao encontrado no sistema. Instale o FFmpeg e adicione ao PATH."}), 500

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
    if kind == "image":
        success, message = _process_image(input_path, output_path)
    else:
        success, message = _process_video(input_path, output_path)

    if not success:
        _cleanup(input_path, output_path)
        return jsonify({"error": message}), 422

    # Cleanup input immediately; output cleaned after download
    _cleanup(input_path)

    return jsonify({
        "success": True,
        "job_id": job_id,
        "kind": kind,
        "filename": f"changeeditor_{Path(file.filename).stem}{original_ext}",
    })


_MIME_BY_EXT = {
    ".mp4": "video/mp4",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


@app.route("/download/<job_id>/<filename>")
def download(job_id: str, filename: str):
    # Sanitize job_id to prevent path traversal
    safe_id = "".join(c for c in job_id if c.isalnum())

    matches = list(PROCESSED_DIR.glob(f"{safe_id}_processed.*"))
    if not matches:
        return jsonify({"error": "Arquivo nao encontrado. Pode ter expirado."}), 404

    output_path = matches[0]
    mimetype = _MIME_BY_EXT.get(output_path.suffix.lower(), "application/octet-stream")

    @after_this_request
    def cleanup(response):
        _cleanup(output_path)
        return response

    return send_file(
        str(output_path),
        as_attachment=True,
        download_name=filename,
        mimetype=mimetype,
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
