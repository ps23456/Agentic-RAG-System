#!/usr/bin/env python3
"""
Convert a document (PDF, TIFF, image, etc.) to Markdown or HTML using Docling,
then save it under data/structured/ so the Insurance Claim Search app can index it.

Recommended (fixes numpy/opencv and runs conversion):
  ./scripts/run_convert_with_docling.sh data/uploads/Amalgamated_APS_01.tif
  ./scripts/run_convert_with_docling.sh data/uploads/Amalgamated_APS_01.tif --format html

Or manually (after: pip install 'numpy>=1.26.0,<2' 'opencv-python>=4.8,<4.10'):
  python scripts/convert_with_docling.py data/uploads/Amalgamated_APS_01.tif
  python scripts/convert_with_docling.py data/uploads/Amalgamated_APS_01.tif --format html

Output: data/structured/Amalgamated_APS_01.md or data/structured/Amalgamated_APS_01.html
Then in the app: click "Index / Re-index documents" to include the new .md file.
"""

import os
import sys
import json
from pathlib import Path


def _preprocess_for_docling(source: str, project_root: Path) -> str:
    """
    Lightly preprocess image inputs before sending to Docling:
    - Only for raster formats (tif/tiff/png/jpg/jpeg/bmp).
    - Convert each frame to grayscale, autocontrast, upscale ~1.5x.

    Returns the path that should be passed to DocumentConverter (either the
    original source for non-images, or a temporary preprocessed TIFF path).
    """
    suffix = Path(source).suffix.lower()
    if suffix not in (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"):
        return source

    try:
        from PIL import Image, ImageOps, ImageSequence
    except Exception:
        # If Pillow isn't available for some reason, fall back to original.
        return source

    tmp_dir = project_root / "data" / "tmp_docling"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / (Path(source).stem + "_preproc.tif")

    try:
        with Image.open(source) as img:
            frames = []
            for frame in ImageSequence.Iterator(img):
                if frame.mode not in ("L", "RGB"):
                    frame = frame.convert("RGB")
                gray = frame.convert("L")
                gray = ImageOps.autocontrast(gray)
                w, h = gray.size
                scale = 1.5
                gray = gray.resize((int(w * scale), int(h * scale)))
                frames.append(gray)

            if not frames:
                return source

            if len(frames) == 1:
                frames[0].save(tmp_path)
            else:
                frames[0].save(tmp_path, save_all=True, append_images=frames[1:], compression="tiff_deflate")
    except Exception:
        # If anything goes wrong, just use the original.
        return source

    return str(tmp_path)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_with_docling.py <path-to-document> [--format md|html|json]", file=sys.stderr)
        print("Example: python scripts/convert_with_docling.py data/uploads/Amalgamated_APS_01.tif --format html", file=sys.stderr)
        sys.exit(1)

    source = os.path.abspath(sys.argv[1])
    out_format = "md"
    if "--format" in sys.argv:
        try:
            out_format = sys.argv[sys.argv.index("--format") + 1].strip().lower()
        except Exception:
            out_format = "md"
    if out_format not in ("md", "html", "json"):
        print(f"Error: unsupported --format {out_format!r}. Use 'md', 'html', or 'json'.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(source):
        print(f"Error: file not found: {source}", file=sys.stderr)
        sys.exit(2)

    # Project root = parent of scripts/
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "structured"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(source).stem
    if out_format == "md":
        out_ext = "md"
    elif out_format == "html":
        out_ext = "html"
    else:
        out_ext = "json"
    out_path = out_dir / f"{stem}.{out_ext}"

    print(f"Converting: {source}")
    print(f"Output:     {out_path}")

    try:
        from docling.document_converter import DocumentConverter

        # Optional image preprocessing to help Docling/OCR:
        effective_source = _preprocess_for_docling(source, project_root)

        converter = DocumentConverter()
        result = converter.convert(effective_source)
        doc = result.document
        if out_format == "md":
            output_text = doc.export_to_markdown()
        elif out_format == "html":
            # Docling HTML export is via serializer.
            from docling_core.transforms.serializer.html import HTMLDocSerializer

            serializer = HTMLDocSerializer(doc=doc)
            ser_result = serializer.serialize()
            output_text = ser_result.text
        else:
            # JSON (DoclingDocument dict)
            output_text = json.dumps(doc.export_to_dict(), ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Docling conversion failed: {e}", file=sys.stderr)
        print("Tip: Ensure numpy<2 (e.g. pip install 'numpy>=1.26.0,<2' --force-reinstall), then run again.", file=sys.stderr)
        sys.exit(3)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(output_text)

    print(f"Saved to {out_path}")
    print("Next: open the app and click 'Index / Re-index documents' to search this content.")


if __name__ == "__main__":
    main()
