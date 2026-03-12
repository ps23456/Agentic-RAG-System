"""
Standalone OmniDocs OCR test script.
Uses the VLM API backend (no GPU / PyTorch required).

Usage:
    # Activate the test venv first
    source omnidocs_test_env/bin/activate

    # Set at least one API key
    export GEMINI_API_KEY="your-key"       # or
    export OPENAI_API_KEY="your-key"

    # Run on a PDF
    python test_omnidocs_ocr.py data/uploads/APS_TBrown.pdf

    # Run on an image
    python test_omnidocs_ocr.py data/uploads/Amalgamated_APS_01-2.jpg

    # Run on all uploads
    python test_omnidocs_ocr.py data/uploads/
"""

import os
import sys
import time
from pathlib import Path


def _pick_model() -> str:
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini/gemini-2.5-flash"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai/gpt-4o-mini"
    if os.environ.get("GROQ_API_KEY"):
        return "groq/meta-llama/llama-4-scout-17b-16e-instruct"
    print("ERROR: Set GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY.")
    sys.exit(1)


def extract_text_from_image(image_path: str, model: str) -> str:
    from omnidocs.vlm import VLMAPIConfig
    from omnidocs.tasks.text_extraction import VLMTextExtractor
    from PIL import Image

    config = VLMAPIConfig(model=model)
    extractor = VLMTextExtractor(config=config)

    img = Image.open(image_path).convert("RGB")
    result = extractor.extract(img, output_format="markdown")
    return result.content


def extract_text_from_pdf(pdf_path: str, model: str, max_pages: int = 5) -> str:
    from omnidocs import Document
    from omnidocs.vlm import VLMAPIConfig
    from omnidocs.tasks.text_extraction import VLMTextExtractor

    config = VLMAPIConfig(model=model)
    extractor = VLMTextExtractor(config=config)

    doc = Document.from_pdf(pdf_path)
    num_pages = min(doc.page_count, max_pages)

    all_text = []
    for i in range(num_pages):
        print(f"  Processing page {i + 1}/{num_pages} ...")
        page_img = doc.get_page(i)
        result = extractor.extract(page_img, output_format="markdown")
        all_text.append(f"--- PAGE {i + 1} ---\n{result.content}")

    return "\n\n".join(all_text)


def process_file(file_path: str, model: str) -> None:
    ext = Path(file_path).suffix.lower()
    name = Path(file_path).name
    stem = Path(file_path).stem

    print(f"\n{'=' * 60}")
    print(f"File : {name}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")

    start = time.time()

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        text = extract_text_from_image(file_path, model)
    elif ext == ".pdf":
        text = extract_text_from_pdf(file_path, model)
    else:
        print(f"  Skipping unsupported format: {ext}")
        return

    elapsed = time.time() - start

    output_dir = Path("ocr_output")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{stem}_ocr.md"
    output_file.write_text(text, encoding="utf-8")

    print(f"\n--- EXTRACTED TEXT ({elapsed:.1f}s) ---\n")
    print(text)
    print(f"\n--- END ({len(text)} chars) ---")
    print(f"--- SAVED TO: {output_file} ---\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python test_omnidocs_ocr.py <file_or_directory>")
        sys.exit(1)

    target = sys.argv[1]
    model = _pick_model()
    print(f"Using model: {model}")

    target_path = Path(target)

    if target_path.is_dir():
        supported = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".pdf"}
        files = sorted(
            f for f in target_path.iterdir()
            if f.suffix.lower() in supported
        )
        if not files:
            print(f"No supported files found in {target}")
            sys.exit(1)
        print(f"Found {len(files)} file(s) to process.")
        for f in files:
            process_file(str(f), model)
    elif target_path.is_file():
        process_file(target, model)
    else:
        print(f"Path not found: {target}")
        sys.exit(1)


if __name__ == "__main__":
    main()
