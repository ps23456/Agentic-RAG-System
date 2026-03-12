"""
Optional LLM integration for RAG-style insight: use retrieved chunks as context
and get an answer from OpenAI (ChatGPT) or Google Gemini.
Vision: explain_image() uses vision-capable models to describe what's in an image.
Set OPENAI_API_KEY or GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.
"""
import base64
import io
import os
from typing import List, Tuple

# Chunk-like: need .text and .file_name
def _user_prompt(query: str, context: str, include_summary: bool = False) -> str:
    """Build the user message for any question type — insurance claims or financial/annual report docs."""
    base = f"""Document excerpts (retrieved as the most relevant passages for the query below):

{context}

Question / Topic: {query}

Instructions:
- Read ALL the excerpts carefully before answering.
- If the query is a **section heading or topic name** (e.g. "Financial Inclusion and Development", "Risk Management"): 
  Extract and present the COMPLETE content from the matching section(s). Include all sub-points, figures, initiatives, and statements. Do NOT summarize it into one line.
- If the query is a **specific question** (e.g. "What is the return-to-work date?"): 
  Answer directly and precisely using only the information in the excerpts.
- For insurance/medical forms: extract dates, checkboxes (☑/☐), and restriction levels EXACTLY as written.
- For annual reports / financial docs: quote key data, initiatives, and statements directly from the source.
- NEVER infer or add information not present in the excerpts.
- If the answer is not in the excerpts, say: "Not found in provided documents.\""""
    if include_summary:
        base += "\n\nAfter your answer, add a brief **Summary** (1–3 sentences) under a heading: **Summary:**"
    return base


def build_context(chunks: List, max_chars: int = 15000) -> str:
    """Build a single context string from chunk texts for the LLM."""
    parts = []
    total = 0
    for c in chunks:
        fname = getattr(c, 'file_name', 'unknown')
        page = getattr(c, 'page_number', None)
        source_label = f"[Source: {fname}, Page {page}]" if page else f"[Source: {fname}]"
        block = f"{source_label}\n{c.text}\n"
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)
    return "\n---\n".join(parts) if parts else "No relevant passages found."


SYSTEM_PROMPT = """You are a precise document analysis assistant. Answer ONLY the specific question asked using the provided document excerpts.

CRITICAL RULES:

1. UNDERSTAND THE QUESTION CONTEXT:
   - First, determine what kind of question is being asked:
     * Insurance claim questions: about dates, patient names, diagnoses, return-to-work, policy numbers
     * Annual report / financial document questions: about financials, strategies, initiatives, performance, governance
     * General document questions: about any factual content in the excerpts

2. ANSWER FROM THE CONTEXT - EXACTLY:
   - Use ONLY the information in the excerpts above. Do NOT use general knowledge.
   - If the answer is fully present in the excerpts, provide ALL of it — don't truncate or generalize.
   - For annual reports / financial docs: quote key figures, initiatives, and statements directly.
   - For insurance claims: extract dates, checkboxes (☑/☐), and restriction levels EXACTLY as written.
   - Do NOT fabricate, infer, or add information not stated in the excerpts.

3. COMPLETENESS:
   - If multiple sections are relevant, synthesize them into a complete answer.
   - For long answers (e.g. full sections from a report), present ALL relevant content with headings.
   - Do NOT summarize away key details.

4. FORMAT:
   - Use markdown headings and bullet points for structured answers.
   - Use tables for comparative data (dates, checkboxes, financials).
   - Cite source files after the answer: [Source: filename]
   - If not found in the documents: say 'Not found in the provided documents.'"""


CLINICAL_ANALYSIS_PROMPT = """You are a Specialized Medical Review Agent. 
Your task is to provide a structured clinical analysis of the provided medical image(s).

**CLINICAL QUERY**: "{query}"

**REQUIRED STRUCTURE**:
1. **CLINICAL DATA**: (Patient name, claim number, or study identifiers if available)
2. **TECHNIQUE**: (Views and modalities presented in the image(s). If multiple images are provided, list them as Image A, Image B, etc.)
3. **FINDINGS**: (Systematic review of visible anatomical structures, anomalies, or markers. If comparing two images, provide a side-by-side assessment of key regions.)
4. **TEMPORAL COMPARISON / CHANGES**: (Analyze the relationship between the images or between the current image and the provided historical records. Highlight progress, stability, or new developments.)
5. **IMPRESSION**: (Professional summary answering the specific query based on all available visual evidence.)

**INSTRUCTIONS**:
- Use professional medical terminology but remain clear.
- Prioritize answering the user's specific Query.
- If two images are provided, treat them as a comparative set (e.g., "Image A shows... while Image B shows...").
- If historical text/OCR is provided, use it for context.
- **For trauma/bone X-rays (skull, spine, extremities)**: Systematically search for fracture lines. Fractures often appear as linear radiolucencies (dark lines) that are smoother and more regular than sutures (which are jagged/serrated) and do not branch like vascular markings. Describe any such lines and their location; do not dismiss subtle linear lucencies without noting them.
- **DISCLAIMER**: This analysis is for informational purposes only. A medical professional's interpretation is mandatory for definitive assessment."""


GENERIC_IMAGE_DESCRIPTION_PROMPT = """You are an AI assistant examining an image for a search engine index.
Your goal is to provide a comprehensive, highly descriptive caption of this image so that users can find it later using text searches.

Instructions:
1. Describe the main subject, setting, and context.
2. If there are any people, describe their actions or roles (do not guess names unless written).
3. Extract and transcribe any prominent, readable text, signs, logos, or labels exactly as written.
4. If it's a chart, diagram, or UI screenshot, explain what data or interface it represents.
5. If it's a medical scan (like an X-ray or MRI), describe what body part it is and the view (e.g. "Frontal X-ray of a skull").
6. Be highly detailed. Use keywords that someone might type when looking for this exact image.

Do not use conversational filler (like "Here is a description" or "This image shows"). Just output the description directly."""


def get_insight_openai(api_key: str, query: str, context: str, model: str = "gpt-4o-mini", include_summary: bool = False) -> Tuple[str | None, str | None]:
    """Call OpenAI Chat Completions. Returns (response_text, error_message)."""
    if not api_key or not api_key.strip():
        return None, "OpenAI API key is not set."
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(query, context, include_summary=include_summary)},
            ],
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        return (text.strip() if text else None), None
    except Exception as e:
        return None, str(e)


def get_insight_groq(
    api_key: str,
    query: str,
    context: str,
    model: str = "llama-3.3-70b-versatile",
    include_summary: bool = False,
) -> Tuple[str | None, str | None]:
    """
    Call Groq Chat Completions (OpenAI-compatible). Returns (response_text, error_message).
    """
    if not api_key or not api_key.strip():
        return None, "Groq API key is not set."
    try:
        from groq import Groq  # type: ignore[import-not-found]

        client = Groq(api_key=api_key.strip())

        # Try preferred model first, then a lighter fallback if needed.
        tried: list[str] = []
        for m in (model, "llama-3.1-8b-instant"):
            try:
                response = client.chat.completions.create(
                    model=m,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": _user_prompt(query, context, include_summary=include_summary)},
                    ],
                    max_tokens=2000,
                )
                text = response.choices[0].message.content
                if text:
                    # Prepend which model actually answered.
                    return f"(Model: {m})\n\n{text.strip()}", None
            except Exception as inner:
                tried.append(f"{m}: {inner}")

        return None, "Groq call failed for all models tried:\n" + "\n".join(tried)
    except Exception as e:
        return None, str(e)

def _call_gemini_model(api_key: str, model: str, prompt: str) -> Tuple[str | None, str | None]:
    """
    Low-level helper to call a single Gemini model ID.

    Prefers the new google-genai SDK (v1), falls back to the older
    google-generativeai SDK if installed.
    """
    api_key = api_key.strip()
    if not api_key:
        return None, "Gemini API key is not set."

    # First try the new official SDK: google-genai
    try:
        from google import genai  # type: ignore[import-not-found]

        client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(resp, "text", None)
        if not text:
            return None, "Empty response from Gemini (google-genai)."
        return text.strip(), None
    except Exception as e_new:
        last_err = f"google-genai ({model}): {e_new}"

    # Fallback: deprecated google-generativeai SDK (v1beta)
    try:
        import google.generativeai as genai  # type: ignore[import-not-found]

        genai.configure(api_key=api_key)
        gemini = genai.GenerativeModel(model)
        resp = gemini.generate_content(prompt)
        text = getattr(resp, "text", None)
        if not text:
            return None, "Empty response from Gemini (google-generativeai)."
        return text.strip(), None
    except Exception as e_old:
        return None, last_err + f"; google-generativeai ({model}): {e_old}"


def get_insight_gemini(api_key: str, query: str, context: str, model: str = "gemini-1.5-flash", include_summary: bool = False) -> Tuple[str | None, str | None]:
    """
    Call Google Gemini. Returns (response_text, error_message).

    Some API keys / regions only support certain model IDs. We try the requested
    model first, then fall back through a small list of common text models.
    """
    if not api_key or not api_key.strip():
        return None, "Gemini API key is not set."
    prompt = f"{SYSTEM_PROMPT}\n\n{_user_prompt(query, context, include_summary=include_summary)}"

    # Try preferred model + sensible fallbacks.
    # For many free-tier keys, "gemini-pro" (legacy) is still the only text model.
    tried_errors: list[str] = []
    candidate_models = [
        model,
        "gemini-1.5-flash-latest",
        "gemini-1.5-pro",
        "gemini-1.0-pro",
        "gemini-pro",
    ]
    seen = set()
    for m in candidate_models:
        if not m or m in seen:
            continue
        seen.add(m)
        text, err = _call_gemini_model(api_key, m, prompt)
        if text is not None:
            # Indicate which model actually worked in a lightweight way
            return f"(Model: {m})\n\n{text}", None
        if err:
            tried_errors.append(f"{m}: {err}")

    return None, "All Gemini model attempts failed. Details:\n" + "\n".join(tried_errors)


# Default: use "hf/" prefix so the router uses Hugging Face's own inference (avoids Together AI, which can block by IP).
HF_DEFAULT_MODEL = "hf/HuggingFaceH4/zephyr-7b-beta"


def get_insight_huggingface(
    api_key: str,
    query: str,
    context: str,
    model: str = HF_DEFAULT_MODEL,
    include_summary: bool = False,
) -> Tuple[str | None, str | None]:
    """
    Call Hugging Face Router (OpenAI-compatible chat at /v1/chat/completions).
    Returns (response_text, error_message). Uses HUGGINGFACE_API_KEY.
    """
    if not api_key or not api_key.strip():
        return None, "Hugging Face API key is not set."
    import urllib.request
    import json

    user_content = _user_prompt(query, context, include_summary=include_summary)
    url = "https://router.huggingface.co/v1/chat/completions"
    data = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 1024,
    }).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key.strip()}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            out = json.loads(resp.read().decode())
        if isinstance(out, dict) and "choices" in out and len(out["choices"]) > 0:
            msg = out["choices"][0].get("message", {})
            text = msg.get("content") if isinstance(msg, dict) else None
            if text and text.strip():
                return f"(Model: {model})\n\n{text.strip()}", None
            return None, "Empty content from Hugging Face."
        if isinstance(out, dict) and "error" in out:
            err = out["error"]
            msg = err.get("message", err) if isinstance(err, dict) else err
            return None, f"Hugging Face API: {msg}"
        return None, f"Unexpected Hugging Face response: {out}"
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err_json = json.loads(body)
            err = err_json.get("error", {})
            msg = err.get("message", err_json.get("error", body)) if isinstance(err, dict) else err_json.get("error", body)
        except Exception:
            msg = body or str(e)
        return None, f"Hugging Face API ({e.code}): {msg}"
    except Exception as e:
        return None, str(e)


def _image_to_base64(image_path: str, page: int | None = None) -> Tuple[str | None, str, str | None]:
    """
    Load image from file path or PDF page and return (base64_string, mime_type, error).
    If page is set and path ends with .pdf, render that page to PNG and encode.
    """
    if not image_path or not os.path.isfile(image_path):
        return None, "image/jpeg", "Image file not found."
    path_lower = image_path.lower()
    if path_lower.endswith(".pdf") and page is not None:
        try:
            import fitz
            doc = fitz.open(image_path)
            p = doc.load_page(int(page) - 1)
            pix = p.get_pixmap(dpi=150, alpha=False)
            buf = pix.tobytes("png")
            doc.close()
            return base64.b64encode(buf).decode("utf-8"), "image/png", None
        except Exception as e:
            return None, "image/png", str(e)
    try:
        with open(image_path, "rb") as f:
            raw = f.read()
        b64 = base64.b64encode(raw).decode("utf-8")
        if path_lower.endswith(".png"):
            return b64, "image/png", None
        if path_lower.endswith(".gif"):
            return b64, "image/gif", None
        return b64, "image/jpeg", None
    except Exception as e:
        return None, "image/jpeg", str(e)


VISION_SYSTEM_PROMPT = """You are an assistant that explains what is shown in images, diagrams, and screenshots. Describe the content clearly: labels, flow, steps, and structure. If the user asks a specific question (e.g. "explain this diagram"), answer that question using what you see in the image. Use markdown for structure (lists, headings) when helpful."""


def explain_image_openai(api_key: str, query: str, image_path: str, page: int | None = None) -> Tuple[str | None, str | None]:
    """Use OpenAI vision (gpt-4o) to explain what's in the image. Returns (text, error)."""
    if not api_key or not api_key.strip():
        return None, "OpenAI API key is not set."
    b64, mime, err = _image_to_base64(image_path, page)
    if err:
        return None, err
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        url = f"data:{mime};base64,{b64}"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": query or "Describe what is in this image in detail."},
                    {"type": "image_url", "image_url": {"url": url}},
                ]},
            ],
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        return (text.strip() if text else None), None
    except Exception as e:
        return None, str(e)


def explain_image_gemini(api_key: str, query: str, image_path: str, page: int | None = None) -> Tuple[str | None, str | None]:
    """Use Gemini vision to explain what's in the image. Returns (text, error)."""
    if not api_key or not api_key.strip():
        return None, "Gemini API key is not set."
    b64, mime, err = _image_to_base64(image_path, page)
    if err:
        return None, err
    prompt = (query or "Describe what is in this image in detail.").strip()
    api_key = api_key.strip()
    # New SDK: google-genai with inline image (inline_data + Blob)
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
        image_bytes = base64.b64decode(b64)
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(inline_data=types.Blob(data=image_bytes, mime_type=mime)),
                    types.Part(text=prompt),
                ],
            ),
        ]
        resp = client.models.generate_content(model="gemini-1.5-flash", contents=contents)
        text = getattr(resp, "text", None)
        if not text:
            return None, "Empty response from Gemini."
        return text.strip(), None
    except Exception as e_new:
        pass
    # Fallback: google-generativeai (takes PIL Image)
    try:
        import google.generativeai as genai
        from PIL import Image as PILImage
        genai.configure(api_key=api_key)
        img = PILImage.open(io.BytesIO(base64.b64decode(b64)))
        model = genai.GenerativeModel("gemini-1.5-flash")
        resp = model.generate_content([prompt, img])
        text = getattr(resp, "text", None)
        if not text:
            return None, "Empty response from Gemini."
        return text.strip(), None
    except Exception as e_old:
        return None, str(e_old)


def explain_image_groq(api_key: str, query: str, image_path: str, page: int | None = None) -> Tuple[str | None, str | None]:
    """Use Groq vision model to explain what's in the image. Returns (text, error)."""
    if not api_key or not api_key.strip():
        return None, "Groq API key is not set."
    b64, mime, err = _image_to_base64(image_path, page)
    if err:
        return None, err
    try:
        from groq import Groq
        client = Groq(api_key=api_key.strip())
        url = f"data:{mime};base64,{b64}"
        # Use meta-llama/llama-4-scout-17b-16e-instruct (current Groq vision model)
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": query or "Describe what is in this image in detail."},
                    {"type": "image_url", "image_url": {"url": url}},
                ]},
            ],
            max_tokens=2000,
        )
        text = response.choices[0].message.content
        return (text.strip() if text else None), None
    except Exception as e:
        return None, str(e)


CLASSIFY_MEDICAL_PROMPT = """Look at this medical image carefully. What type of medical imaging or document is it?

Choose exactly ONE from this list (reply with only that word, nothing else):
- X-ray (grayscale bone/chest images, typical radiology)
- MRI (soft tissue, brain, spine, often multiple slices)
- CT Scan (cross-sectional body scans)
- Medical Report (document with text, form, or PDF)
- Ultrasound (fluid/organs, often with Doppler)
- Other (only if none of the above fit)

Be decisive. Look at the visual style: X-rays are grayscale with bones; MRI has different contrast; CT shows cross-sections. Reply with ONLY the single category word."""


def _parse_classification(raw: str) -> str:
    """Parse LLM response to extract category. Handles varied formats."""
    if not raw:
        return "Other"
    raw_lower = raw.lower().strip()
    # Normalize: remove punctuation, collapse spaces
    raw_clean = "".join(c for c in raw_lower if c.isalnum() or c.isspace())
    # Match in order of specificity (avoid "other" matching first)
    if "xray" in raw_clean or "x-ray" in raw_lower or "x ray" in raw_lower:
        return "X-ray"
    if "mri" in raw_clean or "magnetic resonance" in raw_lower:
        return "MRI"
    if "ctscan" in raw_clean or "ct scan" in raw_lower or "ctscan" in raw_lower or ("ct" in raw_clean and "scan" in raw_clean):
        return "CT Scan"
    if "ultrasound" in raw_clean or "sonogram" in raw_lower:
        return "Ultrasound"
    if "medicalreport" in raw_clean or "medical report" in raw_lower or "document" in raw_lower or "pdf" in raw_lower or "form" in raw_lower:
        return "Medical Report"
    return "Other"


def classify_medical_document(provider: str, api_key: str, image_path: str, page: int = 1) -> Tuple[str | None, str | None]:
    """
    Use vision LLM to classify a medical image/PDF into: X-ray, MRI, CT Scan, Medical Report, Ultrasound, Other.
    Returns (category_string, error_message). category is normalized to our display names.
    """
    if not api_key or not api_key.strip():
        return None, "API key is not set. Use OpenAI, Gemini, or Groq for classification."
    b64, mime, err = _image_to_base64(image_path, page if image_path.lower().endswith(".pdf") else None)
    if err:
        return None, err
    url = f"data:{mime};base64,{b64}"
    prompt = CLASSIFY_MEDICAL_PROMPT
    raw = None
    if provider == "openai":
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key.strip())
            r = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}],
                max_tokens=50,
            )
            raw = (r.choices[0].message.content or "").strip()
        except Exception as e:
            return None, str(e)
    elif provider == "gemini":
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key.strip(), http_options={"api_version": "v1"})
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents=[types.Content(role="user", parts=[
                    types.Part(inline_data=types.Blob(data=base64.b64decode(b64), mime_type=mime)),
                    types.Part(text=prompt),
                ])],
            )
            raw = (getattr(resp, "text", None) or "").strip()
        except Exception as e:
            return None, str(e)
    elif provider == "groq":
        try:
            from groq import Groq
            client = Groq(api_key=api_key.strip())
            r = client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": prompt}]}],
                max_tokens=100,
            )
            raw = (r.choices[0].message.content or "").strip()
        except Exception as e:
            return None, str(e)
    elif provider == "mistral":
        try:
            from mistralai import Mistral
            client = Mistral(api_key=api_key.strip())
            r = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": url}, {"type": "text", "text": prompt}]}],
                max_tokens=100,
            )
            raw = (r.choices[0].message.content or "").strip()
        except Exception as e:
            return None, str(e)
    else:
        return None, f"Classification not supported for provider: {provider}"
    if not raw:
        return None, "Empty classification response."
    return _parse_classification(raw), None


def explain_image(provider: str, api_key: str, query: str, image_path: str, page: int | None = None) -> Tuple[str | None, str | None]:
    """
    Explain what is in an image using a vision-capable LLM.
    provider: "openai" | "gemini" | "groq"
    image_path: path to image file or PDF (if PDF, set page to the 1-based page number).
    Returns (explanation_text, error_message).
    """
    if provider == "openai":
        return explain_image_openai(api_key, query, image_path, page)
    if provider == "gemini":
        return explain_image_gemini(api_key, query, image_path, page)
    if provider == "groq":
        return explain_image_groq(api_key, query, image_path, page)
    return None, f"Vision not supported for provider: {provider}. Use OpenAI, Gemini, or Groq."


def medical_analysis_openai(api_key: str, query: str, image_paths: List[str], pages: List[int | None] = None, history_context: str = "") -> Tuple[str | None, str | None]:
    """Analyze one or more medical images with OpenAI Vision, including historical context."""
    if not api_key:
        return None, "OpenAI API key is not set."
    
    # Process all images
    user_content = [{"type": "text", "text": CLINICAL_ANALYSIS_PROMPT.format(query=query) + (f"\n\n**HISTORICAL CONTEXT (from past reports/OCR)**:\n{history_context}" if history_context else "")}]
    
    for i, path in enumerate(image_paths):
        page = pages[i] if pages and i < len(pages) else None
        b64, mime, err = _image_to_base64(path, page)
        if err:
            return None, f"Error processing image {i+1}: {err}"
        label = "Image A" if len(image_paths) > 1 and i == 0 else ("Image B" if len(image_paths) > 1 and i == 1 else "Image")
        user_content.append({"type": "text", "text": f"--- {label} ---"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key.strip())
        response = client.chat.completions.create(
            model="gpt-4o", # Use 4o for medical vision
            messages=[{"role": "user", "content": user_content}],
            max_tokens=2500,
        )
        text = response.choices[0].message.content
        return (text.strip() if text else None), None
    except Exception as e:
        return None, str(e)


def medical_analysis_gemini(api_key: str, query: str, image_paths: List[str], pages: List[int | None] = None, history_context: str = "") -> Tuple[str | None, str | None]:
    """Analyze one or more medical images with Gemini Vision."""
    if not api_key:
        return None, "Gemini API key is not set."
    
    prompt = CLINICAL_ANALYSIS_PROMPT.format(query=query)
    if history_context:
        prompt += f"\n\n**HISTORICAL CONTEXT (from past reports/OCR)**:\n{history_context}"
    
    api_key = api_key.strip()
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
        
        parts = []
        for i, path in enumerate(image_paths):
            page = pages[i] if pages and i < len(pages) else None
            b64, mime, err = _image_to_base64(path, page)
            if err: return None, f"Error processing image {i+1}: {err}"
            label = "Image A" if len(image_paths) > 1 and i == 0 else ("Image B" if len(image_paths) > 1 and i == 1 else "Image")
            parts.append(types.Part(text=f"--- {label} ---"))
            parts.append(types.Part(inline_data=types.Blob(data=base64.b64decode(b64), mime_type=mime)))
        parts.append(types.Part(text=prompt))
        
        contents = [types.Content(role="user", parts=parts)]
        resp = client.models.generate_content(model="gemini-1.5-pro", contents=contents) 
        text = getattr(resp, "text", None)
        return (text.strip() if text else None), None
    except Exception:
        # Fallback omitted for brevity/multi-image complexity in old SDK
        return None, "Gemini multi-image analysis requires the new google-genai SDK."


def medical_analysis_groq(api_key: str, query: str, image_paths: List[str], pages: List[int | None] = None, history_context: str = "") -> Tuple[str | None, str | None]:
    """Analyze one or more medical images with Groq Vision."""
    if not api_key:
        return None, "Groq API key is not set."
    
    prompt = CLINICAL_ANALYSIS_PROMPT.format(query=query)
    if history_context:
        prompt += f"\n\n**HISTORICAL CONTEXT (from past reports/OCR)**:\n{history_context}"

    user_content = [{"type": "text", "text": prompt}]
    for i, path in enumerate(image_paths):
        page = pages[i] if pages and i < len(pages) else None
        b64, mime, err = _image_to_base64(path, page)
        if err: return None, f"Error processing image {i+1}: {err}"
        label = "Image A" if len(image_paths) > 1 and i == 0 else ("Image B" if len(image_paths) > 1 and i == 1 else "Image")
        user_content.append({"type": "text", "text": f"--- {label} ---"})
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    try:
        from groq import Groq
        client = Groq(api_key=api_key.strip())
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct", # Standard Vision model
            messages=[{"role": "user", "content": user_content}],
            max_tokens=2500,
        )
        text = response.choices[0].message.content
        return (text.strip() if text else None), None
    except Exception as e:
        return None, str(e)


def medical_analysis_mistral(api_key: str, query: str, image_paths: List[str], pages: List[int | None] = None, history_context: str = "") -> Tuple[str | None, str | None]:
    """Analyze one or more medical images with Mistral Vision."""
    if not api_key:
        return None, "Mistral API key is not set."

    prompt = CLINICAL_ANALYSIS_PROMPT.format(query=query)
    if history_context:
        prompt += f"\n\n**HISTORICAL CONTEXT (from past reports/OCR)**:\n{history_context}"

    content: List[dict] = [{"type": "text", "text": prompt}]
    for i, path in enumerate(image_paths):
        page = pages[i] if pages and i < len(pages) else None
        b64, mime, err = _image_to_base64(path, page)
        if err:
            return None, f"Error processing image {i+1}: {err}"
        label = "Image A" if len(image_paths) > 1 and i == 0 else ("Image B" if len(image_paths) > 1 and i == 1 else "Image")
        content.append({"type": "text", "text": f"--- {label} ---"})
        content.append({"type": "image_url", "image_url": f"data:{mime};base64,{b64}"})

    try:
        from mistralai import Mistral
        client = Mistral(api_key=api_key.strip())
        r = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": content}],
            max_tokens=2500,
        )
        text = r.choices[0].message.content
        return (text.strip() if text else None), None
    except Exception as e:
        return None, str(e)


def medical_analysis(provider: str, api_key: str, query: str, image_paths: List[str], pages: List[int | None] = None, history_context: str = "") -> Tuple[str | None, str | None]:
    """
    Perform specialized medical analysis and temporal comparison for one or more images.
    provider: "openai" | "gemini" | "groq" | "mistral"
    """
    if provider == "openai":
        return medical_analysis_openai(api_key, query, image_paths, pages, history_context)
    if provider == "gemini":
        return medical_analysis_gemini(api_key, query, image_paths, pages, history_context)
    if provider == "groq":
        return medical_analysis_groq(api_key, query, image_paths, pages, history_context)
    if provider == "mistral":
        return medical_analysis_mistral(api_key, query, image_paths, pages, history_context)
    return None, f"Medical analysis not supported for provider: {provider}"


def generate_image_description(provider: str, api_key: str, image_path: str, page: int | None = None) -> Tuple[str | None, str | None]:
    """
    Generate a rich semantic description of an image for indexing and search.
    """
    query = GENERIC_IMAGE_DESCRIPTION_PROMPT
    if provider.lower() == "openai":
        return medical_analysis_openai(api_key, query, [image_path], [page], "")
    elif provider.lower() == "gemini":
        return medical_analysis_gemini(api_key, query, [image_path], [page], "")
    elif provider.lower() == "groq":
        return medical_analysis_groq(api_key, query, [image_path], [page], "")
    
    return None, f"Image description generation not supported for provider: {provider}"


def get_insight(provider: str, api_key: str, query: str, chunks: List, max_context_chars: int = 15000, include_summary: bool = False) -> Tuple[str | None, str | None]:
    """
    Get an LLM-generated answer using retrieved chunks as context.
    provider: "openai" | "gemini" | "groq" | "huggingface"
    include_summary: if True, append a brief summary at the end of the answer.
    Returns (response_text, error_message). If error_message is set, response_text is None.
    """
    context = build_context(chunks, max_chars=max_context_chars)
    if provider == "openai":
        return get_insight_openai(api_key, query, context, include_summary=include_summary)
    if provider == "gemini":
        return get_insight_gemini(api_key, query, context, include_summary=include_summary)
    if provider == "groq":
        return get_insight_groq(api_key, query, context, include_summary=include_summary)
    if provider == "huggingface":
        return get_insight_huggingface(api_key, query, context, include_summary=include_summary)
    return None, f"Unknown provider: {provider}"
