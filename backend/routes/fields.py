"""Field extraction endpoints for form-like PDFs/Markdown."""
from __future__ import annotations

import os
import re
from typing import Any

from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_ORDERED_FIELD_PATTERNS: list[tuple[str, str]] = [
    ("Name", r"\bname\b"),
    ("Claim Number", r"\bclaim\s*number\b"),
    ("Policy Number", r"\bpolicy\s*number\b"),
    ("Date", r"\bdate\b"),
    ("Educational Level", r"\beducation(?:al)?\s+level\b"),
    ("Yes / No", r"\byes(?:\s|/|☐|☑|x|X){0,8}no\b"),
    ("If no, what was the last grade you attended?", r"if\s+no[, ]+what\s+was\s+the\s+last\s+grade\s+you\s+attended"),
    ("Grade School Graduate", r"grade\s+school\s+graduate"),
    ("High School Graduate", r"high\s+school\s+graduate"),
    ("GED", r"\bged\b"),
    ("College Graduate", r"college\s+graduate"),
    ("Years completed", r"years?\s+completed"),
    ("Major", r"\bmajor\b"),
    ("Degree and Year obtained", r"degree\s+and\s+year\s+obtained"),
    ("Post Graduate", r"post\s+graduate"),
    ("Training", r"\btraining\b"),
    ("Please list any additional training you have received", r"please\s+list\s+any\s+additional\s+training\s+you\s+have\s+received"),
    ("List any certifications or licenses", r"list\s+any\s+certifications?\s+or\s+licenses"),
    ("Do you currently have a valid Drivers license?", r"do\s+you\s+currently\s+have\s+a\s+valid\s+drivers?\s+license"),
    ("Do you have an active CDL license?", r"do\s+you\s+have\s+an\s+active\s+cdl\s+license"),
    ("Are you working on seeking employment?", r"are\s+you\s+working\s+on\s+seeking\s+employment"),
    ("Military Experience", r"military\s+experience"),
    ("Dates of Service", r"dates?\s+of\s+service"),
    ("Branch of Service", r"branch\s+of\s+service"),
    ("Highest Rank", r"highest\s+rank"),
    ("Special training/How was skill used?", r"special\s+training/?how\s+was\s+skill\s+used"),
    ("Work History and Experience", r"work\s+history\s+and\s+experience"),
    ("Job Title and Employer", r"job\s+title\s+and\s+employer"),
    ("Dates of Employment", r"dates?\s+of\s+employment"),
    ("Salary", r"\bsalary\b"),
    ("Describe Duties/Responsibilities", r"describe\s+duties/?responsibilities"),
    ("Do you have experience operating a computer?", r"do\s+you\s+have\s+experience\s+operating\s+a\s+computer"),
    ("Do you use a computer for internet searches?", r"do\s+you\s+use\s+a\s+computer\s+for\s+internet\s+searches"),
    ("Have you ever input numbers or text into an automated system?", r"have\s+you\s+ever\s+input\s+numbers?\s+or\s+text\s+into\s+an\s+automated\s+system"),
    ("Word processing skills", r"word\s+processing\s+skills"),
    ("Spreadsheet experience", r"spreadsheet\s+experience"),
    ("Database experience", r"database\s+experience"),
    ("Do you use a smartphone?", r"do\s+you\s+use\s+a\s+smartphone"),
    ("Do you text or send emails?", r"do\s+you\s+text\s+or\s+send\s+emails"),
    ("Do you use Apps on your smartphone?", r"do\s+you\s+use\s+apps?\s+on\s+your\s+smartphone"),
    ("Do you have experience assisting customers, in-person or telephonically?", r"in-?person\s+or\s+telephonically"),
    ("Experience interviewing others?", r"experience\s+interviewing\s+others"),
    ("Typing, data entry, or keyboarding (WPM)?", r"typing[, ]+data\s+entry[, ]+or\s+keyboarding"),
    ("Have you ever used voice activated software?", r"voice\s+activated\s+software"),
    ("Experience selling product or service (indicate type)?", r"experience\s+selling\s+product\s+or\s+service"),
    ("Filing, classifying or organizing?", r"filing[, ]+classifying\s+or\s+organizing"),
    ("Experience composing letters or reports?", r"experience\s+composing\s+letters?\s+or\s+reports?"),
    ("Experience tabulating/composing bills or invoices?", r"tabulating/?composing\s+bills?\s+or\s+invoices"),
    ("Presentation Skills?", r"presentation\s+skills"),
    ("Experience with record-keeping?", r"record-keeping"),
    ("Experience with shipping/receiving or monitoring inventory?", r"shipping/?receiving\s+or\s+monitoring\s+inventory"),
    ("Do you belong to any professional association memberships?", r"professional\s+association\s+memberships"),
    ("Do you perform any volunteer work including community volunteerwork?", r"(?:do|o)\s+you\s+perform\s+any\s+volunteer\s+work(?:.|\s){0,120}?(?:community\s+volunteerwork|school/work/recreation)"),
    ("Do you speak other languages, please specify?", r"do\s+you\s+speak\s+other\s+languages"),
    ("Do you understand other languages, please specify?", r"do\s+you\s+understand\s+other\s+languages"),
    ("Are you currently engaged in a vocational retraining program?", r"currently\s+engaged\s+in\s+a\s+vocational\s+retraining\s+program"),
    ("Personal interests / occupational interests / hobbies", r"briefly\s+describe\s+your\s+personal\s+interests"),
    ("Acknowledgement", r"\backnowledg(e)?ment\b"),
    ("Signature", r"\bsignature\b"),
]

_SCHEMA_RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"^name$", re.I), "patientName", "Full name of the patient/insured. Find in the top header row labeled 'Name'."),
    (re.compile(r"^claim number$", re.I), "claimNumber", "Claim identifier. Find in the header row labeled 'Claim Number' below the Name field."),
    (re.compile(r"^policy number$", re.I), "policyNumber", "Policy identifier. Find in the header row labeled 'Policy Number'."),
    (re.compile(r"^date$", re.I), "formDate", "Form date field in the top header row labeled 'Date'."),
    (re.compile(r"^educational level$", re.I), "educationalLevel", "Education section row label 'Educational Level' (checkbox/status field)."),
    (re.compile(r"^yes\s*/?\s*no$", re.I), "yesNoCheckbox", "Checkbox column label used in form tables. For each related row, extract which option is checked: Yes or No."),
    (re.compile(r"if no, what was the last grade you attended", re.I), "lastGradeIfNotGraduate", "Education section prompt asking for the last grade attended if not graduate."),
    (re.compile(r"^grade school graduate$", re.I), "gradeSchoolGraduate", "Education checkbox row for Grade School Graduate. Return checked option: Yes or No."),
    (re.compile(r"^high school graduate$", re.I), "highSchoolGraduate", "Education checkbox row for High School Graduate. Return checked option: Yes or No."),
    (re.compile(r"^ged$", re.I), "ged", "Education checkbox row for GED. Return checked option: Yes or No."),
    (re.compile(r"^college graduate$", re.I), "collegeGraduate", "Education checkbox row for College Graduate. Return checked option: Yes or No."),
    (re.compile(r"^years completed$", re.I), "yearsCompleted", "Education subsection field labeled 'Years completed'."),
    (re.compile(r"^major$", re.I), "major", "Education subsection field labeled 'Major'."),
    (re.compile(r"degree and year obtained", re.I), "degreeAndYearObtained", "Education subsection field labeled 'Degree and Year obtained'."),
    (re.compile(r"^post graduate", re.I), "postGraduate", "Education checkbox row for Post Graduate. Return checked option: Yes or No."),
    (re.compile(r"please list any additional training", re.I), "additionalTraining", "Training section text-entry line: 'Please list any additional training you have received'."),
    (re.compile(r"list any certifications? or licenses", re.I), "certificationsOrLicenses", "Training section text-entry line: 'List any certifications or licenses'."),
    (re.compile(r"valid drivers? license", re.I), "validDriversLicense", "Yes/No checkbox question: 'Do you currently have a valid Drivers license?'"),
    (re.compile(r"active cdl license", re.I), "activeCdlLicense", "Yes/No checkbox question: 'Do you have an active CDL license?'"),
    (re.compile(r"seeking employment", re.I), "seekingEmployment", "Yes/No question in Training section: 'Are you working on seeking employment?'"),
    (re.compile(r"^military experience$", re.I), "militaryExperience", "Military Experience section heading/value area."),
    (re.compile(r"^dates of service$", re.I), "datesOfService", "Military Experience line labeled 'Dates of Service'."),
    (re.compile(r"^branch of service$", re.I), "branchOfService", "Military Experience line labeled 'Branch of Service'."),
    (re.compile(r"^highest rank$", re.I), "highestRank", "Military Experience line labeled 'Highest Rank'."),
    (re.compile(r"special training/?how was skill used", re.I), "specialTrainingHowSkillUsed", "Military Experience line labeled 'Special training/How was skill used?'."),
    (re.compile(r"work history and experience", re.I), "workHistoryAndExperience", "Section heading 'Work History and Experience'."),
    (re.compile(r"job title and employer", re.I), "jobTitleAndEmployer", "Work History table column: 'Job Title and Employer'."),
    (re.compile(r"dates of employment", re.I), "datesOfEmployment", "Work History table column: 'Dates of Employment'."),
    (re.compile(r"^salary$", re.I), "salary", "Work History table column: 'Salary'."),
    (re.compile(r"describe duties/?responsibilities", re.I), "dutiesResponsibilities", "Work History table column: 'Describe Duties/Responsibilities...'."),
    (re.compile(r"experience operating a computer", re.I), "computerExperience", "Skills question about operating a computer (Mac/PC/tablet)."),
    (
        re.compile(r"use a computer for internet searches", re.I),
        "computerInternetSearches",
        "Computer/Technology Skills section (page 2): one long Yes/No row whose text starts with "
        "'Do you use a computer for internet searches' and continues with examples "
        "(e.g. google, on-line banking, job search, social media, on-line shopping, games/videos, read articles/books on-line). "
        "On the same printed line it ends with 'Do you have an email account?' — extract the Yes/No (and any tick) for this combined row; "
        "do not treat email as a separate field label on the form.",
    ),
    (re.compile(r"input numbers? or text into an automated system", re.I), "automatedSystemInputExperience", "Question about entering numbers/text into automated systems (e.g., cash register, library/bookstore search)."),
    (re.compile(r"word processing skills", re.I), "wordProcessingSkills", "Skills question: Word processing skills (Microsoft Office/Word/PowerPoint)."),
    (re.compile(r"spreadsheet experience", re.I), "spreadsheetExperience", "Skills question: Spreadsheet experience (e.g., Excel)."),
    (re.compile(r"database experience", re.I), "databaseExperience", "Skills question: Database/scheduling/inventory experience."),
    (re.compile(r"do you use a smartphone", re.I), "usesSmartphone", "Skills question: smartphone usage."),
    (re.compile(r"text or send emails", re.I), "textsOrEmails", "Skills question: texting/email usage."),
    (re.compile(r"use apps? on your smartphone", re.I), "smartphoneApps", "Skills question: app usage on smartphone."),
    (re.compile(r"assisting customers,? in-?person or telephonically", re.I), "customerAssistInPersonOrPhone", "Skills question about assisting customers in-person or telephonically."),
    (re.compile(r"interviewing others", re.I), "interviewingExperience", "Skills question: experience interviewing others."),
    (re.compile(r"typing,? data entry,? or keyboarding", re.I), "typingDataEntryKeyboarding", "Skills question: typing/data-entry/keyboarding (WPM)."),
    (re.compile(r"voice activated software", re.I), "voiceActivatedSoftware", "Skills question: experience with voice-activated software."),
    (re.compile(r"selling product or service", re.I), "salesExperience", "Skills question: experience selling product or service."),
    (re.compile(r"filing,? classifying or organizing", re.I), "filingClassifyingOrganizing", "Skills question: filing/classifying/organizing experience."),
    (re.compile(r"composing letters? or reports?", re.I), "lettersReportsExperience", "Skills question: composing letters/reports."),
    (re.compile(r"tabulating/?composing bills? or invoices", re.I), "billsInvoicesExperience", "Skills question: tabulating/composing bills or invoices."),
    (re.compile(r"presentation skills", re.I), "presentationSkills", "Skills question: presentation skills."),
    (re.compile(r"record-keeping", re.I), "recordKeepingExperience", "Skills question: record-keeping experience."),
    (re.compile(r"shipping/?receiving or monitoring inventory", re.I), "shippingReceivingInventory", "Skills question: shipping/receiving/monitoring inventory."),
    (re.compile(r"professional association memberships", re.I), "professionalMemberships", "Question: membership in professional associations."),
    (re.compile(r"volunteer work", re.I), "volunteerWork", "Question about volunteer work (school/work/recreation/community)."),
    (re.compile(r"speak other languages", re.I), "speakOtherLanguages", "Question: 'Do you speak other languages, please specify?'"),
    (re.compile(r"understand other languages", re.I), "understandOtherLanguages", "Question: 'Do you understand other languages, please specify?'"),
    (re.compile(r"vocational retraining program", re.I), "vocationalRetrainingProgram", "Question about vocational retraining via workers' compensation/rehab."),
    (re.compile(r"personal interests", re.I), "personalOccupationalInterests", "Prompt to describe personal/occupational interests and hobbies."),
    (re.compile(r"^acknowledgement$", re.I), "acknowledgement", "Acknowledgement section certifying responses are true and complete."),
    (re.compile(r"^signature$", re.I), "signature", "Signature line in the Acknowledgement section."),
]

_HEADING_ONLY_LABELS = {
    "training",
    "education",
    "military experience",
    "work history and experience",
    "acknowledgement",
    "personal interests",
    "personal interests / occupational interests / hobbies",
}


def _find_file(file_name: str) -> str | None:
    from backend.services.rag_service import rag

    for root, _dirs, files in os.walk(rag.data_folder):
        if file_name in files:
            return os.path.join(root, file_name)
    return None


def _clean_label(label: str) -> str:
    x = (label or "").strip()
    x = x.replace("*", "")
    x = re.sub(r"\[[xX ]\]", "", x)
    x = re.sub(r"\s+", " ", x).strip(" -:;|")
    x = re.sub(r"^\d+\.\s*", "", x)
    return x


def _normalize_field_label(label: str) -> str:
    """Trim OCR values from labels so we keep field names only."""
    x = _clean_label(label)
    low = x.lower()
    if low.startswith("name "):
        return "Name"
    if "last grade you attended" in low:
        return "If no, what was the last grade you attended?"
    if low.startswith("list any certifications or licenses"):
        return "List any certifications or licenses"
    if re.match(r"^no\s+do you have an active cdl license", low):
        return "Do you have an active CDL license?"
    if re.match(r"^are you working on seeking employment.*explain", low):
        return "Are you working on seeking employment?"
    if low == "classifying or organizing?":
        return "Filing, classifying or organizing?"
    if low.startswith("claim number "):
        return "Claim Number"
    if low.startswith("policy number "):
        return "Policy Number"
    if low.startswith("years completed "):
        return "Years completed"
    if low.startswith("major "):
        return "Major"
    if low.startswith("degree and year obtained "):
        return "Degree and Year obtained"
    # Generic: remove obvious trailing pure values after known label keywords.
    x = re.sub(r"^(.*?\b(?:date|salary|from|to|branch of service|highest rank|job title and employer)\b)\s+.+$", r"\1", x, flags=re.IGNORECASE)
    return _clean_label(x)

def _normalize_text_for_match(text: str) -> str:
    t = (text or "")
    t = t.replace("&amp;", "&")
    # Fix OCR word breaks like "D o you", "i n", "o r"
    t = re.sub(r"\b([A-Za-z])\s+([A-Za-z])\b", r"\1\2", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def _is_probable_field(label: str) -> bool:
    l = _normalize_field_label(label)
    if not l:
        return False
    if len(l) < 2 or len(l) > 90:
        return False
    # Exclude instruction paragraphs, but keep actual field prompts (including "Please list ...").
    if re.search(r"describe each job worked|resumes are appreciated", l, re.IGNORECASE):
        return False
    if re.fullmatch(r"yes\s*no|yes\s*☐\s*no\s*☐|yes\s*☑\s*no\s*☐|yes\s*☐\s*no\s*☑", l, re.IGNORECASE) and l.strip().lower() != "yes / no":
        return False
    if l.lower() in {"---", "n/a", "na"}:
        return False
    if re.search(r"\b(?:car sales|big car sales|sell cars)\b", l, re.IGNORECASE):
        return False
    if re.match(r"^(from|to)\s*:?\s*\d", l, re.IGNORECASE):
        return False
    if re.fullmatch(r"from|to|yes|no", l, re.IGNORECASE):
        return False
    if re.fullmatch(r"(tablet|excel|etc|fax|phone)\)?\??", l, re.IGNORECASE):
        return False
    if re.fullmatch(r"(power point|scheduling/inventory)\)?\??", l, re.IGNORECASE):
        return False
    # OCR-tail fragments only (avoid blocking full question lines that contain these phrases).
    if re.fullmatch(r"(read articles/books on-line\??|search engine like in a library or bookstore\??|school/work/recreation\??)", l, re.IGNORECASE):
        return False
    if re.search(r"please explain your use of this tool at work or home", l, re.IGNORECASE):
        return False
    if re.search(r"please explain your experience as well as your length of experience", l, re.IGNORECASE):
        return False
    if re.search(r"do you have an email account\??", l, re.IGNORECASE):
        return False
    if l.strip().lower() in _HEADING_ONLY_LABELS:
        return False
    if sum(ch.isalpha() for ch in l) < 2:
        return False
    return True


def _to_key(label: str, used: set[str]) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", label)
    if not tokens:
        base = "field"
    else:
        first = tokens[0].lower()
        rest = [t[:1].upper() + t[1:].lower() for t in tokens[1:]]
        base = first + "".join(rest)
    k = base
    i = 2
    while k in used:
        k = f"{base}{i}"
        i += 1
    used.add(k)
    return k

def _label_signature(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (label or "").lower())

def _canonical_label(candidate: str) -> str:
    """
    Map noisy/expanded OCR label text to one canonical field label based on
    ordered field patterns. This prevents duplicate variants in output.
    """
    c = _normalize_field_label(candidate)
    if not c:
        return c
    norm = _normalize_text_for_match(c)
    for canonical, pat in _ORDERED_FIELD_PATTERNS:
        if re.search(pat, norm, re.IGNORECASE):
            return canonical
    return c

def _schema_for_label(label: str, used: set[str]) -> tuple[str, str]:
    for pat, key, desc in _SCHEMA_RULES:
        if pat.search(label or ""):
            out = key
            i = 2
            while out in used:
                out = f"{key}{i}"
                i += 1
            used.add(out)
            return out, desc
    # Fallback generic
    key = _to_key(label, used)
    return key, f"Value entered for '{label}' in the uploaded form. Locate the matching label text and extract the adjacent entry/checkbox value."


def _extract_field_names(text: str) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    norm_text = _normalize_text_for_match(text)

    # 0) High-precision ordered extraction from known field patterns.
    for label, pat in _ORDERED_FIELD_PATTERNS:
        if re.search(pat, norm_text, re.IGNORECASE):
            if not _is_probable_field(label):
                continue
            key = _label_signature(label)
            if key not in seen:
                seen.add(key)
                found.append(label)

    # 1) Table cells from OCR markdown
    for line in (text or "").splitlines():
        if "|" in line:
            cells = [c.strip() for c in line.split("|")]
            for c in cells:
                c = _canonical_label(c)
                if not _is_probable_field(c):
                    continue
                low = _label_signature(c)
                if low in seen:
                    continue
                seen.add(low)
                found.append(c)

    # 2) Prompt-like lines and labels with colon/question-mark
    patterns = [
        r"([A-Za-z][A-Za-z0-9 /()'&\-]{2,80}\?)",
        r"([A-Za-z][A-Za-z0-9 /()'&\-]{2,80}):",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text or ""):
            c = _canonical_label(m.group(1))
            if not _is_probable_field(c):
                continue
            low = _label_signature(c)
            if low in seen:
                continue
            seen.add(low)
            found.append(c)
    preserve_exact = {
        "name",
        "claim number",
        "policy number",
        "date",
        "yes / no",
        "salary",
    }

    # 3) Remove noisy subset fragments if a longer canonical label exists.
    canonical_lower = {lbl.lower() for lbl, _ in _ORDERED_FIELD_PATTERNS}
    cleaned: list[str] = []
    for item in found:
        low = item.lower()
        if low in preserve_exact:
            cleaned.append(item)
            continue
        # Skip short fragment when it is part of a known canonical question.
        if any(low != k and low in k and len(low) < 24 for k in canonical_lower):
            continue
        cleaned.append(item)

    # 4) Remove residual fragments if a longer detected field already contains them.
    # Keep core header fields even if they are short/common substrings.
    final_fields: list[str] = []
    lowers = [x.lower() for x in cleaned]
    for i, item in enumerate(cleaned):
        low = lowers[i]
        if low in preserve_exact:
            final_fields.append(item)
            continue
        is_fragment = False
        if len(low) <= 28 and low.split() and not re.match(r"^(do|are|have|list|describe|special|dates|branch|highest)\b", low):
            for j, other in enumerate(lowers):
                if i == j:
                    continue
                if len(other) > len(low) + 8 and low in other:
                    is_fragment = True
                    break
        if not is_fragment:
            final_fields.append(item)
    return final_fields


def _build_schema(field_names: list[str]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    used_keys: set[str] = set()
    seen_labels: set[str] = set()
    for label in field_names:
        sig = _label_signature(label)
        if sig in seen_labels:
            continue
        seen_labels.add(sig)
        key, desc = _schema_for_label(label, used_keys)
        schema[key] = {
            "type": "string",
            "method": "extract",
            "description": desc,
        }
    return schema


@router.get("/api/fields/extract")
async def extract_fields(file: str = Query(...)):
    """
    Extract likely user-fillable form field names from a PDF or Markdown file.
    Returns:
      - field_names list (text preview)
      - schema object (JSON format suitable for download/use)
    """
    if ".." in file or "/" in file or "\\" in file:
        raise HTTPException(400, "Invalid filename")

    path = _find_file(file)
    if not path:
        raise HTTPException(404, f"File not found: {file}")

    ext = os.path.splitext(file)[1].lower()
    text = ""
    mode = "native"
    try:
        if ext == ".md":
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                text = f.read()
            # Accuracy boost: if companion PDF exists, merge Mistral OCR text.
            pdf_path = os.path.splitext(path)[0] + ".pdf"
            if os.path.isfile(pdf_path):
                try:
                    from document_loader import mistral_ocr_pdf_to_markdown
                    text = text + "\n\n" + mistral_ocr_pdf_to_markdown(pdf_path)
                    mode = "native+pdf_mistral"
                except Exception:
                    mode = "native"
        elif ext == ".pdf":
            from document_loader import (
                extract_text_from_pdf,
                mistral_ocr_pdf_to_markdown,
            )

            try:
                text = mistral_ocr_pdf_to_markdown(path)
                # Merge native extraction too; each catches different labels.
                pages = extract_text_from_pdf(path)
                text += "\n\n" + "\n\n".join(t for _, t in pages)
                mode = "mistral_ocr"
            except Exception:
                pages = extract_text_from_pdf(path)
                text = "\n\n".join(t for _, t in pages)
                mode = "pdf_extract"
        else:
            raise HTTPException(400, "Only .pdf and .md are supported")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, str(e))

    field_names = _extract_field_names(text)
    schema = _build_schema(field_names)
    preview = "\n".join(f"- {x}" for x in field_names)
    return {
        "file_name": file,
        "mode": mode,
        "field_names": field_names,
        "text_preview": preview,
        "schema": schema,
    }

