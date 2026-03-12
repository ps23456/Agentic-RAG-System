#!/usr/bin/env python3
"""
OCR Markdown → LLM-friendly formatted text.

PURPOSE: Take ANY format of .md file (from Mistral OCR, Docling, or other OCR)
and make the format better so an LLM can give better answers: aligned tables,
clear sections, preserved document order. Use this script on the .md output
before sending content to an LLM.

STATUS:
- Format-agnostic: works on any .md (different forms, different layouts).
- Physical Capacities Evaluation: special layout (two-column stand/walk/sit/drive,
  frequency tables with percentages under headers).
- All other formats: generic aligned pipe tables (HANDLING, ACTIVITY RESTRICTIONS,
  REMARKS, PHYSICIAN INFO, etc.) so structure is clear and columns line up.
- Document order preserved: header text → tables → trailing text.
- No Docling in this script: it only reformats existing markdown. Use
  convert_with_docling.py for PDF/image → markdown extraction.

Usage:
  python scripts/convert_md_to_txt_with_docling.py <any.md> [-o output.txt]
  python scripts/convert_md_to_txt_with_docling.py data/uploads/markdown.md
  python scripts/convert_md_to_txt_with_docling.py "data/uploads/markdown 1.md" -o "data/uploads/markdown 1.txt"
"""

import os
import re
import sys
from pathlib import Path


def _split_cells(line):
    """Split a table line by | into cells (strip outer pipes and each cell)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    return [c.strip() for c in s.split("|")]


def _parse_tables(content):
    """
    Parse markdown into segments in document order: ("text", lines) or ("table", rows).
    Preserves order so trailing text (e.g. after tables) is emitted after tables.
    """
    lines = content.split("\n")
    segments = []
    current_text = []
    in_table = False
    current_table = []

    i = 0
    while i < len(lines):
        line = lines[i]
        has_pipe = "|" in line
        if has_pipe and line.strip():
            if not in_table:
                if current_text:
                    segments.append(("text", current_text))
                    current_text = []
                in_table = True
            cells = _split_cells(line)
            continuation = []
            i += 1
            while i < len(lines) and "|" not in lines[i]:
                if lines[i].strip() == "":
                    break  # blank line ends table cell continuation
                continuation.append(lines[i])
                i += 1
            if continuation:
                first_cell = cells[0] if cells else ""
                first_cell += "\n" + "\n".join(continuation)
                cells = [first_cell] + (cells[1:] if len(cells) > 1 else [])
            current_table.append(cells)
            continue
        else:
            if in_table and (line.strip() == "" or not line.strip()):
                if current_table:
                    segments.append(("table", current_table))
                    current_table = []
                in_table = False
                # do not add blank line to current_text
            if in_table:
                i += 1
                continue
            current_text.append(line)
            i += 1
    if current_table:
        segments.append(("table", current_table))
    if current_text:
        segments.append(("text", current_text))
    return segments


def _first_line(cell):
    return (cell or "").split("\n")[0].strip() if cell != "__BLANK__" else ""


def _is_separator_row(row):
    if row == "__BLANK__" or not row:
        return False
    first = _first_line(row[0] if row else "")
    return re.match(r"^[-—]\s*$", first) or (first.replace("-", "").replace(" ", "") == "" and len(row) > 1)


def _is_work_day_row(row):
    if row == "__BLANK__" or not row:
        return False
    first = (row[0] or "").strip()
    return "In a work day, patient can" in first and "\n" in (row[0] or "")


def _is_frequency_header_start(row):
    """First row of a frequency table: 'Patient can lift:' or 'carry:' or 'Patient can:' with Never, Occasionally."""
    if row == "__BLANK__" or len(row) < 2:
        return False
    first = (_first_line(row[0]) or "").strip()
    second = (_first_line(row[1]) or "").strip() if len(row) > 1 else ""
    third = (_first_line(row[2]) or "").strip() if len(row) > 2 else ""
    if "Patient can lift" in first or "Patient can carry" in first or first.strip() == "Patient can:":
        return second.strip() == "Never" and third.strip() == "Occasionally"
    return False


def _is_frequency_percentage_row(row):
    """Row that is (Up to 33%) | Frequently or (34%-66%) | Continuously or (67%-100%) |."""
    if row == "__BLANK__" or len(row) < 1:
        return False
    first = (_first_line(row[0]) or "").strip()
    return first in ("(Up to 33%)", "(34%-66%)", "(67%-100%)")


def _format_work_day_block(rows, col1_width=52):
    """Format stand/walk/sit/drive as two clear side-by-side columns. Uses 3 rows per activity."""
    out = []
    i = 0
    while i < len(rows):
        row = rows[i]
        if row == "__BLANK__":
            i += 1
            continue
        if not _is_work_day_row(row):
            break  # stop so caller can handle frequency table; return consumed count so far
        if i + 2 >= len(rows):
            i += 1
            continue
        row2, row3 = rows[i + 1], rows[i + 2]
        # Row 1: first cell = "In a work day, patient can X:" + "(Hours at one time)" (no checkbox line in cell)
        lines_in_cell = (row[0] or "").split("\n")
        activity = lines_in_cell[0].strip() if lines_in_cell else ""
        hours_at_one = lines_in_cell[1].strip() if len(lines_in_cell) >= 2 else ""
        # Row 2: first cell = checkbox row 1, second cell = "(TOTAL hours during day)"
        checkbox_row1 = _first_line(row2[0]).strip() if row2 and len(row2) > 0 else ""
        total_label = _first_line(row2[1]).strip() if row2 and len(row2) > 1 else ""
        # Row 3: first cell = checkbox row 2 (TOTAL hours during day)
        checkbox_row2 = _first_line(row3[0]).strip() if row3 and len(row3) > 0 else ""

        # Two columns: left = activity, (Hours at one time), checkbox row 1; right = (TOTAL...), blank, checkbox row 2
        out.append(activity.ljust(col1_width) + "  " + total_label)
        out.append(hours_at_one.ljust(col1_width) + "  " if hours_at_one else (" " * col1_width + "  "))
        out.append(checkbox_row1.ljust(col1_width) + "  " + checkbox_row2)
        out.append("")
        i += 3  # consumed 3 rows per activity
    return out, i


def _format_physical_capacities_table(table):
    """Format the Physical Capacities Evaluation table: work-day two columns + frequency tables with percentages."""
    out = []
    sep = " | "
    col1_width = 52
    i = 0
    if table and ("PHYSICAL CAPACITIES EVALUATION" in _first_line(table[0][0]) if table[0] else False):
        out.append("PHYSICAL CAPACITIES EVALUATION")
        out.append("")
        i = 1
    if i < len(table) and _is_separator_row(table[i]):
        i += 1
    while i < len(table):
        row = table[i]
        if _is_work_day_row(row):
            block_out, consumed = _format_work_day_block(table[i:], col1_width)
            out.extend(block_out)
            i += consumed
            continue
        if _is_frequency_header_start(row):
            break
        i += 1
    while i < len(table):
        row = table[i]
        if _is_frequency_header_start(row) and i + 4 <= len(table) and _is_frequency_percentage_row(table[i + 1]) and _is_frequency_percentage_row(table[i + 2]) and _is_frequency_percentage_row(table[i + 3]):
            r0, r1, r2, r3 = row, table[i + 1], table[i + 2], table[i + 3]
            label = _first_line(r0[0])
            never = _first_line(r0[1]) if len(r0) > 1 else ""
            occ = _first_line(r0[2]) if len(r0) > 2 else ""
            freq = _first_line(r1[1]) if len(r1) > 1 else "Frequently"
            cont = _first_line(r2[1]) if len(r2) > 1 else "Continuously"
            pct1, pct2, pct3 = _first_line(r1[0]), _first_line(r2[0]), _first_line(r3[0])
            w_label = max(14, len(label), len("BEND/TWIST AT WAIST"))
            w_never, w_occ = max(6, len(never)), max(14, len(occ), len(pct1))
            w_freq, w_cont = max(10, len(freq), len(pct2)), max(12, len(cont), len(pct3))
            out.append(sep.join([label.ljust(w_label), never.ljust(w_never), occ.ljust(w_occ), freq.ljust(w_freq), cont.ljust(w_cont)]))
            out.append(sep.join(["".ljust(w_label), "".ljust(w_never), pct1.ljust(w_occ), pct2.ljust(w_freq), pct3.ljust(w_cont)]))
            i += 4
            continue
        w_label, w_never, w_occ, w_freq, w_cont = 14, 6, 14, 12, 12
        for j in range(i, len(table)):
            r = table[j]
            if _is_frequency_header_start(r) or _is_frequency_percentage_row(r):
                break
            if r and len(r) >= 5:
                w_label = max(w_label, len(_first_line(r[0])))
                w_never = max(w_never, len(_first_line(r[1])))
                w_occ = max(w_occ, len(_first_line(r[2])))
                w_freq = max(w_freq, len(_first_line(r[3])))
                w_cont = max(w_cont, len(_first_line(r[4])))
        while i < len(table):
            r = table[i]
            if _is_frequency_header_start(r):
                break
            if _is_frequency_percentage_row(r):
                i += 1
                continue
            cells = [_first_line(r[c]) if c < len(r) else "" for c in range(5)]
            out.append(sep.join([cells[0].ljust(w_label), cells[1].ljust(w_never), cells[2].ljust(w_occ), cells[3].ljust(w_freq), cells[4].ljust(w_cont)]))
            i += 1
        out.append("")
    return out


def _format_text_segment(lines):
    """
    Format plain-text segment for LLM readability: align Label: value pairs,
    emphasize # section headers, collapse excess blanks. Works on .md with no pipe tables.
    """
    if not lines:
        return []
    out = []
    # Collapse consecutive blank lines to one, trim trailing blanks
    cleaned = []
    prev_blank = False
    for line in lines:
        is_blank = not line.strip()
        if is_blank:
            if not prev_blank:
                cleaned.append("")
            prev_blank = True
        else:
            cleaned.append(line)
            prev_blank = False
    lines = cleaned
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        # Section header: # HEADING -> prominent line + blank
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            if heading:
                out.append(heading)
                out.append("")
            i += 1
            continue
        # Collect a block of non-blank lines (until blank or next #)
        block = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("#"):
            block.append(lines[i].strip())
            i += 1
        if not block:
            if i < len(lines) and not lines[i].strip():
                out.append("")
            i += 1
            continue
        # Check if block is mostly "Label: value" lines (form fields)
        label_value_lines = []
        for ln in block:
            if ": " in ln and not ln.strip().startswith("http"):
                idx = ln.find(": ")
                label_value_lines.append((ln[:idx].strip(), ln[idx + 2:].strip()))
            else:
                label_value_lines.append(None)  # not a label-value line
        if all(x is not None for x in label_value_lines) and len(label_value_lines) >= 1:
            w = max(len(l) for l, v in label_value_lines)
            for (label, value) in label_value_lines:
                out.append((label + ":").ljust(w + 2) + " " + value)
        elif any(x is not None for x in label_value_lines):
            # Mixed block: align only the label-value lines, leave others as-is
            w = 0
            for item in label_value_lines:
                if item is not None:
                    w = max(w, len(item[0]))
            for j, ln in enumerate(block):
                if label_value_lines[j] is not None:
                    label, value = label_value_lines[j]
                    out.append((label + ":").ljust(w + 2) + " " + value)
                else:
                    out.append(ln)
        else:
            for ln in block:
                out.append(ln)
        out.append("")
        if i < len(lines) and not lines[i].strip():
            i += 1
    return out


def _format_segments(segments):
    """Produce formatted text from segments (document order). Handles any format: text (align label: value, headers) or tables."""
    out = []
    sep = " | "
    for kind, payload in segments:
        if kind == "text":
            formatted = _format_text_segment(payload)
            out.extend(formatted)
            if formatted and formatted[-1].strip():
                out.append("")
        elif kind == "table":
            rows = payload
            if not rows:
                continue
            if _is_physical_capacities_table(rows):
                out.extend(_format_physical_capacities_table(rows))
            else:
                out.extend(_format_generic_table(rows))
            out.append("")
    return "\n".join(out).rstrip()


def _cell_display_width(cell):
    if not cell or cell == "__BLANK__":
        return 2
    return min(max(len(x) for x in (cell or "").split("\n")), 250)


def _format_generic_table(rows, sep=" | "):
    """Format any pipe table: aligned columns, multi-line cells with continuation indented."""
    if not rows:
        return []
    ncols = max(len(r) for r in rows)
    widths = [2] * ncols
    for r in rows:
        for c in range(min(len(r), ncols)):
            w = _cell_display_width(r[c])
            widths[c] = max(widths[c], w)
    out = []
    for row in rows:
        parts = []
        first_cell_continuations = []
        for c in range(ncols):
            cell = row[c] if c < len(row) else ""
            first = _first_line(cell)
            if len(first) > widths[c]:
                first = first[: widths[c] - 1] + "…"
                if c == 0:
                    rest = (row[0] or "").split("\n")[0]
                    if len(rest) > widths[0]:
                        first_cell_continuations.append("   " + rest[widths[0] - 1:].strip())
            parts.append(first.ljust(widths[c]))
        out.append(sep.join(parts))
        # Continuation lines for first cell (multi-line or truncated long line)
        if row and "\n" in (row[0] or ""):
            for extra in (row[0] or "").split("\n")[1:]:
                if extra.strip():
                    out.append("   " + extra.strip())
        for line in first_cell_continuations:
            out.append(line)
    return out


def _is_physical_capacities_table(rows):
    """True if this table is the Physical Capacities Evaluation format (stand/walk/sit/drive + lift/carry)."""
    if not rows or len(rows) < 2:
        return False
    first = _first_line(rows[0][0] if rows[0] else "")
    if "PHYSICAL CAPACITIES EVALUATION" not in first:
        return False
    for r in rows[1:]:
        if r == "__BLANK__":
            continue
        fl = _first_line(r[0] if r else "")
        if "In a work day, patient can" in fl or _is_frequency_header_start(r):
            return True
    return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/convert_md_to_txt_with_docling.py <path-to-.md> [-o output.txt]", file=sys.stderr)
        sys.exit(1)

    md_path = os.path.abspath(sys.argv[1])
    out_path = None
    if "-o" in sys.argv:
        try:
            i = sys.argv.index("-o")
            out_path = os.path.abspath(sys.argv[i + 1])
        except (IndexError, ValueError):
            pass

    if not os.path.isfile(md_path):
        print(f"Error: file not found: {md_path}", file=sys.stderr)
        sys.exit(2)

    if out_path is None:
        out_path = str(Path(md_path).with_suffix(".txt"))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(md_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    segments = _parse_tables(content)
    formatted = _format_segments(segments)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(formatted)

    print(f"Input:  {md_path}")
    print(f"Output: {out_path} (LLM-friendly formatted)")


if __name__ == "__main__":
    main()
