# Docling vs Mistral OCR – When to Use Which

## The problem you saw

- **Mistral OCR** produced a good `markdown.md` with the full form (stand/walk/sit/drive, all checkboxes ☐/☑, frequency labels).
- **Docling on that .md file** (MD → parse → re-export) **dropped content**: the stand/walk/sit/drive block and multi-line table cells were lost or turned into empty rows. Docling’s Markdown parser is not built for round-trip fidelity on complex tables.

## Correct workflows

### 1. Mistral OCR → text in table format (like the original form)

**Do not** run that markdown through Docling (it drops table content). Use a script that preserves the full table layout and checkboxes:

```bash
# Output: .txt with tables and checkboxes (☐/☑) like the original form
python scripts/mistral_md_to_table_txt.py data/uploads/markdown.md -o data/uploads/markdown.txt
```

Same result (exact copy) with:
```bash
python scripts/md_to_txt_fidelity.py data/uploads/markdown.md -o data/uploads/markdown.txt
```

### 2. You want “Docling AI” to give better extraction than Mistral

Run Docling on the **original document** (PDF or image of the form), not on the Mistral-generated markdown:

```bash
# From project root (ensure numpy<2: pip install 'numpy>=1.26.0,<2')
python scripts/convert_with_docling.py path/to/original_form.pdf
# or
python scripts/convert_with_docling.py path/to/original_form.tif
# or
./scripts/run_convert_with_docling.sh path/to/original_form.tif
```

Output goes to `data/structured/<stem>.md` (or .html/.json with `--format`). That way Docling’s OCR and layout/table logic run on the real form and you can compare that output to Mistral’s.

### 3. Summary

| Goal | Use |
|------|-----|
| Mistral OCR → .txt in **table format like original form** | `scripts/mistral_md_to_table_txt.py` (or `md_to_txt_fidelity.py`) |
| Extract from original PDF/image with Docling (table output) | `scripts/convert_with_docling.py` on the original file |
| Avoid | Docling on an existing .md if you need to preserve tables/checkboxes |

Your `data/uploads/markdown.txt` has been regenerated with the fidelity script so it now matches `markdown.md` exactly (including stand/walk/sit/drive and all checkboxes).
