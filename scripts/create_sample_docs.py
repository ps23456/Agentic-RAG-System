"""
Create sample policy, claim, and medical PDFs in data/ for testing the search app.
Run: python scripts/create_sample_docs.py
"""
import os
from pathlib import Path

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    print("Install reportlab: pip install reportlab")
    raise

# Project root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def write_pdf(path: Path, title: str, body: str) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, 720, title)
    c.setFont("Helvetica", 10)
    y = 690
    for line in body.split("\n"):
        if y < 72:
            c.showPage()
            c.setFont("Helvetica", 10)
            y = 720
        c.drawString(72, y, line[:90])
        y -= 14
    c.save()


def main():
    # Policy PDF
    policy_text = """Policy Terms and Conditions - Health Insurance Plan

PRE-EXISTING CONDITIONS CLAUSE (Section 4.2)
Claims related to pre-existing conditions are excluded for the first 12 months of coverage.
Pre-existing condition means any condition for which the insured received treatment or diagnosis in the 6 months prior to the policy start date.
After the waiting period, coverage for such conditions may be approved subject to underwriting.

CLAIM REJECTION GROUNDS (Section 7)
A claim may be rejected if: (a) the claim form is incomplete; (b) supporting medical reports are missing; (c) the treatment is not medically necessary; (d) the claimant has exceeded benefit limits; (e) the claim relates to an excluded condition or pre-existing condition within the waiting period.

Claim ID format: CLM-XXXX where XXXX is a numeric identifier. All claims must reference this ID in correspondence."""
    write_pdf(DATA / "policy_terms.pdf", "Policy Terms and Conditions", policy_text)
    print("Created policy_terms.pdf")

    # Claim PDF - rejected claim
    claim_text = """Claim Summary - CLM-8891

Status: REJECTED
Date of claim: 2024-01-15
Insured: John Doe
Policy: POL-12345

Reason for rejection:
Claim CLM-8891 was rejected because the treatment (spinal surgery) was deemed to relate to a pre-existing condition documented in the insured's medical history prior to policy inception. Per policy clause Section 4.2, pre-existing conditions are excluded for the first 12 months. The claimant may appeal with additional medical documentation within 90 days."""
    write_pdf(DATA / "claim_CLM-8891.pdf", "Claim CLM-8891 - Rejected", claim_text)
    print("Created claim_CLM-8891.pdf")

    # Medical report
    medical_text = """Medical Report - Attending Physician

Patient: John Doe
Date of examination: 2023-11-20

History: Patient has chronic lower back pain with prior diagnosis of lumbar disc herniation. Patient received physical therapy and pain management in the 6 months prior to the reported policy start date.

Recommendation: Surgical evaluation for spinal decompression. The condition has been present and treated before the current insurance coverage began. This may be considered a pre-existing condition under typical policy definitions."""
    write_pdf(DATA / "medical_report_John_Doe.pdf", "Medical Report", medical_text)
    print("Created medical_report_John_Doe.pdf")

    print(f"Sample documents written to {DATA}")


if __name__ == "__main__":
    main()
