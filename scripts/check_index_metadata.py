#!/usr/bin/env python3
"""
Diagnostic script: verify index metadata extraction.
Run from project root: python scripts/check_index_metadata.py

Shows:
- How many chunks have patient_name, claim_number, etc.
- Sample of known patients/claims
- Whether "list the patients" would work
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from document_loader import load_and_chunk_folder
from retrieval.query_metadata_extractor import get_index_metadata_catalog


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_folder = os.environ.get("CLAIM_SEARCH_DATA", os.path.join(project_root, "data"))
    if not os.path.isdir(data_folder):
        print(f"Data folder not found: {data_folder}")
        return 1

    print("Loading and chunking documents...")
    chunks = load_and_chunk_folder(data_folder)
    if not chunks:
        print("No chunks extracted. Add PDF/md/txt files to the data folder.")
        return 1

    catalog = get_index_metadata_catalog(chunks)
    patients = catalog.get("known_patients", [])
    claims = catalog.get("known_claims", [])
    policies = catalog.get("known_policies", [])

    # Count chunks with metadata
    with_patient = sum(1 for c in chunks if getattr(c, "patient_name", ""))
    with_claim = sum(1 for c in chunks if getattr(c, "claim_number", ""))
    total = len(chunks)

    print(f"\n=== Index metadata summary ===")
    print(f"Total chunks: {total}")
    print(f"Chunks with patient_name: {with_patient} ({100*with_patient/total:.1f}%)")
    print(f"Chunks with claim_number: {with_claim} ({100*with_claim/total:.1f}%)")
    print(f"\nKnown patients: {patients or '(none)'}")
    print(f"Known claims: {claims[:10]}{'...' if len(claims) > 10 else ''}")
    print(f"Known policies: {policies[:5]}{'...' if len(policies) > 5 else ''}")

    if not patients:
        print("\n⚠️  No patient names extracted. 'List the patients' will NOT use the special path.")
        print("   Ensure documents contain patterns like 'Patient Name: X' or 'Patient Name X'")
        print("   Then re-index in the app.")
    else:
        print(f"\n✓ 'List the patients' will show: {', '.join(patients)}")

    # Sample chunks with patient_name
    if patients:
        print("\nSample chunks per patient:")
        for p in patients[:5]:
            sample = next((c for c in chunks if getattr(c, "patient_name", "") == p), None)
            if sample:
                preview = (sample.text or "")[:120].replace("\n", " ")
                print(f"  {p}: {sample.file_name} p.{sample.page_number} — {preview}...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
