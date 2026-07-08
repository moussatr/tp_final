from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

REPORT_DIR = os.getenv("REPORT_DIR", str(Path(__file__).resolve().parent / "reports"))
REPORT_PDF_PATH = os.getenv(
    "REPORT_PATH",
    str(Path(REPORT_DIR) / "sentiment_evaluation_report.pdf"),
)
REPORT_JSON_PATH = os.getenv(
    "REPORT_JSON_PATH",
    str(Path(REPORT_DIR) / "sentiment_evaluation_report.json"),
)

LINES_PER_PAGE = 48
LINE_HEIGHT = 14


def _escape_pdf_text(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _paginate_lines(lines: list[str]) -> list[list[str]]:
    pages: list[list[str]] = []
    for index in range(0, len(lines), LINES_PER_PAGE):
        pages.append(lines[index : index + LINES_PER_PAGE])
    return pages or [[]]


def _build_page_stream(lines: list[str]) -> bytes:
    stream = "BT\n/F1 10 Tf\n50 760 Td\n"
    for index, line in enumerate(lines):
        if index > 0:
            stream += f"0 -{LINE_HEIGHT} Td\n"
        stream += f"({_escape_pdf_text(line)}) Tj\n"
    stream += "ET\n"
    return stream.encode("latin-1", "replace")


def generate_pdf_report(output_path: str, report_payload: dict[str, Any]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    classification_text = json.dumps(report_payload["classification_report"], ensure_ascii=False, indent=2)
    lines = [
        "Rapport d'évaluation du modèle de sentiment",
        "",
        f"Date : {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}",
        "",
        "Matrice de confusion - classe positive",
        str(report_payload["positive_confusion_matrix"]),
        "",
        "Matrice de confusion - classe negative",
        str(report_payload["negative_confusion_matrix"]),
        "",
        "Précision, rappel et F1-score",
        classification_text,
        "",
        "Observations :",
        "- Le modèle est sensible au vocabulaire positif/negatif présent dans les données annotées.",
        "- Les erreurs fréquentes proviennent des phrases ambiguës ou du sarcasme.",
        "- Les biais potentiels sont liés à une distribution inégale des exemples positifs et négatifs.",
        "",
        "Recommandations :",
        "- Ajouter des exemples annotés plus variés et plus équilibrés.",
        "- Introduire des caractéristiques lexicales plus riches et des n-grammes.",
        "- Réentraîner le modèle régulièrement avec des données récentes.",
    ]

    pages = _paginate_lines(lines)
    page_count = len(pages)

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
    ]

    kids_refs = " ".join(f"{3 + index} 0 R" for index in range(page_count))
    objects.append(f"<< /Type /Pages /Kids [{kids_refs}] /Count {page_count} >>".encode("latin-1"))

    for page_index, page_lines in enumerate(pages):
        content_obj_number = 3 + page_count + page_index
        objects.append(
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_obj_number} 0 R "
                f"/Resources << /Font << /F1 {3 + 2 * page_count} 0 R >> >> >>"
            ).encode("latin-1")
        )

    for page_lines in pages:
        stream_bytes = _build_page_stream(page_lines)
        objects.append(
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("latin-1")
            + stream_bytes
            + b"\nendstream"
        )

    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    pdf = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf.extend(f"{index} 0 obj\n".encode("latin-1"))
        pdf.extend(obj)
        pdf.extend(b"\nendobj\n")

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("latin-1"))
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("latin-1"))

    pdf.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode(
            "latin-1"
        )
    )
    Path(output_path).write_bytes(pdf)


def save_json_report(output_path: str, report_payload: dict[str, Any]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_evaluation_reports(report_payload: dict[str, Any]) -> tuple[str, str]:
    generate_pdf_report(REPORT_PDF_PATH, report_payload)
    save_json_report(REPORT_JSON_PATH, report_payload)
    return REPORT_PDF_PATH, REPORT_JSON_PATH
