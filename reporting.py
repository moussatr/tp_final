from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

REPORT_DIR = os.getenv("REPORT_DIR", str(Path(__file__).resolve().parent / "reports"))
REPORT_PDF_PATH = os.getenv(
    "REPORT_PATH",
    str(Path(REPORT_DIR) / "sentiment_evaluation_report.pdf"),
)
REPORT_JSON_PATH = os.getenv(
    "REPORT_JSON_PATH",
    str(Path(REPORT_DIR) / "sentiment_evaluation_report.json"),
)


def _styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="BodyTight",
            parent=styles["BodyText"],
            leading=14,
            spaceAfter=6,
        )
    )
    return styles


def _metric(report: dict[str, Any], label: str, metric: str) -> float:
    return float(report.get(label, {}).get(metric, 0))


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f} %"


def _matrix_table(title: str, matrix: list[list[int]], styles: dict[str, ParagraphStyle]) -> list[Any]:
    table = Table(
        [
            ["", "Prédit 0", "Prédit 1"],
            ["Réel 0", matrix[0][0], matrix[0][1]],
            ["Réel 1", matrix[1][0], matrix[1][1]],
        ],
        colWidths=[4 * cm, 3 * cm, 3 * cm],
    )
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8EEF7")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F3F5F8")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9AA7B5")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 7),
            ]
        )
    )
    return [Paragraph(title, styles["Heading3"]), table, Spacer(1, 0.35 * cm)]


def _metrics_table(report_payload: dict[str, Any]) -> Table:
    rows = [["Modèle", "Classe", "Précision", "Rappel", "F1-score", "Support"]]
    for model_name, report in report_payload["classification_report"].items():
        for label in ("0", "1"):
            rows.append(
                [
                    model_name,
                    label,
                    _format_percent(_metric(report, label, "precision")),
                    _format_percent(_metric(report, label, "recall")),
                    _format_percent(_metric(report, label, "f1-score")),
                    int(report.get(label, {}).get("support", 0)),
                ]
            )

    table = Table(rows, repeatRows=1, colWidths=[2.4 * cm, 1.6 * cm, 2.5 * cm, 2.3 * cm, 2.3 * cm, 2 * cm])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#243B53")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#B8C2CC")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F7F9FB")]),
                ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _analysis_lines(report_payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for model_name, report in report_payload["classification_report"].items():
        class_one_f1 = _metric(report, "1", "f1-score")
        class_one_recall = _metric(report, "1", "recall")
        class_one_precision = _metric(report, "1", "precision")
        class_zero_f1 = _metric(report, "0", "f1-score")
        lines.append(
            f"{model_name}: F1 classe 1 = {_format_percent(class_one_f1)}, "
            f"précision = {_format_percent(class_one_precision)}, rappel = {_format_percent(class_one_recall)}."
        )
        if class_one_recall < 0.70:
            lines.append(
                f"{model_name}: le rappel de la classe 1 est faible; le modèle risque de manquer des tweets de cette catégorie."
            )
        if class_one_precision < 0.70:
            lines.append(
                f"{model_name}: la précision de la classe 1 est faible; certains tweets peuvent être classés à tort dans cette catégorie."
            )
        if abs(class_one_f1 - class_zero_f1) > 0.20:
            lines.append(
                f"{model_name}: l'écart de F1 entre les classes 0 et 1 suggère un possible biais de distribution."
            )

    return lines or ["Les métriques ne sont pas disponibles pour produire une analyse détaillée."]


def generate_pdf_report(output_path: str, report_payload: dict[str, Any]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    story: list[Any] = [
        Paragraph("Rapport d'évaluation du modèle de sentiment", styles["Title"]),
        Paragraph(f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M:%S')}", styles["BodyTight"]),
        Spacer(1, 0.3 * cm),
    ]
    story.extend(_matrix_table("Matrice de confusion - prédictions positives", report_payload["positive_confusion_matrix"], styles))
    story.extend(_matrix_table("Matrice de confusion - prédictions négatives", report_payload["negative_confusion_matrix"], styles))
    story.extend(
        [
            Paragraph("Précision, rappel et F1-score", styles["Heading2"]),
            _metrics_table(report_payload),
            Spacer(1, 0.4 * cm),
            Paragraph("Analyse des performances", styles["Heading2"]),
        ]
    )
    story.extend(Paragraph(line, styles["BodyTight"]) for line in _analysis_lines(report_payload))
    story.extend(
        [
            Paragraph("Biais et erreurs fréquentes", styles["Heading2"]),
            Paragraph(
                "Le modèle dépend fortement des mots présents dans les tweets annotés. Avec peu de données, "
                "il peut confondre les phrases courtes, ironiques ou contenant un vocabulaire absent du dataset.",
                styles["BodyTight"],
            ),
            Paragraph("Recommandations", styles["Heading2"]),
            Paragraph(
                "Ajouter davantage de tweets annotés, équilibrer les classes positive et négative, intégrer des n-grammes, "
                "puis comparer les métriques à chaque réentraînement hebdomadaire.",
                styles["BodyTight"],
            ),
        ]
    )
    doc.build(story)


def save_json_report(output_path: str, report_payload: dict[str, Any]) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_evaluation_reports(report_payload: dict[str, Any]) -> tuple[str, str]:
    generate_pdf_report(REPORT_PDF_PATH, report_payload)
    save_json_report(REPORT_JSON_PATH, report_payload)
    return REPORT_PDF_PATH, REPORT_JSON_PATH
