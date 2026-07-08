from reporting import save_evaluation_reports
from training import MODEL_PATH, evaluate_saved_model, save_model_artifacts, train_model


def retrain_and_save() -> None:
    positive_model, negative_model, vectorizer = train_model()
    save_model_artifacts(positive_model, negative_model, vectorizer)

    report_payload = evaluate_saved_model()
    pdf_path, json_path = save_evaluation_reports(report_payload)

    print(f"Modèle réentraîné et sauvegardé dans {MODEL_PATH}")
    print(f"Rapport PDF enregistré dans {pdf_path}")
    print(f"Rapport JSON enregistré dans {json_path}")


if __name__ == "__main__":
    retrain_and_save()
