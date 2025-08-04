import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os

# Dateipfade (anpassen, falls nötig)
REFERENCE_DATA_PATH = "data/reference.csv"
CURRENT_DATA_PATH = "data/current.csv"
REPORT_HTML_PATH = "reports/drift_report.html"
REPORT_JSON_PATH = "reports/drift_report.json"

# Lade Daten
reference = pd.read_csv(REFERENCE_DATA_PATH)
current = pd.read_csv(CURRENT_DATA_PATH)

# Entferne Zielspalte (falls vorhanden)
reference = reference.drop(columns=["diagnosis"], errors="ignore")
current = current.drop(columns=["diagnosis"], errors="ignore")

# Erstelle Evidently-Report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# Speichere Report
os.makedirs("reports", exist_ok=True)
report.save_html(REPORT_HTML_PATH)
report.save_json(REPORT_JSON_PATH)

print(f"✅ Drift-Report gespeichert unter: {REPORT_HTML_PATH}")
