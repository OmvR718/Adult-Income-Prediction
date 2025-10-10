import streamlit as st
import json
import pandas as pd
from pathlib import Path


st.set_page_config(page_title="Modeling", page_icon="🧠", layout="wide")
st.title("Modeling Results")
st.caption("Displays saved evaluation plots and metrics from local files. No retraining.")

ASSETS = Path("assets")
MODELS = Path("models")

st.subheader("Model Metrics and Plots")

# Load metrics.json if available
metrics_path = MODELS / "metrics.json"
shown_paths = set()

if metrics_path.exists():
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    models_dict = metrics.get("models", {})

    # Enforce specific display order and avoid duplicates/others
    desired_order = [
        "logistic_regression",
        "decision_tree",
        "logistic_regression_undersampled",
        "decision_tree_undersampled",
    ]
    ordered_keys = [k for k in desired_order if k in models_dict]

    # For each model in the desired order, display metrics then matching plots
    for model_key in ordered_keys:
        model_metrics = models_dict.get(model_key, {})
        model_title = model_key.replace("_", " ").title()
        st.markdown(f"## {model_title}")

        # Metrics in a nice formatted block (not raw dict)
        acc = model_metrics.get("accuracy")
        prec = model_metrics.get("precision")
        rec = model_metrics.get("recall")
        f1 = model_metrics.get("f1")
        rows = []
        if acc is not None:
            rows.append(("Accuracy", f"{acc:.4f}"))
        if prec is not None:
            rows.append(("Precision", f"{prec:.4f}"))
        if rec is not None:
            rows.append(("Recall", f"{rec:.4f}"))
        if f1 is not None:
            rows.append(("F1-Score", f"{f1:.4f}"))
        if rows:
            for label, value in rows:
                st.markdown(f"- **{label}**: {value}")
        else:
            st.info("No metrics saved for this model yet.")

        # Display plots that contain the model key in their filename (stacked vertically)
        # Collect plots that belong to this model key
        raw = [p for p in ASSETS.glob("*.png") if model_key in p.name]
        # For undersampled models, only show undersampled plots; for non-undersampled, exclude them
        if "undersampled" in model_key:
            matched = [p for p in raw if "undersampled" in p.name]
        else:
            matched = [p for p in raw if "undersampled" not in p.name]
        # Prioritize decision boundaries first, then confusion matrices, then others
        matched = sorted(
            matched,
            key=lambda p: (
                0 if 'decision_boundary' in p.stem else (1 if 'confusion' in p.stem else 2),
                p.name.lower()
            )
        )
        if matched:
            for p in matched:
                title = p.stem
                st.markdown(f"### {title}")
                st.image(str(p), use_container_width=True)
                shown_paths.add(p)
        else:
            st.info("No plots found for this model in assets/.")

        st.markdown("---")
else:
	st.info("No models/metrics.json found. Run the save-metrics cell in the notebook to generate it.")

# Remove the extra 'Other Plots' grid; only show model sections stacked vertically

st.subheader("Saved Model Artifacts")
artifacts = [
	"best_model.pkl",
	"scaler.pkl",
	"resampler.pkl",
	"label_encoders.pkl",
	"feature_names.pkl",
]

for art in artifacts:
	path = MODELS / art
	st.write(f"{art}: {'Available' if path.exists() else 'Missing'}")


