import streamlit as st
from pathlib import Path


st.set_page_config(page_title="EDA", page_icon="📊", layout="wide")
st.title("Exploratory Data Analysis (EDA)")
st.caption("Each plot is loaded from local files. One plot per section, stacked vertically.")

ASSETS = Path("assets")

plot_order = [
	("age_distribution_by_income.png", "Age Distribution by Income"),
	("gender_income_bar.png", "Income by Gender"),
	("hours_per_week_by_income.png", "Hours per Week by Income"),
	("income_by_occupation.png", "Income by Occupation"),
	("income_sex_workclass_heatmap.png", "Income vs Sex vs Workclass (Heatmap)"),
	("marital_status_income.png", "Income by Marital Status"),
	("native_country_pie.png", "Native Country Distribution"),
	("univariate_numerical_features.png", "Univariate Distributions of Numerical Features"),
]

for filename, title in plot_order:
	path = ASSETS / filename
	st.markdown(f"### {title}")
	if path.exists():
		st.image(str(path), use_container_width=True)
	else:
		st.warning(f"Missing plot: {filename}. Export this figure from the EDA notebook to assets/.")
	st.markdown("---")


