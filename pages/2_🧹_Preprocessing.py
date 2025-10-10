import streamlit as st
import pandas as pd
from pathlib import Path


st.set_page_config(page_title="Preprocessing", page_icon="🧹", layout="wide")
st.title("Preprocessing Overview")
st.caption("This page reflects the dataset and preprocessing performed in the ML notebook. No recomputation is done.")

DATA_DIR = Path("data")
PROCESSED = DATA_DIR / "processed.csv"

if PROCESSED.exists():
	# 1) Dataframe first (as requested)
	st.subheader("Processed Dataset (head)")
	df = pd.read_csv(PROCESSED)
	st.dataframe(df.head(50), use_container_width=True)

	# 2) Summary and schema
	st.subheader("Shape and Column Types")
	col1, col2 = st.columns(2)
	with col1:
		st.write({"rows": df.shape[0], "columns": df.shape[1]})
	with col2:
		st.write(df.dtypes.astype(str).to_dict())

	# 3) Preprocessing walkthrough (verbatim from ML notebook)
	st.subheader("Preprocessing Steps (from ML Notebook)")
	with st.expander("Show exact steps and code (read-only)", expanded=True):
		# a) Column removal
		st.markdown("**Columns removed:** `fnlwgt`, `education`, `marital.status`, `capital.gain`, `capital.loss`, `relationship`, `occupation`, `workclass`")
		st.code(
			"""
df.drop(inplace=True,axis=1,columns=['fnlwgt','education','marital.status','capital.gain','capital.loss','relationship','occupation','workclass'])
""",
			language="python",
		)

		# b) Feature engineering
		st.markdown("**Engineered features:** `net_capital`")
		st.code(
			"""
df['net_capital']=df['capital.gain']-df['capital.loss']
""",
			language="python",
		)

		# c) Train-test split with stratification, then drop helper column
		st.markdown("**Train/Test split (stratified) and cleanup:**")
		st.code(
			"""
train,test=train_test_split(df,random_state=42,test_size=0.2,stratify=df['strat_col'])
train.drop('strat_col', inplace=True, axis=1)
test.drop('strat_col', inplace=True, axis=1)
""",
			language="python",
		)

		# d) Scaling numeric columns
		st.markdown("**Scale numeric columns (StandardScaler):**")
		st.code(
			"""
numeric_cols.remove('income')

scaler = StandardScaler()
train[numeric_cols] = scaler.fit_transform(train[numeric_cols])
test[numeric_cols] = scaler.transform(test[numeric_cols])
""",
			language="python",
		)

		# e) Feature/target split
		st.markdown("**Feature/Target split:**")
		st.code(
			"""
y_train = train['income']
X_train = train.drop('income', axis=1)
y_test = test['income']
X_test = test.drop('income', axis=1)
""",
			language="python",
		)

	# 4) Highlight the engineered feature present per the original code
	st.subheader("Engineered Feature Presence")
	col_a, col_b = st.columns(2)
	with col_a:
		st.success({"present": [c for c in ['net_capital'] if c in df.columns]})
	with col_b:
		st.info({"missing": [c for c in ['net_capital'] if c not in df.columns]})

	# 5) Target distribution and missing values
	st.subheader("Target Distribution")
	if "income" in df.columns:
		st.bar_chart(df["income"].value_counts().sort_index())

	st.subheader("Missing Values per Column")
	st.write(df.isna().sum().sort_values(ascending=False))
else:
	st.error(f"Missing processed file: {PROCESSED}")


