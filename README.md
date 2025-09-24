streamlit run web/app.py

uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

import joblib

model = joblib.load("models/random_forest_model.pkl")
print(model.feature_names_in_)  # Nếu mô hình hỗ trợ thuộc tính này