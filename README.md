

1/ uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

2/ streamlit run web/app.py

import joblib

model = joblib.load("models/random_forest_model.pkl")
print(model.feature_names_in_)  # Nếu mô hình hỗ trợ thuộc tính này
