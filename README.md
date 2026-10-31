# Placement Prediction — ML Minor Project

Student academic profile (10th %, 12th %, degree %, test scores, work experience, etc.)
tీసుకుని, aa student **Placed** avthada leda **Not Placed** ah predict cheseే web app.

## Folder Structure

```
placement_prediction/
├── dataset/
│   ├── Placement_Data_Full_Class.csv   -> training data (300 students)
│   └── generate_dataset.py             -> script that created the CSV (optional, already run)
├── notebook/
│   └── train_model.py                  -> EDA + preprocessing + training + saving the model
├── model/
│   ├── model.pkl                       -> trained classifier (saved after training)
│   ├── encoders.pkl                    -> label encoders for categorical columns
│   └── feature_order.pkl               -> exact column order the model expects
├── templates/
│   ├── index.html                      -> input form (home page)
│   └── result.html                     -> prediction result page
├── static/
│   └── style.css                       -> styling
├── app.py                              -> Flask backend (serves the website)
└── requirements.txt
```

## How to run on your own laptop

1. **Install Python 3.9+** (if not already installed).
2. Open terminal in this folder and install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. (Optional — already done for you) Train the model:
   ```
   python notebook/train_model.py
   ```
   This creates `model/model.pkl`, `model/encoders.pkl`, `model/feature_order.pkl`.
4. Run the website:
   ```
   python app.py
   ```
5. Open your browser and go to: **http://127.0.0.1:5000**
6. Fill the form → click "Predict Placement" → result page shows Placed / Not Placed with confidence %.

## Using the REAL Kaggle dataset instead (recommended for submission)

The CSV included here is a synthetic dataset built to match the exact column
structure of the well-known Kaggle dataset "Campus Recruitment" (search on
Kaggle: **factors-affecting-campus-placement** by benroshan). For your final
submission it's better to:

1. Download that CSV from Kaggle.
2. Replace `dataset/Placement_Data_Full_Class.csv` with it (same filename).
3. Re-run `python notebook/train_model.py` to retrain on the real data.
4. Run `python app.py` again — no other code changes needed, since column names match exactly.

## What to put in your project report / PPT

- **Problem statement**: Predict whether a student will be placed based on academic and test-score features (binary classification).
- **Dataset**: Campus Recruitment dataset — 15 columns, ~215–300 rows, mix of numeric (percentages) and categorical (board, stream, specialisation) features.
- **Preprocessing**: Dropped `sl_no` (index) and `salary` (leaks the answer — only known after placement). Label-encoded categorical columns.
- **Models tried**: Logistic Regression, Random Forest — compared using accuracy, precision, recall, F1-score (`notebook/train_model.py` prints all of this).
- **Best model**: chosen automatically by test accuracy and saved as `model.pkl`.
- **Deployment**: Flask backend serves an HTML form; user inputs are encoded the same way as training data and passed to the model for a live prediction.
- Add screenshots of the home page and result page from your browser.

## Notes for your viva / demo

- If asked "why not use salary as a feature" — explain it's data leakage (salary is only assigned *after* placement happens, so a real-world new applicant would never have this value).
- If asked "how is a new user's text/category converted to numbers" — explain the saved `encoders.pkl` — the same `LabelEncoder` used in training is reused in the app, so the mapping ("Yes"→1, "No"→0, etc.) stays consistent.
