from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Folder where app.py is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Exact model folder
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Exact file paths
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
ENCODERS_PATH = os.path.join(MODEL_DIR, "encoders.pkl")
FEATURE_ORDER_PATH = os.path.join(MODEL_DIR, "feature_order.pkl")

# Debug: print paths
print("APP LOCATION:", BASE_DIR)
print("MODEL PATH:", MODEL_PATH)
print("MODEL EXISTS:", os.path.exists(MODEL_PATH))

# Load model
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Load encoders
with open(ENCODERS_PATH, "rb") as f:
    encoders = pickle.load(f)

# Load feature order
with open(FEATURE_ORDER_PATH, "rb") as f:
    feature_order = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    form = request.form

    # Build a dict of raw inputs from the form
    raw = {
        'gender': form.get('gender'),
        'ssc_p': float(form.get('ssc_p')),
        'ssc_b': form.get('ssc_b'),
        'hsc_p': float(form.get('hsc_p')),
        'hsc_b': form.get('hsc_b'),
        'hsc_s': form.get('hsc_s'),
        'degree_p': float(form.get('degree_p')),
        'degree_t': form.get('degree_t'),
        'workex': form.get('workex'),
        'etest_p': float(form.get('etest_p')),
        'specialisation': form.get('specialisation'),
        'mba_p': float(form.get('mba_p')),
    }
    
@app.route('/analytics')
def analytics():

    # Load original dataset
    df = pd.read_csv('dataset/placement.csv')

    # Create graphs folder if it doesn't exist
    os.makedirs('static/graphs', exist_ok=True)


    # ==================================================
    # 1. PLACEMENT STATUS DISTRIBUTION GRAPH
    # ==================================================

    plt.figure(figsize=(7, 5))

    status_counts = df['status'].value_counts()

    plt.bar(
        status_counts.index,
        status_counts.values
    )

    plt.title('Placement Status Distribution')
    plt.xlabel('Placement Status')
    plt.ylabel('Number of Students')

    plt.tight_layout()

    plt.savefig(
        'static/graphs/status_distribution.png'
    )

    plt.close()


    # ==================================================
    # 2. FEATURE ANALYSIS GRAPH
    # ==================================================

    numeric_features = [
        'ssc_p',
        'hsc_p',
        'degree_p',
        'etest_p',
        'mba_p'
    ]

    plt.figure(figsize=(10, 6))

    for feature in numeric_features:

        plt.scatter(
            df[feature],
            df['status'].map({
                'Placed': 1,
                'Not Placed': 0
            }),
            label=feature,
            alpha=0.5
        )

    plt.title('Feature Analysis with Placement Status')

    plt.xlabel('Feature Values')

    plt.ylabel('Placement Status')

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        'static/graphs/feature_analysis.png'
    )

    plt.close()


    # ==================================================
    # PREPARE DATA FOR ACTUAL VS PREDICTED + ROC
    # ==================================================

    X = df.drop('status', axis=1)

    y = df['status']

    categorical_cols = [
        'gender',
        'ssc_b',
        'hsc_b',
        'hsc_s',
        'degree_t',
        'workex',
        'specialisation'
    ]


    # Encode categorical values
    for col in categorical_cols:

        X[col] = encoders[col].transform(
            X[col]
        )


    # Encode target
    y_encoded = encoders['status'].transform(y)


    # Ensure feature order
    X = X[feature_order]


    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y_encoded,

        test_size=0.2,

        random_state=42,

        stratify=y_encoded

    )


    # ==================================================
    # 3. ACTUAL VS PREDICTED PLACEMENT STATUS
    # ==================================================

    y_pred = model.predict(X_test)


    cm = confusion_matrix(
        y_test,
        y_pred
    )


    plt.figure(figsize=(6, 5))


    sns.heatmap(

        cm,

        annot=True,

        fmt='d',

        cmap='Blues',

        xticklabels=[
            'Not Placed',
            'Placed'
        ],

        yticklabels=[
            'Not Placed',
            'Placed'
        ]

    )


    plt.title(
        'Actual vs Predicted Placement Status'
    )


    plt.xlabel(
        'Predicted Status'
    )


    plt.ylabel(
        'Actual Status'
    )


    plt.tight_layout()


    plt.savefig(
        'static/graphs/actual_vs_predicted.png'
    )


    plt.close()


    # ==================================================
    # 4. ROC CURVE
    # ==================================================

    y_probability = model.predict_proba(
        X_test
    )[:, 1]


    fpr, tpr, thresholds = roc_curve(

        y_test,

        y_probability

    )


    roc_auc = auc(
        fpr,
        tpr
    )


    plt.figure(
        figsize=(7, 5)
    )


    plt.plot(

        fpr,

        tpr,

        label=f'ROC Curve (AUC = {roc_auc:.2f})'

    )


    plt.plot(

        [0, 1],

        [0, 1],

        linestyle='--'

    )


    plt.xlabel(
        'False Positive Rate'
    )


    plt.ylabel(
        'True Positive Rate'
    )


    plt.title(
        'ROC Curve for Placement Prediction'
    )


    plt.legend(
        loc='lower right'
    )


    plt.grid()


    plt.tight_layout()


    plt.savefig(
        'static/graphs/roc_curve.png'
    )


    plt.close()


    # Accuracy

    accuracy = accuracy_score(
        y_test,
        y_pred
    )


    return render_template(

        'analytics.html',

        accuracy=round(
            accuracy * 100,
            2
        ),

        auc_score=round(
            roc_auc,
            3
        )

    )
    # Encode categorical fields using the SAME encoders used during training
    categorical_cols = ['gender', 'ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'workex', 'specialisation']
    for col in categorical_cols:
        raw[col] = encoders[col].transform([raw[col]])[0]

    # Build feature vector in the exact order the model was trained on
    features = pd.DataFrame([[raw[col] for col in feature_order]], columns=feature_order)

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    status = encoders['status'].inverse_transform([pred])[0]
    confidence = round(max(proba) * 100, 1)

    return render_template(
        'result.html',
        status=status,
        confidence=confidence,
        placed=(status == 'Placed')
    )


if __name__ == '__main__':
    app.run(debug=True)
