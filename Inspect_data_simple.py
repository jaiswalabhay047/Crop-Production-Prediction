import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, log_loss, roc_auc_score,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)


# LOAD DATASET


df = pd.read_csv(r"C:\Users\Abhay\OneDrive\Desktop\Crop-Production-Prediction\Dataset\CROP.csv")

print("\n--- DATASET SHAPE ---")
print(df.shape)

print("\n--- FIRST FIVE ROWS ---")
print(df.head())

print("\n--- LAST FIVE ROWS ---")
print(df.tail())

print("\n--- DATASET INFORMATION ---")
df.info()

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\n--- DUPLICATE ROWS ---")
print(df.duplicated().sum())

print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())


# UNIQUE CATEGORIES


print("\n--- UNIQUE CATEGORIES ---")

for column in ["State", "Crop", "Season", "Production Units", "Area Units"]:
    print(f"\n{column}:")
    print("Unique values:", df[column].nunique())
    print(df[column].value_counts(dropna=False).head(10))


# MISSING VALUE DETAILS


print("\n--- MISSING VALUE DETAILS ---")

columns_to_check = [
    "Crop", "Season", "Area", "Production", "Yield"
]

for column in columns_to_check:
    missing_rows = df[df[column].isna()]

    print(f"\n{column}: {len(missing_rows)} missing values")

    print(
        missing_rows[
            ["State", "District", "Crop", "Year",
             "Season", "Area", "Production", "Yield"]
        ].head(5)
    )


# MISSING VALUE PATTERN

print("\n--- MISSING VALUE PATTERN ---")

missing_per_row = df.isnull().sum(axis=1)

print("Rows with 0 missing values:", (missing_per_row == 0).sum())
print("Rows with 1 missing value:", (missing_per_row == 1).sum())
print("Rows with 2 missing values:", (missing_per_row == 2).sum())
print("Rows with 3 or more missing values:", (missing_per_row >= 3).sum())


# CLEANING ANALYSIS


print("\n--- CLEANING ANALYSIS ---")

missing_target = df["Production"].isna().sum()
print("Rows missing Production:", missing_target)

df_analysis = df.copy()
df_analysis = df_analysis.dropna(subset=["Production"])

print("Original dataset rows:", len(df))
print("Rows after excluding missing Production:", len(df_analysis))

print("Remaining missing values:")
print(df_analysis.isnull().sum())

# remove extra spaces such as "Kharif     " so one category is not counted twice
for column in ["State", "District", "Crop", "Season"]:
    df_analysis[column] = df_analysis[column].str.strip()

# remove duplicate rows
rows_before = len(df_analysis)
df_analysis = df_analysis.drop_duplicates()
print("Duplicate rows removed:", rows_before - len(df_analysis))


# PRODUCTION UNIT ANALYSIS


print("\n--- PRODUCTION UNIT ANALYSIS ---")
print(df_analysis["Production Units"].value_counts())

print("\n--- PRODUCTION SUMMARY BY UNIT ---")

print(
    df_analysis.groupby("Production Units")["Production"]
    .agg(["count", "mean", "median", "min", "max"])
)


# TONNES DATASET


print("\n--- TONNES DATASET ANALYSIS ---")

df_tonnes = df_analysis[
    df_analysis["Production Units"] == "Tonnes"
].copy()

# keep one area unit and positive area only
df_tonnes = df_tonnes[df_tonnes["Area Units"] == "Hectare"].copy()
df_tonnes = df_tonnes[df_tonnes["Area"] > 0].copy()

# turn Year like "2001-02" into the number 2001
df_tonnes["Year_num"] = df_tonnes["Year"].astype(str).str[:4].astype(int)

# log values make very large numbers smaller and easier to learn from
df_tonnes["Log_Area"] = np.log1p(df_tonnes["Area"])
df_tonnes["Log_Production"] = np.log1p(df_tonnes["Production"])

print("Rows in tonnes dataset:", len(df_tonnes))
print("\nMissing values:")
print(df_tonnes.isnull().sum())

print("\nNumber of unique crops:")
print(df_tonnes["Crop"].nunique())

print("\nTop 15 crops by record count:")
print(df_tonnes["Crop"].value_counts().head(15))

print("\nProduction summary:")
print(df_tonnes["Production"].describe())

print("\nArea summary:")
print(df_tonnes["Area"].describe())


# TOP 10 CROPS BY TOTAL PRODUCTION


top_crops = (
    df_tonnes.groupby("Crop")["Production"]
    .sum()
    .nlargest(10)
    .sort_values()
) / 1e6

plt.figure(figsize=(10, 6))
top_crops.plot(kind="barh", color="tab:green")
plt.title("Top 10 Crops by Total Recorded Production")
plt.xlabel("Total Production (Million Tonnes)")
plt.ylabel("Crop")
plt.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("graph1_top_crops.png", dpi=150)
plt.show()


# AREA VS PRODUCTION


plot_data = df_tonnes[df_tonnes["Production"] > 0].sample(
    n=5000,
    random_state=42
)

plt.figure(figsize=(9, 6))
plt.scatter(
    plot_data["Area"],
    plot_data["Production"],
    alpha=0.35,
    s=12,
    color="tab:green"
)
plt.title("Cultivated Area vs Crop Production")
plt.xlabel("Area (Hectare, log scale)")
plt.ylabel("Production (Tonnes, log scale)")
plt.xscale("log")
plt.yscale("log")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graph2_area_vs_production.png", dpi=150)
plt.show()


# PRODUCTION DISTRIBUTION


fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].hist(df_tonnes["Production"], bins=50, color="tab:blue")
axes[0].set_yscale("log")
axes[0].set_title("Production (Original)")
axes[0].set_xlabel("Production (Tonnes)")
axes[0].set_ylabel("Number of Records (log scale)")

axes[1].hist(df_tonnes["Log_Production"], bins=50, color="tab:orange")
axes[1].set_title("Production (After Log)")
axes[1].set_xlabel("log(1 + Production)")
axes[1].set_ylabel("Number of Records")

plt.tight_layout()
plt.savefig("graph3_production_distribution.png", dpi=150)
plt.show()


# CORRELATION


print("\n--- CORRELATION ---")

corr_columns = ["Area", "Production", "Yield", "Year_num", "Log_Area", "Log_Production"]
corr_matrix = df_tonnes[corr_columns].corr()
print(corr_matrix.round(3))


# GRAPH 4: CORRELATION HEATMAP


plt.figure(figsize=(8, 7))
plt.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr_columns)), corr_columns, rotation=45, ha="right")
plt.yticks(range(len(corr_columns)), corr_columns)

for i in range(len(corr_columns)):
    for j in range(len(corr_columns)):
        plt.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center")

plt.title("Correlation Between Numerical Columns")
plt.tight_layout()
plt.savefig("graph4_correlation_heatmap.png", dpi=150)
plt.show()


# FEATURE SELECTION


print("\n--- FEATURE SELECTION ---")

features = [
    "State", "District", "Crop", "Year_num", "Season", "Log_Area"
]
target = "Production"

X = df_tonnes[features].copy()
y = df_tonnes[target].copy()

print("Feature matrix shape:", X.shape)
print("Target shape:", y.shape)
print("\nFeature columns:")
print(X.columns.tolist())


# TRAIN-TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

# the models learn from log(1 + Production), predictions are converted back to tonnes
y_train_log = np.log1p(y_train)

print("\n--- TRAIN TEST SPLIT ---")
print("Training X:", X_train.shape)
print("Testing X:", X_test.shape)


# ONE-HOT ENCODING


categorical_features = [
    "State", "District", "Crop", "Season"
]
numerical_features = ["Year_num", "Log_Area"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        ("numerical", StandardScaler(), numerical_features)
    ]
)

X_train_encoded = preprocessor.fit_transform(X_train)
X_test_encoded = preprocessor.transform(X_test)

print("\n--- ENCODING RESULTS ---")
print("Training data before encoding:", X_train.shape)
print("Training data after encoding:", X_train_encoded.shape)

model_names = []
r2_scores = []


# BASELINE: ALWAYS PREDICT THE AVERAGE


baseline_pred = np.full(len(y_test), y_train.mean())

print("\n--- BASELINE (AVERAGE) RESULTS ---")
print("MAE:", round(mean_absolute_error(y_test, baseline_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, baseline_pred)), 2))
print("R2 Score:", round(r2_score(y_test, baseline_pred), 4))

model_names.append("Baseline")
r2_scores.append(r2_score(y_test, baseline_pred))


# SIMPLE LINEAR REGRESSION (ONLY AREA)


simple_model = LinearRegression()
simple_model.fit(X_train[["Log_Area"]], y_train_log)

simple_pred = np.maximum(np.expm1(simple_model.predict(X_test[["Log_Area"]])), 0)

print("\n--- SIMPLE LINEAR REGRESSION RESULTS ---")
print("MAE:", round(mean_absolute_error(y_test, simple_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, simple_pred)), 2))
print("R2 Score:", round(r2_score(y_test, simple_pred), 4))

model_names.append("Simple Linear")
r2_scores.append(r2_score(y_test, simple_pred))


# POLYNOMIAL REGRESSION (ONLY AREA)


poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train[["Log_Area"]])
X_test_poly = poly.transform(X_test[["Log_Area"]])

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train_log)

poly_pred = np.maximum(np.expm1(poly_model.predict(X_test_poly)), 0)

print("\n--- POLYNOMIAL REGRESSION RESULTS ---")
print("MAE:", round(mean_absolute_error(y_test, poly_pred), 2))
print("RMSE:", round(np.sqrt(mean_squared_error(y_test, poly_pred)), 2))
print("R2 Score:", round(r2_score(y_test, poly_pred), 4))

model_names.append("Polynomial")
r2_scores.append(r2_score(y_test, poly_pred))


# SIMPLE VS POLYNOMIAL REGRESSION


line_sample = df_tonnes.sample(n=5000, random_state=42)
area_line = pd.DataFrame(
    {"Log_Area": np.linspace(df_tonnes["Log_Area"].min(), df_tonnes["Log_Area"].max(), 200)}
)

plt.figure(figsize=(9, 6))
plt.scatter(
    line_sample["Log_Area"],
    line_sample["Log_Production"],
    alpha=0.25,
    s=10,
    color="gray",
    label="Data"
)
plt.plot(
    area_line["Log_Area"],
    simple_model.predict(area_line),
    color="red",
    label="Simple Linear"
)
plt.plot(
    area_line["Log_Area"],
    poly_model.predict(poly.transform(area_line)),
    color="blue",
    linestyle="--",
    label="Polynomial (degree 2)"
)
plt.title("Simple vs Polynomial Regression")
plt.xlabel("log(1 + Area)")
plt.ylabel("log(1 + Production)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graph5_simple_vs_polynomial.png", dpi=150)
plt.show()


# MULTIPLE LINEAR REGRESSION


model = LinearRegression()
model.fit(X_train_encoded, y_train_log)

y_pred = np.maximum(np.expm1(model.predict(X_test_encoded)), 0)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- MULTIPLE LINEAR REGRESSION RESULTS ---")
print("MAE:", round(mae, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

model_names.append("Multiple Linear")
r2_scores.append(r2)


# DECISION TREE REGRESSION


tree_model = DecisionTreeRegressor(
    max_depth=12,
    min_samples_leaf=10,
    random_state=42
)

tree_model.fit(X_train_encoded, y_train_log)
tree_pred = np.maximum(np.expm1(tree_model.predict(X_test_encoded)), 0)

tree_mae = mean_absolute_error(y_test, tree_pred)
tree_rmse = np.sqrt(mean_squared_error(y_test, tree_pred))
tree_r2 = r2_score(y_test, tree_pred)

print("\n--- DECISION TREE REGRESSION RESULTS ---")
print("MAE:", round(tree_mae, 2))
print("RMSE:", round(tree_rmse, 2))
print("R2 Score:", round(tree_r2, 4))

model_names.append("Decision Tree")
r2_scores.append(tree_r2)


# RANDOM FOREST REGRESSION


rf_model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
rf_model.fit(X_train_encoded, y_train_log)

rf_pred = np.maximum(np.expm1(rf_model.predict(X_test_encoded)), 0)

rf_mae = mean_absolute_error(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)

print("\n--- RANDOM FOREST RESULTS ---")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 4))

model_names.append("Random Forest")
r2_scores.append(rf_r2)


# MODEL COMPARISON (R2)


plt.figure(figsize=(10, 6))
bars = plt.bar(model_names, r2_scores, color="tab:blue")

plt.title("Regression Model Comparison - R2 Score")
plt.xlabel("Machine Learning Model")
plt.ylabel("R2 Score")
plt.ylim(min(0, min(r2_scores)) - 0.05, 1.05)
plt.xticks(rotation=20)
plt.grid(axis="y", alpha=0.3)

for bar, score in zip(bars, r2_scores):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        max(bar.get_height(), 0),
        f"{score:.3f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig("graph6_model_comparison.png", dpi=150)
plt.show()


# ACTUAL VS PREDICTED PRODUCTION


plt.figure(figsize=(8, 8))
plt.scatter(y_test + 1, rf_pred + 1, alpha=0.2, s=10, color="tab:purple")

min_value = min((y_test + 1).min(), (rf_pred + 1).min())
max_value = max((y_test + 1).max(), (rf_pred + 1).max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--",
    color="red",
    label="Perfect prediction"
)

plt.xscale("log")
plt.yscale("log")
plt.title("Random Forest: Actual vs Predicted Production")
plt.xlabel("Actual Production (Tonnes, log scale)")
plt.ylabel("Predicted Production (Tonnes, log scale)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("graph7_actual_vs_predicted.png", dpi=150)
plt.show()


# CLASSIFICATION: MAKE PRODUCTION CLASSES


print("\n--- CLASSIFICATION DATASET ---")

class_names = ["Low", "Medium", "High"]

df_clf = df_tonnes.copy()

# split production into 3 equal groups: Low, Medium, High
df_clf["Production_Class"] = pd.qcut(
    df_clf["Production"].rank(method="first"),
    q=3,
    labels=class_names
).astype(str)

print(df_clf["Production_Class"].value_counts())

# a sample of rows keeps KNN and SVM fast
df_clf = df_clf.sample(n=10000, random_state=42)

X_clf = df_clf[["State", "Crop", "Season", "Year_num", "Log_Area"]]
y_clf = df_clf["Production_Class"]

X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

clf_preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            ["State", "Crop", "Season"]
        ),
        ("numerical", StandardScaler(), ["Year_num", "Log_Area"])
    ],
    sparse_threshold=0
)

X_clf_train_encoded = clf_preprocessor.fit_transform(X_clf_train)
X_clf_test_encoded = clf_preprocessor.transform(X_clf_test)

print("Training data:", X_clf_train_encoded.shape)
print("Testing data:", X_clf_test_encoded.shape)


# CLASSIFICATION MODELS


classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=10, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
}

clf_names = []
clf_accuracy = []
clf_precision = []
clf_recall = []
clf_f1 = []
clf_predictions = {}

for name, clf in classifiers.items():
    print(f"\nTraining {name}...")
    clf.fit(X_clf_train_encoded, y_clf_train)

    clf_pred = clf.predict(X_clf_test_encoded)
    clf_proba = clf.predict_proba(X_clf_test_encoded)

    accuracy = accuracy_score(y_clf_test, clf_pred)
    precision = precision_score(y_clf_test, clf_pred, average="weighted")
    recall = recall_score(y_clf_test, clf_pred, average="weighted")
    f1 = f1_score(y_clf_test, clf_pred, average="weighted")
    loss = log_loss(y_clf_test, clf_proba, labels=clf.classes_)
    auc = roc_auc_score(y_clf_test, clf_proba, multi_class="ovr", labels=clf.classes_)

    print(f"\n--- {name.upper()} RESULTS ---")
    print("Accuracy:", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall:", round(recall, 4))
    print("F1 Score:", round(f1, 4))
    print("Log Loss:", round(loss, 4))
    print("AUC:", round(auc, 4))

    print("Confusion Matrix (rows = actual, columns = predicted):")
    print(pd.DataFrame(
        confusion_matrix(y_clf_test, clf_pred, labels=class_names),
        index=class_names,
        columns=class_names
    ))

    clf_names.append(name)
    clf_accuracy.append(accuracy)
    clf_precision.append(precision)
    clf_recall.append(recall)
    clf_f1.append(f1)
    clf_predictions[name] = clf_pred

best_name = clf_names[int(np.argmax(clf_f1))]
print("\nBest classifier (highest F1 Score):", best_name)


# CLASSIFICATION MODEL COMPARISON


positions = np.arange(len(clf_names))
width = 0.2

plt.figure(figsize=(11, 6))
plt.bar(positions - 1.5 * width, clf_accuracy, width, label="Accuracy")
plt.bar(positions - 0.5 * width, clf_precision, width, label="Precision")
plt.bar(positions + 0.5 * width, clf_recall, width, label="Recall")
plt.bar(positions + 1.5 * width, clf_f1, width, label="F1 Score")

plt.xticks(positions, clf_names)
plt.ylim(0, 1.05)
plt.title("Classification Model Comparison")
plt.xlabel("Machine Learning Model")
plt.ylabel("Score")
plt.legend(loc="lower right")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("graph8_classification_comparison.png", dpi=150)
plt.show()


# CONFUSION MATRIX OF THE BEST CLASSIFIER


fig, ax = plt.subplots(figsize=(7, 6))
ConfusionMatrixDisplay.from_predictions(
    y_clf_test,
    clf_predictions[best_name],
    labels=class_names,
    cmap="Blues",
    ax=ax
)
ax.set_title(f"Confusion Matrix - {best_name}")
plt.tight_layout()
plt.savefig("graph9_confusion_matrix.png", dpi=150)
plt.show()
