import io
import time
import textwrap
from dataclasses import dataclass

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from sklearn.base import clone
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris, load_wine
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from streamlit_image_coordinates import streamlit_image_coordinates
except ModuleNotFoundError:
    streamlit_image_coordinates = None


st.set_page_config(
    page_title="ML Studio GUI",
    page_icon="ML",
    layout="wide",
    initial_sidebar_state="expanded",
)


APP_CSS = """
<style>
    .block-container { padding-top: 1.3rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.55rem; }
    div[data-testid="stTabs"] button { font-size: 1rem; }
    .small-muted { color: #667085; font-size: 0.9rem; }
    .lab-panel {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 0.9rem 1rem;
        background: #fbfcff;
    }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


@dataclass
class TrainResult:
    model: Pipeline
    best_params: dict
    cv_scores: dict
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray | None
    target_names: list[str]


def load_builtin_dataset(name: str) -> tuple[pd.DataFrame, str]:
    loaders = {
        "Iris classification": load_iris,
        "Breast cancer classification": load_breast_cancer,
        "Wine classification": load_wine,
        "Diabetes regression": load_diabetes,
    }
    dataset = loaders[name]()
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df["target"] = dataset.target
    return df, "target"


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def coerce_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df.columns = [str(col).strip() for col in df.columns]
    return df


def add_manual_point(df: pd.DataFrame, point: dict) -> pd.DataFrame:
    row = {column: point.get(column, np.nan) for column in df.columns}
    return pd.concat([df, pd.DataFrame([row])], ignore_index=True)


def make_estimator(task: str, algorithm: str, params: dict):
    if task == "Regression":
        estimators = {
            "Linear Regression": LinearRegression(),
            "SVM": SVR(C=params["C"], kernel=params["kernel"], gamma=params["gamma"]),
            "KNN": KNeighborsRegressor(n_neighbors=params["n_neighbors"], weights=params["weights"]),
            "Decision Tree": DecisionTreeRegressor(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42,
            ),
            "Random Forest": RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
            ),
        }
    else:
        estimators = {
            "Logistic Regression": LogisticRegression(
                C=params["C"],
                max_iter=2000,
                solver="lbfgs",
                random_state=42,
            ),
            "SVM": SVC(
                C=params["C"],
                kernel=params["kernel"],
                gamma=params["gamma"],
                probability=True,
                random_state=42,
            ),
            "KNN": KNeighborsClassifier(n_neighbors=params["n_neighbors"], weights=params["weights"]),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                random_state=42,
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=42,
                n_jobs=-1,
            ),
        }
    return estimators[algorithm]


def make_grid(task: str, algorithm: str) -> dict:
    grids = {
        "Regression": {
            "Linear Regression": {},
            "SVM": {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale", "auto"]},
            "KNN": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
            "Decision Tree": {"model__max_depth": [None, 3, 5, 10], "model__min_samples_split": [2, 5, 10]},
            "Random Forest": {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 5, 10]},
        },
        "Classification": {
            "Logistic Regression": {"model__C": [0.1, 1.0, 10.0]},
            "SVM": {"model__C": [0.1, 1.0, 10.0], "model__kernel": ["linear", "rbf"], "model__gamma": ["scale", "auto"]},
            "KNN": {"model__n_neighbors": [3, 5, 9, 15], "model__weights": ["uniform", "distance"]},
            "Decision Tree": {"model__max_depth": [None, 3, 5, 10], "model__min_samples_split": [2, 5, 10]},
            "Random Forest": {"model__n_estimators": [50, 100, 200], "model__max_depth": [None, 5, 10]},
        },
    }
    return grids[task][algorithm]


def make_pipeline(estimator, scale_features: bool) -> Pipeline:
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_features:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def train_model(
    df: pd.DataFrame,
    target_col: str,
    task: str,
    algorithm: str,
    params: dict,
    scale_features: bool,
    tune_hyperparameters: bool,
    cv_folds: int,
    test_size: float,
) -> TrainResult:
    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df.drop(columns=[target_col])
    X = X.select_dtypes(include=[np.number])
    y_raw = clean_df[target_col].to_numpy()

    if task == "Classification":
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_raw)
        target_names = [str(item) for item in encoder.classes_]
        stratify = y if len(np.unique(y)) > 1 else None
    else:
        y = pd.to_numeric(clean_df[target_col], errors="coerce").to_numpy()
        keep = ~np.isnan(y)
        X, y = X.loc[keep], y[keep]
        target_names = ["target"]
        stratify = None

    if X.empty:
        raise ValueError("No numeric feature columns are available for training.")
    if len(X) < 8:
        raise ValueError("At least 8 valid rows are required for training and validation.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify,
    )

    estimator = make_estimator(task, algorithm, params)
    pipeline = make_pipeline(estimator, scale_features)
    grid = make_grid(task, algorithm)
    best_params = {}

    scoring = "r2" if task == "Regression" else "accuracy"
    if tune_hyperparameters and grid:
        search = GridSearchCV(
            pipeline,
            grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        model = search.best_estimator_
        best_params = search.best_params_
    else:
        model = pipeline.fit(X_train, y_train)

    cv_scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"} if task == "Regression" else {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    cv_scores = cross_validate(clone(model), X, y, cv=cv_folds, scoring=cv_scoring, n_jobs=-1)
    y_pred = model.predict(X_test)

    y_score = None
    if task == "Classification":
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)

    return TrainResult(
        model=model,
        best_params=best_params,
        cv_scores=cv_scores,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        y_pred=y_pred,
        y_score=y_score,
        target_names=target_names,
    )


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
    }


def classification_metrics(y_true, y_pred, y_score) -> dict:
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1 weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_score is not None and len(np.unique(y_true)) > 1:
        try:
            if np.ndim(y_score) == 1 or y_score.shape[1] == 2:
                score = y_score if np.ndim(y_score) == 1 else y_score[:, 1]
                metrics["ROC AUC"] = roc_auc_score(y_true, score)
            else:
                metrics["ROC AUC"] = roc_auc_score(y_true, y_score, multi_class="ovr")
        except ValueError:
            metrics["ROC AUC"] = np.nan
    return metrics


def plot_regression_line(result: TrainResult, feature: str, target_name: str):
    fig, ax = plt.subplots(figsize=(8, 5))
    X_plot = result.X_test.copy()
    y_true = result.y_test
    ax.scatter(X_plot[feature], y_true, alpha=0.75, label="Actual")

    line_frame = pd.DataFrame(
        np.tile(result.X_train.median(numeric_only=True).to_numpy(), (100, 1)),
        columns=result.X_train.columns,
    )
    line_frame[feature] = np.linspace(X_plot[feature].min(), X_plot[feature].max(), 100)
    line_pred = result.model.predict(line_frame)
    ax.plot(line_frame[feature], line_pred, color="#d62728", linewidth=2.4, label="Model prediction")
    ax.set_xlabel(feature)
    ax.set_ylabel(target_name)
    ax.set_title("Scatter Plot with Regression Line")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig


def plot_confusion(result: TrainResult):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    cm = confusion_matrix(result.y_test, result.y_pred)
    display = ConfusionMatrixDisplay(cm, display_labels=result.target_names)
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    plt.xticks(rotation=30, ha="right")
    return fig


def plot_roc(result: TrainResult):
    fig, ax = plt.subplots(figsize=(7, 5.2))
    classes = np.unique(result.y_test)
    if result.y_score is None or len(classes) < 2:
        ax.text(0.5, 0.5, "ROC is unavailable for this model/test split.", ha="center", va="center")
        ax.axis("off")
        return fig

    if len(classes) == 2:
        score = result.y_score if np.ndim(result.y_score) == 1 else result.y_score[:, 1]
        RocCurveDisplay.from_predictions(result.y_test, score, ax=ax)
    else:
        y_bin = label_binarize(result.y_test, classes=classes)
        for class_index, class_label in enumerate(classes):
            RocCurveDisplay.from_predictions(
                y_bin[:, class_index],
                result.y_score[:, class_index],
                ax=ax,
                name=result.target_names[class_label],
            )
    ax.set_title("ROC Curve")
    ax.grid(alpha=0.25)
    return fig


def plot_pca(df: pd.DataFrame, target_col: str, task: str):
    numeric_features = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    target = df[target_col]
    fig, ax = plt.subplots(figsize=(8, 5.2))
    if numeric_features.shape[1] < 2:
        ax.text(0.5, 0.5, "PCA needs at least two numeric feature columns.", ha="center", va="center")
        ax.axis("off")
        return fig

    X_imputed = SimpleImputer(strategy="median").fit_transform(numeric_features)
    X_scaled = StandardScaler().fit_transform(X_imputed)
    components = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    scatter = ax.scatter(components[:, 0], components[:, 1], c=pd.factorize(target)[0], cmap="viridis", alpha=0.78)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title(f"PCA Visualization ({task})")
    ax.grid(alpha=0.25)
    if target.nunique() <= 12:
        handles, _ = scatter.legend_elements()
        labels = [str(label) for label in sorted(target.dropna().unique())]
        ax.legend(handles[: len(labels)], labels, title=target_col, loc="best")
    return fig


def compare_models(df, target_col, task, algorithms, params, scale_features, cv_folds):
    rows = []
    clean_df = df.dropna(subset=[target_col]).copy()
    X = clean_df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    if task == "Classification":
        y = LabelEncoder().fit_transform(clean_df[target_col].to_numpy())
        scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    else:
        y = pd.to_numeric(clean_df[target_col], errors="coerce")
        keep = ~y.isna()
        X, y = X.loc[keep], y.loc[keep].to_numpy()
        scoring = {"r2": "r2", "neg_rmse": "neg_root_mean_squared_error"}

    for algorithm in algorithms:
        estimator = make_estimator(task, algorithm, params)
        pipeline = make_pipeline(estimator, scale_features)
        scores = cross_validate(pipeline, X, y, cv=cv_folds, scoring=scoring, n_jobs=-1)
        if task == "Classification":
            rows.append(
                {
                    "Algorithm": algorithm,
                    "Accuracy": scores["test_accuracy"].mean(),
                    "F1 weighted": scores["test_f1_weighted"].mean(),
                }
            )
        else:
            rows.append(
                {
                    "Algorithm": algorithm,
                    "R2": scores["test_r2"].mean(),
                    "RMSE": -scores["test_neg_rmse"].mean(),
                }
            )
    return pd.DataFrame(rows)


def plot_comparison(comparison_df: pd.DataFrame, task: str):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    metric = "Accuracy" if task == "Classification" else "R2"
    ax.bar(comparison_df["Algorithm"], comparison_df[metric], color="#315c9e")
    ax.set_title(f"Model Comparison by {metric}")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    return fig


def figure_to_image(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=100, bbox_inches=None)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def plot_clickable_scatter(df: pd.DataFrame, x_col: str, y_col: str, target_col: str):
    plot_df = df.dropna(subset=[x_col, y_col, target_col]).copy()
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=100)
    target_codes, _ = pd.factorize(plot_df[target_col])
    ax.scatter(plot_df[x_col], plot_df[y_col], c=target_codes, cmap="viridis", alpha=0.72, edgecolors="white", linewidths=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title("Click Plot to Add a Data Point")
    ax.grid(alpha=0.25)

    if len(plot_df) > 0:
        x_min, x_max = float(plot_df[x_col].min()), float(plot_df[x_col].max())
        y_min, y_max = float(plot_df[y_col].min()), float(plot_df[y_col].max())
    else:
        x_min, x_max, y_min, y_max = 0.0, 1.0, 0.0, 1.0

    x_pad = max((x_max - x_min) * 0.08, 1e-6)
    y_pad = max((y_max - y_min) * 0.08, 1e-6)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    fig.canvas.draw()

    image = figure_to_image(fig)
    bbox = ax.get_window_extent()
    image_width, image_height = image.size
    meta = {
        "image_width": image_width,
        "image_height": image_height,
        "left": float(bbox.x0),
        "right": float(bbox.x1),
        "top": float(image_height - bbox.y1),
        "bottom": float(image_height - bbox.y0),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
    }
    plt.close(fig)
    return image, meta


def image_click_to_data(click_value: dict, meta: dict):
    x_pixel = float(click_value["x"])
    y_pixel = float(click_value["y"])
    if not (meta["left"] <= x_pixel <= meta["right"] and meta["top"] <= y_pixel <= meta["bottom"]):
        return None

    x_min, x_max = meta["xlim"]
    y_min, y_max = meta["ylim"]
    x_ratio = (x_pixel - meta["left"]) / (meta["right"] - meta["left"])
    y_ratio = (y_pixel - meta["top"]) / (meta["bottom"] - meta["top"])
    data_x = x_min + x_ratio * (x_max - x_min)
    data_y = y_max - y_ratio * (y_max - y_min)
    return data_x, data_y


def make_point_from_click(base_df, target_col, x_col, y_col, data_x, data_y, target_value):
    point = {}
    for col in base_df.columns:
        if col == x_col:
            point[col] = float(data_x)
        elif col == y_col:
            point[col] = float(data_y)
        elif col == target_col:
            point[col] = target_value
        elif pd.api.types.is_numeric_dtype(base_df[col]):
            point[col] = float(pd.to_numeric(base_df[col], errors="coerce").median())
        else:
            values = base_df[col].dropna()
            point[col] = values.iloc[0] if len(values) else ""
    return point


def plot_training_frame(df, target_col, task, algorithm, params, scale_features, x_col, y_col, fraction):
    clean_df = df.dropna(subset=[target_col, x_col, y_col]).copy()
    sample_size = max(6, int(len(clean_df) * fraction))
    frame_df = clean_df.iloc[:sample_size].copy()
    fig, ax = plt.subplots(figsize=(7.4, 4.8))

    if len(frame_df) < 6:
        ax.text(0.5, 0.5, "Need at least 6 rows for animation.", ha="center", va="center")
        ax.axis("off")
        return fig

    X = frame_df[[x_col, y_col]]
    x_min, x_max = float(clean_df[x_col].min()), float(clean_df[x_col].max())
    y_min, y_max = float(clean_df[y_col].min()), float(clean_df[y_col].max())
    x_pad = max((x_max - x_min) * 0.1, 1e-6)
    y_pad = max((y_max - y_min) * 0.1, 1e-6)
    xx, yy = np.meshgrid(
        np.linspace(x_min - x_pad, x_max + x_pad, 130),
        np.linspace(y_min - y_pad, y_max + y_pad, 130),
    )
    grid = pd.DataFrame({x_col: xx.ravel(), y_col: yy.ravel()})

    try:
        estimator = make_estimator(task, algorithm, params)
        model = make_pipeline(estimator, scale_features)
        if task == "Classification":
            y = LabelEncoder().fit_transform(frame_df[target_col].to_numpy())
            if len(np.unique(y)) < 2:
                raise ValueError("Need at least two classes in the current frame.")
            model.fit(X, y)
            prediction_grid = model.predict(grid).reshape(xx.shape)
            target_codes, _ = pd.factorize(frame_df[target_col])
            ax.contourf(xx, yy, prediction_grid, alpha=0.22, cmap="viridis")
            ax.scatter(frame_df[x_col], frame_df[y_col], c=target_codes, cmap="viridis", edgecolors="white", linewidths=0.5)
            ax.set_title(f"{algorithm} Classification Animation - {sample_size}/{len(clean_df)} rows")
        else:
            y = pd.to_numeric(frame_df[target_col], errors="coerce").to_numpy()
            model.fit(X, y)
            prediction_grid = model.predict(grid).reshape(xx.shape)
            contour = ax.contourf(xx, yy, prediction_grid, levels=16, alpha=0.35, cmap="viridis")
            fig.colorbar(contour, ax=ax, label="Predicted target")
            ax.scatter(frame_df[x_col], frame_df[y_col], c=y, cmap="viridis", edgecolors="white", linewidths=0.5)
            ax.set_title(f"{algorithm} Regression Animation - {sample_size}/{len(clean_df)} rows")
    except Exception as exc:
        ax.scatter(frame_df[x_col], frame_df[y_col], alpha=0.75)
        ax.text(0.5, 0.95, str(exc), transform=ax.transAxes, ha="center", va="top")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(alpha=0.22)
    return fig


LAB_EXPLANATIONS = {
    "Linear Regression": "Fits a straight line through clicked or generated points and reports R2/RMSE.",
    "Multilinear Regression": "Uses x and y together as features and predicts a synthetic target surface.",
    "K-Means": "Partitions points into k clusters by repeatedly moving centroids to cluster means.",
    "KNN": "Classifies the 2D space by the labels of the nearest clicked/generated neighbors.",
    "DBSCAN": "Finds dense point groups and marks sparse points as noise without choosing k.",
    "Decision Tree": "Builds rule-based rectangular decision regions from labeled points.",
}


def empty_lab_points():
    return pd.DataFrame(columns=["x", "y", "label"])


def generate_lab_points(count: int, noise: float, algorithm: str):
    rng = np.random.default_rng()
    count = int(count)
    noise_scale = noise / 100.0

    if algorithm in {"Linear Regression", "Multilinear Regression"}:
        x = rng.uniform(5, 95, count)
        base_y = 0.62 * x + 18
        y = np.clip(base_y + rng.normal(0, 7 + 28 * noise_scale, count), 0, 100)
        labels = np.where(y > np.median(y), "B", "A")
    else:
        half = max(1, count // 2)
        cluster_a = rng.normal([32, 35], [10 + 12 * noise_scale, 9 + 12 * noise_scale], size=(half, 2))
        cluster_b = rng.normal([70, 68], [11 + 12 * noise_scale, 10 + 12 * noise_scale], size=(count - half, 2))
        points = np.vstack([cluster_a, cluster_b])
        points = np.clip(points, 0, 100)
        x, y = points[:, 0], points[:, 1]
        labels = np.array(["A"] * half + ["B"] * (count - half))

    return pd.DataFrame({"x": x, "y": y, "label": labels})


def plot_lab_base(points: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(8.2, 5.6), dpi=100)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    if not points.empty:
        colors = points["label"].map({"A": "#2563eb", "B": "#ef4444"}).fillna("#64748b")
        ax.scatter(points["x"], points["y"], c=colors, s=58, edgecolors="white", linewidths=0.7, alpha=0.9)
    return fig, ax


def plot_lab_click_canvas(points: pd.DataFrame, algorithm: str):
    fig, _ = plot_lab_base(points, f"Interactive Canvas - {algorithm}")
    fig.canvas.draw()
    image = figure_to_image(fig)
    ax = fig.axes[0]
    bbox = ax.get_window_extent()
    image_width, image_height = image.size
    meta = {
        "left": float(bbox.x0),
        "right": float(bbox.x1),
        "top": float(image_height - bbox.y1),
        "bottom": float(image_height - bbox.y0),
        "xlim": ax.get_xlim(),
        "ylim": ax.get_ylim(),
    }
    plt.close(fig)
    return image, meta


def add_lab_point(points: pd.DataFrame, x: float, y: float, label: str):
    new_row = pd.DataFrame([{"x": float(np.clip(x, 0, 100)), "y": float(np.clip(y, 0, 100)), "label": label}])
    return pd.concat([points, new_row], ignore_index=True)


def remove_nearest_lab_point(points: pd.DataFrame, x: float, y: float):
    if points.empty:
        return points
    distances = np.sqrt((points["x"].astype(float) - x) ** 2 + (points["y"].astype(float) - y) ** 2)
    nearest_index = distances.idxmin()
    if distances.loc[nearest_index] <= 7:
        return points.drop(index=nearest_index).reset_index(drop=True)
    return points


def lab_grid():
    xx, yy = np.meshgrid(np.linspace(0, 100, 160), np.linspace(0, 100, 160))
    grid = np.c_[xx.ravel(), yy.ravel()]
    return xx, yy, grid


def plot_lab_algorithm(points: pd.DataFrame, algorithm: str, controls: dict):
    fig, ax = plot_lab_base(points, f"{algorithm} Playground Result")
    metrics = {"Status": "Ready"}

    if len(points) < 2:
        metrics["Status"] = "Add at least 2 points"
        return fig, metrics

    X = points[["x", "y"]].to_numpy(dtype=float)
    y_labels = LabelEncoder().fit_transform(points["label"].astype(str))

    try:
        if algorithm == "Linear Regression":
            model = LinearRegression().fit(points[["x"]], points["y"])
            line_x = np.linspace(0, 100, 120)
            line_y = model.predict(pd.DataFrame({"x": line_x}))
            ax.plot(line_x, line_y, color="#111827", linewidth=2.4)
            pred = model.predict(points[["x"]])
            metrics = {
                "R2": f"{r2_score(points['y'], pred):.3f}",
                "RMSE": f"{np.sqrt(mean_squared_error(points['y'], pred)):.3f}",
                "Slope": f"{model.coef_[0]:.3f}",
            }

        elif algorithm == "Multilinear Regression":
            target = 0.45 * points["x"].to_numpy() + 0.55 * points["y"].to_numpy()
            model = LinearRegression().fit(X, target)
            xx, yy, grid = lab_grid()
            zz = model.predict(grid).reshape(xx.shape)
            contour = ax.contourf(xx, yy, zz, levels=18, alpha=0.32, cmap="viridis")
            fig.colorbar(contour, ax=ax, label="Predicted value")
            metrics = {
                "R2": f"{r2_score(target, model.predict(X)):.3f}",
                "Coef X": f"{model.coef_[0]:.3f}",
                "Coef Y": f"{model.coef_[1]:.3f}",
            }

        elif algorithm == "K-Means":
            k = min(int(controls["k"]), len(points))
            model = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
            labels = model.labels_
            for collection in list(ax.collections):
                collection.remove()
            ax.scatter(points["x"], points["y"], c=labels, cmap="tab10", s=58, edgecolors="white", linewidths=0.7)
            ax.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], marker="X", s=220, c="#111827", label="Centroids")
            ax.legend(loc="best")
            metrics = {"Clusters": str(k), "Inertia": f"{model.inertia_:.2f}"}
            if len(set(labels)) > 1 and len(points) > k:
                metrics["Silhouette"] = f"{silhouette_score(X, labels):.3f}"

        elif algorithm == "KNN":
            if len(np.unique(y_labels)) < 2:
                metrics = {"Status": "Need both class A and B"}
            else:
                model = KNeighborsClassifier(n_neighbors=min(int(controls["neighbors"]), len(points))).fit(X, y_labels)
                xx, yy, grid = lab_grid()
                pred = model.predict(grid).reshape(xx.shape)
                ax.contourf(xx, yy, pred, alpha=0.2, cmap="coolwarm")
                metrics = {"Training accuracy": f"{accuracy_score(y_labels, model.predict(X)):.3f}", "Neighbors": str(model.n_neighbors)}

        elif algorithm == "DBSCAN":
            model = DBSCAN(eps=float(controls["eps"]), min_samples=int(controls["min_samples"])).fit(X)
            labels = model.labels_
            for collection in list(ax.collections):
                collection.remove()
            ax.scatter(points["x"], points["y"], c=labels, cmap="tab10", s=58, edgecolors="white", linewidths=0.7)
            cluster_count = len(set(labels)) - (1 if -1 in labels else 0)
            noise_count = int(np.sum(labels == -1))
            metrics = {"Clusters": str(cluster_count), "Noise points": str(noise_count)}
            valid_labels = set(labels) - {-1}
            if len(valid_labels) > 1:
                metrics["Silhouette"] = f"{silhouette_score(X, labels):.3f}"

        elif algorithm == "Decision Tree":
            if len(np.unique(y_labels)) < 2:
                metrics = {"Status": "Need both class A and B"}
            else:
                model = DecisionTreeClassifier(max_depth=controls["max_depth"], random_state=42).fit(X, y_labels)
                xx, yy, grid = lab_grid()
                pred = model.predict(grid).reshape(xx.shape)
                ax.contourf(xx, yy, pred, alpha=0.2, cmap="coolwarm")
                metrics = {"Training accuracy": f"{accuracy_score(y_labels, model.predict(X)):.3f}", "Depth": str(model.get_depth())}

    except Exception as exc:
        metrics = {"Status": str(exc)}

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    return fig, metrics


def model_report_text(task, algorithm, result, metrics, comparison_df) -> str:
    best_params = result.best_params if result.best_params else "Not used"
    cv_lines = []
    for key, value in result.cv_scores.items():
        if key.startswith("test_"):
            cv_lines.append(f"{key.replace('test_', '')}: mean={np.mean(value):.4f}, std={np.std(value):.4f}")

    comparison_text = "No comparison data generated."
    if comparison_df is not None and not comparison_df.empty:
        comparison_text = comparison_df.to_string(index=False)

    return textwrap.dedent(
        f"""
        ML Studio Report
        ================
        Task: {task}
        Selected algorithm: {algorithm}

        Test metrics:
        {pd.Series(metrics).to_string()}

        Cross-validation:
        {chr(10).join(cv_lines)}

        Best GridSearchCV parameters:
        {best_params}

        Model comparison:
        {comparison_text}
        """
    ).strip()


def render_parameter_controls(task: str, algorithm: str) -> dict:
    params = {
        "C": 1.0,
        "kernel": "rbf",
        "gamma": "scale",
        "n_neighbors": 5,
        "weights": "uniform",
        "max_depth": None,
        "min_samples_split": 2,
        "n_estimators": 100,
    }

    if algorithm in {"Logistic Regression", "SVM"}:
        params["C"] = st.sidebar.slider("Regularization C", 0.01, 20.0, 1.0, 0.01)
    if algorithm == "SVM":
        params["kernel"] = st.sidebar.selectbox("Kernel", ["rbf", "linear", "poly", "sigmoid"])
        params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
    if algorithm == "KNN":
        params["n_neighbors"] = st.sidebar.slider("Neighbors", 1, 31, 5, 2)
        params["weights"] = st.sidebar.selectbox("Weights", ["uniform", "distance"])
    if algorithm in {"Decision Tree", "Random Forest"}:
        depth_enabled = st.sidebar.checkbox("Limit tree depth", value=False)
        params["max_depth"] = st.sidebar.slider("Max depth", 1, 30, 6) if depth_enabled else None
        params["min_samples_split"] = st.sidebar.slider("Min samples split", 2, 20, 2)
    if algorithm == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Trees", 10, 400, 100, 10)
    return params


def initialize_state():
    if "manual_points" not in st.session_state:
        st.session_state.manual_points = []
    if "last_plot_click" not in st.session_state:
        st.session_state.last_plot_click = None
    if "lab_points" not in st.session_state:
        st.session_state.lab_points = pd.DataFrame(columns=["x", "y", "label"])
    if "last_lab_click" not in st.session_state:
        st.session_state.last_lab_click = None


initialize_state()

st.title("Machine Learning GUI Studio")
st.caption("End-to-end scikit-learn training, visualization, comparison, documentation, and export in Streamlit.")

with st.sidebar:
    st.header("Configuration")
    task = st.radio("Task", ["Classification", "Regression"], horizontal=True)
    algorithm_options = {
        "Classification": ["Logistic Regression", "SVM", "KNN", "Decision Tree", "Random Forest"],
        "Regression": ["Linear Regression", "SVM", "KNN", "Decision Tree", "Random Forest"],
    }
    algorithm = st.selectbox("Algorithm", algorithm_options[task])
    params = render_parameter_controls(task, algorithm)
    st.divider()
    scale_features = st.checkbox("Feature scaling", value=True)
    tune_hyperparameters = st.checkbox("Hyperparameter tuning (GridSearchCV)", value=False)
    cv_folds = st.slider("Cross-validation folds", 3, 10, 5)
    test_size = st.slider("Test size", 0.15, 0.45, 0.25, 0.05)
    st.button("Run / refresh algorithm", type="primary", use_container_width=True)
    st.markdown('<div class="small-muted">The selected algorithm trains automatically whenever settings or data change.</div>', unsafe_allow_html=True)

dataset_tab, metrics_tab, graphs_tab, playground_tab = st.tabs(["Dataset", "Metrics", "Graphs", "Playground"])

with dataset_tab:
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.subheader("Dataset Source")
        source = st.radio("Choose data source", ["Built-in dataset", "Upload CSV"], horizontal=True)
        default_builtin = "Iris classification" if task == "Classification" else "Diabetes regression"
        builtin_options = ["Iris classification", "Breast cancer classification", "Wine classification"] if task == "Classification" else ["Diabetes regression"]

        if source == "Built-in dataset":
            builtin_name = st.selectbox("Built-in dataset", builtin_options, index=builtin_options.index(default_builtin))
            base_df, suggested_target = load_builtin_dataset(builtin_name)
        else:
            uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
            if uploaded_file is None:
                base_df, suggested_target = load_builtin_dataset(default_builtin)
                st.info("Using a built-in dataset until a CSV file is uploaded.")
            else:
                base_df = coerce_uploaded_dataset(uploaded_file)
                suggested_target = base_df.columns[-1]

        target_col = st.selectbox(
            "Target column",
            base_df.columns.tolist(),
            index=base_df.columns.tolist().index(suggested_target) if suggested_target in base_df.columns else len(base_df.columns) - 1,
        )

    with right:
        st.subheader("Manual Point Input")
        st.markdown('<div class="small-muted">Add rows interactively with coordinate inputs. Numeric feature fields are prefilled with column medians.</div>', unsafe_allow_html=True)
        numeric_cols = get_numeric_columns(base_df)
        point = {}
        with st.form("manual_point_form", clear_on_submit=False):
            for col in base_df.columns:
                if col in numeric_cols:
                    default_value = float(pd.to_numeric(base_df[col], errors="coerce").median())
                    point[col] = st.number_input(col, value=default_value, format="%.6f")
                else:
                    values = [str(v) for v in base_df[col].dropna().unique()[:50]]
                    point[col] = st.selectbox(col, values if values else [""], index=0)
            add_point = st.form_submit_button("Add data point")
        if add_point:
            st.session_state.manual_points.append(point)
            st.success("Point added to the active dataset.")

        if st.button("Clear manual points", disabled=not st.session_state.manual_points):
            st.session_state.manual_points = []
            st.session_state.last_plot_click = None
            st.rerun()

    active_df = base_df.copy()
    for manual_point in st.session_state.manual_points:
        active_df = add_manual_point(active_df, manual_point)

    st.subheader("Click-Based Point Plot")
    plot_feature_cols = [col for col in get_numeric_columns(active_df) if col != target_col]
    if len(plot_feature_cols) < 2:
        st.info("Click plotting needs at least two numeric feature columns.")
    elif streamlit_image_coordinates is None:
        st.error("Install streamlit-image-coordinates from requirements.txt to enable click-based point plotting.")
    else:
        click_left, click_right = st.columns([0.72, 0.28], gap="large")
        with click_right:
            click_x_col = st.selectbox("Click plot X", plot_feature_cols, index=0)
            y_default_index = 1 if len(plot_feature_cols) > 1 else 0
            click_y_col = st.selectbox("Click plot Y", plot_feature_cols, index=y_default_index)
            if task == "Classification":
                target_options = sorted(active_df[target_col].dropna().unique().tolist())
                click_target_value = st.selectbox("Class for clicked point", target_options)
            else:
                target_default = float(pd.to_numeric(active_df[target_col], errors="coerce").median())
                click_target_value = st.number_input("Target for clicked point", value=target_default, format="%.6f")
            st.markdown('<div class="small-muted">Click inside the plot area to append a row. Other feature values use dataset medians.</div>', unsafe_allow_html=True)

        with click_left:
            click_image, click_meta = plot_clickable_scatter(active_df, click_x_col, click_y_col, target_col)
            click_value = streamlit_image_coordinates(click_image, key=f"click_plot_{click_x_col}_{click_y_col}_{target_col}", width=700)
            if click_value is not None:
                data_coords = image_click_to_data(click_value, click_meta)
                click_signature = (
                    click_x_col,
                    click_y_col,
                    target_col,
                    round(float(click_value["x"]), 2),
                    round(float(click_value["y"]), 2),
                    str(click_target_value),
                )
                if data_coords is not None and st.session_state.last_plot_click != click_signature:
                    new_point = make_point_from_click(
                        base_df,
                        target_col,
                        click_x_col,
                        click_y_col,
                        data_coords[0],
                        data_coords[1],
                        click_target_value,
                    )
                    st.session_state.manual_points.append(new_point)
                    st.session_state.last_plot_click = click_signature
                    st.rerun()

    st.subheader("Preview")
    st.table(active_df.tail(20))
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(active_df):,}")
    c2.metric("Columns", f"{active_df.shape[1]:,}")
    c3.metric("Manual points", f"{len(st.session_state.manual_points):,}")

with st.spinner("Training model and calculating validation results..."):
    try:
        result = train_model(
            active_df,
            target_col,
            task,
            algorithm,
            params,
            scale_features,
            tune_hyperparameters,
            cv_folds,
            test_size,
        )
        train_error = None
    except Exception as exc:
        result = None
        train_error = str(exc)

comparison_df = pd.DataFrame()
if result is not None:
    try:
        comparison_df = compare_models(active_df, target_col, task, algorithm_options[task], params, scale_features, cv_folds)
    except Exception as exc:
        st.warning(f"Model comparison could not be completed: {exc}")

with metrics_tab:
    st.subheader("Model Metrics")
    if train_error:
        st.error(train_error)
    else:
        if task == "Regression":
            metrics = regression_metrics(result.y_test, result.y_pred)
            m1, m2, m3 = st.columns(3)
            m1.metric("R2", f"{metrics['R2']:.4f}")
            m2.metric("RMSE", f"{metrics['RMSE']:.4f}")
            m3.metric("MAE", f"{metrics['MAE']:.4f}")
        else:
            metrics = classification_metrics(result.y_test, result.y_pred, result.y_score)
            cols = st.columns(len(metrics))
            for col, (name, value) in zip(cols, metrics.items()):
                col.metric(name, "N/A" if pd.isna(value) else f"{value:.4f}")
            report = classification_report(result.y_test, result.y_pred, target_names=result.target_names, zero_division=0)
            st.code(report, language="text")

        st.subheader("Cross-validation")
        cv_rows = []
        for key, value in result.cv_scores.items():
            if key.startswith("test_"):
                cv_rows.append({"Metric": key.replace("test_", ""), "Mean": np.mean(value), "Std": np.std(value)})
        st.table(pd.DataFrame(cv_rows))

        if result.best_params:
            st.subheader("Best GridSearchCV Parameters")
            st.json(result.best_params)

        st.subheader("Exports")
        model_buffer = io.BytesIO()
        joblib.dump(result.model, model_buffer)
        model_buffer.seek(0)
        st.download_button(
            "Export model (.joblib)",
            data=model_buffer,
            file_name=f"{algorithm.lower().replace(' ', '_')}_model.joblib",
            mime="application/octet-stream",
        )

        report_text = model_report_text(task, algorithm, result, metrics, comparison_df)
        st.download_button(
            "Export report (.txt)",
            data=report_text,
            file_name="ml_studio_report.txt",
            mime="text/plain",
        )

with graphs_tab:
    st.subheader("Visualizations")
    if train_error:
        st.error(train_error)
    else:
        g1, g2 = st.columns(2, gap="large")
        with g1:
            if task == "Regression":
                feature = st.selectbox("Regression plot feature", result.X_test.columns.tolist())
                st.pyplot(plot_regression_line(result, feature, target_col), clear_figure=True)
            else:
                st.pyplot(plot_confusion(result), clear_figure=True)
        with g2:
            if task == "Classification":
                st.pyplot(plot_roc(result), clear_figure=True)
            else:
                residual_fig, residual_ax = plt.subplots(figsize=(7, 5.2))
                residual_ax.scatter(result.y_pred, result.y_test - result.y_pred, alpha=0.75)
                residual_ax.axhline(0, color="#d62728", linewidth=2)
                residual_ax.set_xlabel("Predicted")
                residual_ax.set_ylabel("Residual")
                residual_ax.set_title("Residual Plot")
                residual_ax.grid(alpha=0.25)
                st.pyplot(residual_fig, clear_figure=True)

        pca_col, comparison_col = st.columns(2, gap="large")
        with pca_col:
            st.pyplot(plot_pca(active_df.dropna(subset=[target_col]), target_col, task), clear_figure=True)
        with comparison_col:
            if not comparison_df.empty:
                st.pyplot(plot_comparison(comparison_df, task), clear_figure=True)
                st.table(comparison_df)

        st.subheader("Interactive Training Animation")
        animation_features = [col for col in result.X_train.columns.tolist() if col != target_col]
        if len(animation_features) < 2:
            st.info("Animation needs at least two numeric feature columns.")
        else:
            a1, a2, a3, a4 = st.columns([0.26, 0.26, 0.22, 0.26])
            with a1:
                animation_x = st.selectbox("Animation X", animation_features, index=0)
            with a2:
                animation_y = st.selectbox("Animation Y", animation_features, index=1 if len(animation_features) > 1 else 0)
            with a3:
                animation_frames = st.slider("Frames", 4, 20, 10)
            with a4:
                run_animation = st.button("Run animation", use_container_width=True)

            animation_slot = st.empty()
            if run_animation:
                progress = st.progress(0)
                for frame_index, fraction in enumerate(np.linspace(0.18, 1.0, animation_frames), start=1):
                    frame_fig = plot_training_frame(
                        active_df,
                        target_col,
                        task,
                        algorithm,
                        params,
                        scale_features,
                        animation_x,
                        animation_y,
                        float(fraction),
                    )
                    animation_slot.pyplot(frame_fig, clear_figure=True)
                    progress.progress(frame_index / animation_frames)
                    time.sleep(0.18)
                progress.empty()
            else:
                animation_slot.pyplot(
                    plot_training_frame(
                        active_df,
                        target_col,
                        task,
                        algorithm,
                        params,
                        scale_features,
                        animation_x,
                        animation_y,
                        1.0,
                    ),
                    clear_figure=True,
                )

with playground_tab:
    st.subheader("Interactive Algorithm Lab")
    st.markdown(
        '<div class="small-muted">A canvas-style sandbox inspired by the TypeScript lab: add points, generate noisy data, switch algorithms, and inspect live results.</div>',
        unsafe_allow_html=True,
    )

    lab_controls, lab_canvas, lab_side = st.columns([0.24, 0.52, 0.24], gap="large")

    with lab_controls:
        st.markdown("#### Controls")
        lab_algorithm = st.selectbox(
            "Algorithm",
            ["Linear Regression", "Multilinear Regression", "K-Means", "KNN", "DBSCAN", "Decision Tree"],
            key="lab_algorithm",
        )

        lab_params = {}
        if lab_algorithm == "K-Means":
            lab_params["k"] = st.slider("Clusters", 2, 8, 3)
        elif lab_algorithm == "KNN":
            lab_params["neighbors"] = st.slider("Neighbors", 1, 25, 5, 2)
        elif lab_algorithm == "DBSCAN":
            lab_params["eps"] = st.slider("Epsilon", 2.0, 30.0, 10.0, 0.5)
            lab_params["min_samples"] = st.slider("Min samples", 2, 12, 4)
        elif lab_algorithm == "Decision Tree":
            limit_lab_depth = st.checkbox("Limit depth", value=True)
            lab_params["max_depth"] = st.slider("Max depth", 1, 10, 4) if limit_lab_depth else None

        random_count = st.number_input("Add random points", min_value=1, max_value=200, value=20, step=1)
        noise_level = st.slider("Noise", 0, 100, 12)
        if st.button("Add random", use_container_width=True):
            generated = generate_lab_points(random_count, noise_level, lab_algorithm)
            st.session_state.lab_points = pd.concat([st.session_state.lab_points, generated], ignore_index=True)
            st.session_state.last_lab_click = None
            st.rerun()

        click_mode = st.radio("Click mode", ["Add point", "Remove nearest"], horizontal=False)
        next_label = st.radio("Class for next click", ["A", "B"], horizontal=True, disabled=lab_algorithm in {"K-Means", "DBSCAN"})

        if st.button("Clear all data", use_container_width=True):
            st.session_state.lab_points = empty_lab_points()
            st.session_state.last_lab_click = None
            st.rerun()

    with lab_canvas:
        if streamlit_image_coordinates is None:
            st.error("Install streamlit-image-coordinates from requirements.txt to enable the interactive canvas.")
        else:
            canvas_image, canvas_meta = plot_lab_click_canvas(st.session_state.lab_points, lab_algorithm)
            lab_click = streamlit_image_coordinates(canvas_image, key=f"lab_canvas_{lab_algorithm}", width=760)
            if lab_click is not None:
                data_coords = image_click_to_data(lab_click, canvas_meta)
                lab_signature = (lab_algorithm, click_mode, round(float(lab_click["x"]), 2), round(float(lab_click["y"]), 2), next_label)
                if data_coords is not None and st.session_state.last_lab_click != lab_signature:
                    if click_mode == "Add point":
                        label = "A" if lab_algorithm in {"K-Means", "DBSCAN"} else next_label
                        st.session_state.lab_points = add_lab_point(st.session_state.lab_points, data_coords[0], data_coords[1], label)
                    else:
                        st.session_state.lab_points = remove_nearest_lab_point(st.session_state.lab_points, data_coords[0], data_coords[1])
                    st.session_state.last_lab_click = lab_signature
                    st.rerun()

            st.caption("Click to add points. Pick class B from the control panel before clicking. Switch to remove mode and click near a point to delete it.")

        result_fig, lab_metrics = plot_lab_algorithm(st.session_state.lab_points, lab_algorithm, lab_params)
        st.pyplot(result_fig, clear_figure=True)

    with lab_side:
        st.markdown("#### How it works")
        st.markdown(f'<div class="lab-panel">{LAB_EXPLANATIONS[lab_algorithm]}</div>', unsafe_allow_html=True)
        st.markdown("#### Key Metrics")
        st.metric("Points", len(st.session_state.lab_points))
        for metric_name, metric_value in lab_metrics.items():
            st.metric(metric_name, metric_value)

        if not st.session_state.lab_points.empty:
            st.markdown("#### Latest Points")
            st.table(st.session_state.lab_points.tail(8).round(3))
