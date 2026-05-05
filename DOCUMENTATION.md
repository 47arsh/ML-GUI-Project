# Machine Learning GUI Studio Documentation

## 1. Project Overview

Machine Learning GUI Studio is a Streamlit application designed to provide an interactive graphical interface for training, evaluating, visualizing, comparing, and exporting machine learning models. The project is built using pure Python with scikit-learn for machine learning and matplotlib for visual analysis.

The application supports both classification and regression workflows. Users can select algorithms, adjust model parameters, choose built-in datasets, upload custom CSV datasets, manually enter data points, add data points by clicking on a plot, compare multiple algorithms, and export trained models and text reports.

## 2. Technologies Used

- Frontend GUI: Streamlit
- Machine learning backend: scikit-learn
- Data processing: pandas, numpy
- Visualization: matplotlib
- Model export: joblib
- Interactive click plotting: streamlit-image-coordinates

## 3. Supported Algorithms

### Linear Regression

Linear Regression is used for regression problems where the target value is continuous. It attempts to fit a straight-line relationship between input features and the output variable.

Use case in this GUI:
- Available for regression tasks.
- Displays a scatter plot with a model prediction line.
- Reports R2, RMSE, MAE, and cross-validation scores.

### Logistic Regression

Logistic Regression is used for classification problems. It estimates class probabilities and creates linear decision boundaries between classes.

Use case in this GUI:
- Available for classification tasks.
- Supports regularization parameter adjustment through `C`.
- Provides accuracy, weighted F1 score, confusion matrix, ROC curve, and classification report.

### Support Vector Machine

SVM can be used for both classification and regression. It tries to find a decision boundary or regression function with a strong margin between samples.

Use case in this GUI:
- Available for classification and regression.
- User can adjust `C`, kernel type, and gamma.
- Supports GridSearchCV tuning across common SVM parameter combinations.

### K-Nearest Neighbors

KNN predicts based on the closest training examples in feature space. It is simple, visual, and sensitive to feature scaling.

Use case in this GUI:
- Available for classification and regression.
- User can adjust number of neighbors and weighting strategy.
- Works especially well with the interactive Playground for showing local decision regions.

### Decision Tree

Decision Tree models split data into rule-based branches. They are useful for nonlinear relationships and easy-to-understand decision logic.

Use case in this GUI:
- Available for classification and regression.
- User can limit tree depth and adjust minimum samples required for a split.
- Playground shows rectangular decision regions for classification.

### Random Forest

Random Forest combines many decision trees to create a more stable and accurate model. It usually performs well on tabular datasets.

Use case in this GUI:
- Available for classification and regression.
- User can adjust number of trees and max depth.
- Included in model comparison for performance benchmarking.

### Additional Playground Algorithms

The Playground tab also includes:

- Multilinear Regression
- K-Means clustering
- DBSCAN clustering

These are included as visual learning tools to demonstrate regression surfaces, centroid-based clustering, density-based clustering, and decision boundaries.

## 4. GUI Features

### Sidebar Configuration

The sidebar controls the main machine learning workflow.

Users can:

- Select task type: Classification or Regression
- Select algorithm
- Adjust algorithm-specific parameters
- Enable or disable feature scaling
- Enable or disable GridSearchCV hyperparameter tuning
- Select number of cross-validation folds
- Adjust train-test split size
- Run or refresh the selected algorithm

### Dataset Tab

The Dataset tab handles data input and inspection.

Users can:

- Use built-in datasets
- Upload custom CSV files
- Select the target column
- Manually add data points using coordinate/value inputs
- Add data points by clicking on a scatter plot
- Clear manually added points
- Preview the active dataset

Built-in classification datasets:

- Iris
- Breast Cancer
- Wine

Built-in regression dataset:

- Diabetes

### Metrics Tab

The Metrics tab presents model evaluation results.

For classification models, the app displays:

- Accuracy
- Weighted F1 score
- ROC AUC when available
- Classification report
- Cross-validation scores
- Best GridSearchCV parameters when tuning is enabled

For regression models, the app displays:

- R2 score
- RMSE
- MAE
- Cross-validation scores
- Best GridSearchCV parameters when tuning is enabled

The Metrics tab also provides export buttons for:

- Trained model as `.joblib`
- Text report as `.txt`

### Graphs Tab

The Graphs tab provides visual analysis.

Classification visualizations:

- Confusion matrix
- ROC curve
- PCA visualization
- Model comparison graph
- Interactive training animation

Regression visualizations:

- Scatter plot with regression prediction line
- Residual plot
- PCA visualization
- Model comparison graph
- Interactive training animation

### Playground Tab

The Playground tab is an interactive visual learning environment.

Users can:

- Select a playground algorithm
- Click on the canvas to add points
- Remove nearest points
- Assign class A or class B
- Add random points
- Adjust noise
- Clear all data
- View live algorithm visualizations
- View key metrics
- Read short algorithm explanations

Supported Playground algorithms:

- Linear Regression
- Multilinear Regression
- K-Means
- KNN
- DBSCAN
- Decision Tree

This tab is useful for demonstrating how algorithms behave visually on simple 2D data.

## 5. Adjustable Parameters

The GUI includes relevant model parameters for practical experimentation.

### Logistic Regression

- `C`: controls regularization strength

### SVM

- `C`: regularization parameter
- `kernel`: rbf, linear, poly, sigmoid
- `gamma`: scale or auto

### KNN

- `n_neighbors`: number of nearest neighbors
- `weights`: uniform or distance

### Decision Tree

- `max_depth`: optional tree depth limit
- `min_samples_split`: minimum samples required to split a node

### Random Forest

- `n_estimators`: number of trees
- `max_depth`: optional tree depth limit
- `min_samples_split`: minimum samples required to split a node

### General Parameters

- Feature scaling
- Hyperparameter tuning using GridSearchCV
- Cross-validation folds
- Test size

## 6. Model Enhancements

The project includes several enhancements beyond basic model training.

### Feature Scaling

Feature scaling is implemented using `StandardScaler`. This is especially useful for algorithms such as SVM, KNN, Logistic Regression, and SVR because these models are sensitive to feature magnitude.

### Missing Value Handling

The app uses `SimpleImputer` with median strategy to handle missing numeric values before training.

### Hyperparameter Tuning

GridSearchCV is available through the sidebar. When enabled, the app searches through predefined parameter grids and selects the best model based on:

- Accuracy for classification
- R2 score for regression

### Cross-Validation

The app uses cross-validation to provide more reliable performance estimates. Users can select between 3 and 10 folds.

### Model Comparison

The app compares multiple algorithms for the selected task type and displays a comparison graph.

Classification comparison metrics:

- Accuracy
- Weighted F1 score

Regression comparison metrics:

- R2
- RMSE

## 7. Graphical Representations

The project includes multiple graphical outputs to make model behavior understandable.

### Scatter Plot With Regression Line

Shows actual regression test points and the model prediction line for a selected feature.

### Confusion Matrix

Displays correct and incorrect classification predictions for each class.

### ROC Curve

Shows classifier discrimination ability. For multiclass datasets, the app plots one-vs-rest ROC curves.

### PCA Visualization

Projects high-dimensional data into two principal components, allowing the user to visually inspect class or target separation.

### Residual Plot

Shows regression prediction errors by plotting residuals against predicted values.

### Model Comparison Graph

Displays algorithm performance side by side, making it easier to identify which model performs best for the current dataset.

### Training Animation

Shows how the selected model changes as it trains on increasing portions of the dataset. For classification, the animation displays changing decision regions. For regression, it displays a prediction surface.

## 8. Usage Instructions

### Step 1: Install Dependencies

```powershell
cd F:\ml-gui-project
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 2: Run the Application

```powershell
streamlit run app.py
```

### Step 3: Select Task Type

Use the sidebar to select:

- Classification
- Regression

The available algorithms update automatically based on this choice.

### Step 4: Select Dataset

In the Dataset tab:

- Choose a built-in dataset, or
- Upload a CSV file

Then select the target column.

For custom CSV files, numeric columns are used as features. The selected target column is used as the prediction target.

### Step 5: Adjust Parameters

Use the sidebar controls to adjust algorithm-specific parameters.

Optional enhancements:

- Enable feature scaling
- Enable GridSearchCV
- Adjust cross-validation folds
- Adjust test size

### Step 6: Add Manual Data Points

Users can manually add rows in two ways:

- Coordinate/value input form
- Click-based scatter plot

Click-based point addition allows the user to visually place new data points and immediately retrain the model.

### Step 7: View Results

Use the Metrics and Graphs tabs to inspect:

- Scores
- Reports
- Confusion matrix
- ROC curve
- PCA plot
- Regression plot
- Residual plot
- Model comparison
- Training animation

### Step 8: Export

In the Metrics tab:

- Click `Export model (.joblib)` to download the trained model
- Click `Export report (.txt)` to download a report

## 9. How the Project Meets Evaluation Criteria

### Quality and Usability of GUI

The GUI is organized using tabs, sidebar controls, tables, plots, and clearly separated workflows. The user can move from dataset selection to training, evaluation, visualization, and export without using command-line code.

### Number and Relevance of Parameters

The app exposes relevant parameters for each algorithm, including regularization, kernel type, gamma, neighbors, weights, tree depth, number of estimators, cross-validation folds, scaling, tuning, and test size.

### Effectiveness of Graphical Representation

The app includes regression plots, residual plots, confusion matrices, ROC curves, PCA visualizations, model comparison graphs, and animated model behavior. These visualizations help users understand both performance and model behavior.

### Additional Enhancements

Enhancements include feature scaling, missing value imputation, GridSearchCV, cross-validation, model comparison, interactive point plotting, training animation, model export, and report export.

### Documentation Clarity

This documentation explains supported models, GUI features, adjustable parameters, visualizations, enhancements, usage instructions, and how the project satisfies evaluation criteria.

## 10. Limitations and Future Improvements

Current limitations:

- Uploaded datasets should contain numeric feature columns.
- Deep learning models are not included.
- Local image upload is not required because this is a machine learning GUI project, not an image-processing system.

Possible future improvements:

- Add more algorithms such as Naive Bayes, Gradient Boosting, XGBoost, and Neural Networks.
- Add automatic feature encoding for categorical columns.
- Add saved experiment history.
- Add downloadable PDF reports.
- Add more advanced interactive plots.

## 11. Conclusion

Machine Learning GUI Studio provides a complete interactive environment for experimenting with machine learning algorithms. It supports model selection, parameter tuning, dataset upload, manual data entry, visual analysis, model comparison, performance enhancements, and export functionality. The project demonstrates both practical machine learning workflow design and a usable graphical interface suitable for academic evaluation.
