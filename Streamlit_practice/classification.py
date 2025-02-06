# # # apply random forest algorithm 
# # import streamlit as st
# # import pandas as pd
# # from sklearn.datasets import load_iris
# # from sklearn.ensemble import RandomForestClassifier
# # @st.cache_data
# # def loaddata():
# #     iris=load_iris()
# #     df=pd.DataFrame(iris.data , columns=iris.feature_names)
# #     df['species']=iris.target
# #     return df , iris.target_names
# # df , target=loaddata()

# # #model implementation
# # model=RandomForestClassifier()
# # model.fit(df.iloc[:,:-1],df['species'])

# # st.sidebar.title("Input Features")
# # sepal_length = st.sidebar.slider("Sepal length", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()))
# # sepal_width = st.sidebar.slider("Sepal width", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()))
# # petal_length = st.sidebar.slider("Petal length", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()))
# # petal_width = st.sidebar.slider("Petal width", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()))\


# # input_data=[[sepal_length,sepal_width,petal_length,petal_width]]

# # #Prediction
# # prediction=model.predict(input_data)
# # predicted_species=target[prediction[0]]

# # st.write("Predictions of Random Forest Model")
# # st.write(f"The predicted species is {predicted_species}")

# # def accuracy():
# #     y_pred = model.predict(df.iloc[:,:-1])
# #     accuracy = model.score(df.iloc[:,:-1], df['species'])
# #     return accuracy
# # accuracy=accuracy()
# # st.write(f"Accuracy of the model is {accuracy:.2f}")
# # import matplotlib.pyplot as plt
# # from sklearn.metrics import confusion_matrix
# #     # Confusion Matrix
# # st.write("Confusion Matrix of the model")
# # y_pred = model.predict(df.iloc[:,:-1])
# # cm = confusion_matrix(df['species'], y_pred)
# # plt.imshow(cm, interpolation='nearest', cmap='Blues')
# # plt.title('Confusion Matrix')
# # plt.colorbar()
# # plt.show()









# # Import necessary libraries
# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.datasets import load_iris
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# # =============================================================================
# # THEORY:
# #
# # Random Forest is an ensemble learning method that constructs multiple
# # decision trees during training and outputs the class that is the mode of the
# # classes (classification) or mean prediction (regression) of the individual trees.
# #
# # Key aspects:
# # 1. **Bagging (Bootstrap Aggregation):**
# #    - Random Forest uses bootstrapped subsets of data to train each tree.
# #    - This reduces variance and helps prevent overfitting.
# #
# # 2. **Feature Randomness:**
# #    - At each split in a tree, a random subset of features is considered.
# #    - This decorrelates trees and leads to a more robust overall model.
# #
# # 3. **Hyperparameters:**
# #    - n_estimators: Number of trees in the forest.
# #    - max_depth: Maximum depth of each tree.
# #    - min_samples_split: Minimum number of samples required to split an internal node.
# #    - oob_score: Whether to use out-of-bag samples to estimate the generalization accuracy.
# #
# # 4. **Evaluation:**
# #    - Accuracy, confusion matrix, and classification report help assess model performance.
# #
# # In this implementation, we'll explore these aspects, including model training,
# # prediction, accuracy evaluation, and feature importance visualization.
# # =============================================================================

# # -----------------------------------------------------------------------------
# # Data Loading and Preparation
# # -----------------------------------------------------------------------------
# @st.cache_data  # Caching the data loading for performance in Streamlit
# def load_data():
#     iris = load_iris()
#     # Create a DataFrame from the iris dataset
#     df = pd.DataFrame(iris.data, columns=iris.feature_names)
#     df['species'] = iris.target
#     return df, iris.target_names

# df, target_names = load_data()
# st.title("Random Forest Classification Technique")
# st.write("Dataset is all about Iris flowers with 4 features and 3 species veriscolor , setosa, virginica")
# st.write(df)
# # -----------------------------------------------------------------------------
# # Sidebar for User Input and Hyperparameter Tuning
# # -----------------------------------------------------------------------------
# st.sidebar.header("Input Features and Hyperparameters")

# # Feature input using sliders
# sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()),
#                                  float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
# sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()),
#                                 float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
# petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()),
#                                  float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
# petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()),
#                                 float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))

# # Hyperparameters for the Random Forest model
# st.sidebar.subheader("Random Forest Hyperparameters")
# n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 10, 200, 100, step=10)
# max_depth = st.sidebar.slider("Maximum depth of trees (max_depth)", 1, 20, 5)
# min_samples_split = st.sidebar.slider("Min samples to split (min_samples_split)", 2, 10, 2)
# oob_score = st.sidebar.checkbox("Use Out-of-Bag samples (oob_score)", value=True)

# input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# # -----------------------------------------------------------------------------
# # Model Implementation: Random Forest Classifier
# # -----------------------------------------------------------------------------
# # Explanation:
# # - We are initializing the RandomForestClassifier with user-defined hyperparameters.
# # - Random Forest builds several decision trees and aggregates their predictions.
# model = RandomForestClassifier(
#     n_estimators=n_estimators,
#     max_depth=max_depth,
#     min_samples_split=min_samples_split,
#     oob_score=oob_score,  # Use out-of-bag error estimate if selected
#     random_state=42  # For reproducibility
# )
# model.fit(df.iloc[:, :-1], df['species'])

# # -----------------------------------------------------------------------------
# # Prediction
# # -----------------------------------------------------------------------------
# # Predict the species for the input feature values.
# prediction = model.predict(input_data)
# predicted_species = target_names[prediction[0]]
# st.write("### Random Forest Model Prediction")
# st.write(f"**The predicted species is:** {predicted_species}")

# # -----------------------------------------------------------------------------
# # Model Evaluation: Accuracy and Classification Report
# # -----------------------------------------------------------------------------
# # Evaluate the model on the full dataset.
# y_pred = model.predict(df.iloc[:, :-1])
# accuracy = accuracy_score(df['species'], y_pred)
# st.write("### Model Accuracy")
# st.write(f"The accuracy of the model on the full dataset is: **{accuracy:.2f}**")

# # Optionally, display the Out-Of-Bag (OOB) score if used
# if oob_score:
#     st.write("### Out-of-Bag Score")
#     st.write(f"OOB Score: **{model.oob_score_:.2f}**")

# # Display the full classification report
# st.write("### Classification Report")
# report = classification_report(df['species'], y_pred, target_names=target_names, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# st.dataframe(report_df)

# # -----------------------------------------------------------------------------
# # Feature Importance Visualization
# # -----------------------------------------------------------------------------
# # Random Forests can provide an estimate of feature importance.
# st.write("### Feature Importances")
# importances = model.feature_importances_
# feature_names = df.columns[:-1]
# importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': importances
# }).sort_values(by='Importance', ascending=False)
# st.dataframe(importance_df)

# # Plot the feature importances
# fig, ax = plt.subplots()
# sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
# ax.set_title('Feature Importances from Random Forest')
# st.pyplot(fig)

# # -----------------------------------------------------------------------------
# # Confusion Matrix Visualization
# # -----------------------------------------------------------------------------
# # Compute the confusion matrix to visualize the performance of classification.
# st.write("### Confusion Matrix")
# cm = confusion_matrix(df['species'], y_pred)
# fig_cm, ax_cm = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
# ax_cm.set_xlabel('Predicted')
# ax_cm.set_ylabel('True')
# ax_cm.set_title('Confusion Matrix')
# st.pyplot(fig_cm)

# # =============================================================================
# # SUMMARY OF THE RANDOM FOREST TECHNIQUE:
# #
# # - **Ensemble Method:** Combines multiple decision trees to reduce overfitting.
# # - **Randomness in Training:** Uses bagging and random feature selection to
# #   create diverse trees.
# # - **Hyperparameter Tuning:** Adjusting parameters like n_estimators, max_depth,
# #   and min_samples_split can significantly impact performance.
# # - **Evaluation Metrics:** Accuracy, confusion matrix, and classification report
# #   help in understanding the strengths and weaknesses of the model.
# # - **Feature Importance:** Provides insights into which features most influence
# #   the prediction.
# #
# # Use this interactive app to experiment with different hyperparameters and input
# # values, and observe how they affect the model performance and evaluation metrics.
# # =============================================================================





# import streamlit as st
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.datasets import load_wine
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# @st.cache_data
# def load_data():
#     wine = load_wine()
#     df = pd.DataFrame(wine.data, columns=wine.feature_names)
#     df['target'] = wine.target
#     return df, wine.target_names

# df, target_names = load_data()
# st.title("Random Forest Classifier")
# st.write("Dataset we used is Wine Quality dataset from UCI Machine Learning Repository.")
# st.write(df)
# st.sidebar.header("Input Features and Hyperparameters")
# input_data = []
# for feature in df.columns[:-1]:
#     min_val = float(df[feature].min())
#     max_val = float(df[feature].max())
#     default_val = float(df[feature].mean())
#     value = st.sidebar.slider(feature, min_val, max_val, default_val)
#     input_data.append(value)
# st.sidebar.subheader("Random Forest Hyperparameters")
# n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 10, 200, 100, step=10)
# max_depth = st.sidebar.slider("Maximum depth of trees (max_depth)", 1, 20, 5)
# min_samples_split = st.sidebar.slider("Min samples to split (min_samples_split)", 2, 10, 2)
# oob_score = st.sidebar.checkbox("Use Out-of-Bag samples (oob_score)", value=True)
# model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, oob_score=oob_score, random_state=42)
# model.fit(df.iloc[:, :-1], df['target'])
# prediction = model.predict([input_data])
# predicted_class = target_names[prediction[0]]
# st.write("### Random Forest Model Prediction")
# st.write(f"**The predicted wine class is:** {predicted_class}")
# y_pred = model.predict(df.iloc[:, :-1])
# accuracy = accuracy_score(df['target'], y_pred)
# st.write("### Model Accuracy")
# st.write(f"The accuracy of the model on the full dataset is: **{accuracy:.2f}**")
# if oob_score:
#     st.write("### Out-of-Bag Score")
#     st.write(f"OOB Score: **{model.oob_score_:.2f}**")
# st.write("### Classification Report")
# report = classification_report(df['target'], y_pred, target_names=target_names, output_dict=True)
# report_df = pd.DataFrame(report).transpose()
# st.dataframe(report_df)
# st.write("### Feature Importances")
# importances = model.feature_importances_
# feature_names = df.columns[:-1]
# importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
# st.dataframe(importance_df)
# fig, ax = plt.subplots()
# sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
# ax.set_title('Feature Importances from Random Forest')
# st.pyplot(fig)
# st.write("### Confusion Matrix")
# cm = confusion_matrix(df['target'], y_pred)
# fig_cm, ax_cm = plt.subplots()
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
# ax_cm.set_xlabel('Predicted')
# ax_cm.set_ylabel('True')
# ax_cm.set_title('Confusion Matrix')
# st.pyplot(fig_cm)


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    return df, iris.target_names

df, target_names = load_data()
st.title("Random Forest Classification Technique")
st.write("Dataset is all about Iris flowers with 4 features and 3 species veriscolor , setosa, virginica")
st.write(df)
st.sidebar.header("Input Features and Hyperparameters")
sepal_length = st.sidebar.slider("Sepal length (cm)", float(df['sepal length (cm)'].min()), float(df['sepal length (cm)'].max()), float(df['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider("Sepal width (cm)", float(df['sepal width (cm)'].min()), float(df['sepal width (cm)'].max()), float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider("Petal length (cm)", float(df['petal length (cm)'].min()), float(df['petal length (cm)'].max()), float(df['petal length (cm)'].mean()))
petal_width = st.sidebar.slider("Petal width (cm)", float(df['petal width (cm)'].min()), float(df['petal width (cm)'].max()), float(df['petal width (cm)'].mean()))
st.sidebar.subheader("Random Forest Hyperparameters")
n_estimators = st.sidebar.slider("Number of trees (n_estimators)", 10, 200, 100, step=10)
max_depth = st.sidebar.slider("Maximum depth of trees (max_depth)", 1, 20, 5)
min_samples_split = st.sidebar.slider("Min samples to split (min_samples_split)", 2, 10, 2)
oob_score = st.sidebar.checkbox("Use Out-of-Bag samples (oob_score)", value=True)
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]
model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, oob_score=oob_score, random_state=42)
model.fit(df.iloc[:, :-1], df['species'])
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]
st.write("### Random Forest Model Prediction")
st.write(f"**The predicted species is:** {predicted_species}")
y_pred = model.predict(df.iloc[:, :-1])
accuracy = accuracy_score(df['species'], y_pred)
st.write("### Model Accuracy")
st.write(f"The accuracy of the model on the full dataset is: **{accuracy:.2f}**")
if oob_score:
    st.write("### Out-of-Bag Score")
    st.write(f"OOB Score: **{model.oob_score_:.2f}**")
st.write("### Classification Report")
report = classification_report(df['species'], y_pred, target_names=target_names, output_dict=True)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df)
st.write("### Feature Importances")
importances = model.feature_importances_
feature_names = df.columns[:-1]
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False)
st.dataframe(importance_df)
fig, ax = plt.subplots()
sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax, palette='viridis')
ax.set_title('Feature Importances from Random Forest')
st.pyplot(fig)
st.write("### Confusion Matrix")
cm = confusion_matrix(df['species'], y_pred)
fig_cm, ax_cm = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax_cm)
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('True')
ax_cm.set_title('Confusion Matrix')
st.pyplot(fig_cm)
