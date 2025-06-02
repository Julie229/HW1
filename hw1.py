## import packages
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(layout="wide")
st.title("AutoML Application for Tabular Data")
st.markdown("""
With this application, you can upload a tabular dataset and automatically run three machine learning classification models. Please ensure your data is in a CSV file.  
You can view the instructions below in the instructions tab, preview the data and the changes that occur with the preprocessing steps with the 'Preview Data' tab, 
            and view the target class distribution to see how the if the classes are balanced or not with the 'Target Class Distribution' tab. The last two tabs become
            available when data is uploaded and a target variable is chosen.
            """)

## instructions - can be expanded and minimized
with st.expander("Instructions", expanded = True):
    st.markdown("""
    ## Steps:
    1. **Upload your dataset**: Start by uploading your tabular CSV dataset in the left sidebar.
    2. **Select target variable**: Select the target variable for classification from the dropdown menu in the left sidebar.
    3. **Handling missing values**: If there are missing values present, please choose a method of handling missing values in the left sidebar:
        - **Drop rows**: Removes any rows that contain null values.
        - **Imputation**: Fills in missing values using the median for numeric columns and the mode for categorical columns.
        - Note: The option will not appear if there are no missing values in your dataset.
    4. **Remove features (Optional)**: You can choose columns to remove from the dataset before training the models from the multiselect dropdown menu in the left sidebar.
                 If no columns are selected, the models will use all features. 
    5. **Subset data (Optional)**: You can choose to subset the data to 10%, 25%, 50%, or 75%. By default, the models will use 100% of the data. 
    6. **Run the models**: Click on the Run Models button to train and evaluate the three models - Logistic Regression, Random Forest, and Gradient Boosting.
    7. **View results**:The results will display once the models finish running. The following information will be displayed:
        - **Left side**:
                - Overall accuracy for each model.
                - Classification report which includes the precision, recall, and f1-scores for each model.
                - Confusion matric which shows the number of entries that were correctly and incorrectly predicted for each model.
        - **Right side:**
                - Accuracy table summary showing in order how each model did in terms of accuracy. 
                - The top 5 features for each model.  

    **Note:**
    - Accuracies are color-coded as followed:
        - Green: >90% accuracy
        - Yellow-green: 80-90% accuracy
        - Orange: 60-80% accuracy
        - Red: <60% accuracy
                """)

## use a sidebar for inputs and options for preprocessing
with st.sidebar:
    st.header("Upload Data and Select Preprocessing Options")
    ## upload data file widget (csv files only)
    upload_data = st.file_uploader("Upload your dataset (CSV)", type="csv")

## load data when user uploads data and lets user choose the target column 
if upload_data:
    df = pd.read_csv(upload_data)
    ## in sidebar, user selects target variable from available columns in data
    with st.sidebar:
        target = st.selectbox("Select the target column", ["-- Select Target Column --"] + list(df.columns))
        ## defaults to false
        target_selected = False 

        ## if user doesn't select column, show warning
        if target == "-- Select Target Column --":
            st.warning("Please select a valid target column")
            target_selected = False

        else:
            ## user selects column, target_selected is true
            target_selected = True

            ## if target column is selected
            if target_selected:
                ## if there are more than 20 unique values in the target column, more than likely it is not the target variable 
                if df[target].nunique() > 20:
                    st.error("Target appears to be continuous or contains over 20 unique values. This dashboard supports classification tasks only.")
                    target_selected = False
                
                ## if target column contains only 1 unique value, then not valid
                elif df[target].nunique() == 1:
                    st.warning("Target column has only one unique value. Please choose target variable with at least two unique values for the target variable")
                    target_selected = False
                
                ## show variable the user selected as confirmation
                st.write(f"Target variable selected is {target}") 
            

    ## check NAs
    has_na = df.isnull().values.any()
    ## default to false
    nas_handled = False

    ## if nulls present in data
    if has_na:
        ## in sidebar, show options to handle nulls
        with st.sidebar:
            ## description/instructions
            st.write("There are missing values present in the data. Please choose a method of handling the null values.")
            ## buttons to choose, defaults to drop rows
            strategy = st.radio("Choose how to handle missingg values:", ["Drop rows", "Imputation"], index=0)
            ## show method as confirmation
            st.write(f"Method of handling missing values selected is {strategy}")

            ## apply method
            if strategy == "Drop rows":
                df_clean = df.dropna()
                nas_handled = True
            elif strategy == "Imputation":
                ## impute missing numerical values with median
                num_imputer = SimpleImputer(strategy='median')
                numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
                df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

                ## impute missing categorical values with mode
                cat_imputer = SimpleImputer(strategy='most_frequent')
                categorical_cols = df.select_dtypes(include=['object']).columns
                df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

                ## change to df_clean for consistency
                df_clean = df
                nas_handled = True

    ## if not nulls, copy df to df_clean for consistency and nas_handled is true
    else:
        df_clean = df.copy()
        nas_handled = True
    
    ## if target is selected, let user have option to remove features and subset data
    if target_selected:
        ## in sidebar, use multiselect dropdown to remove features
        with st.sidebar:
            remove_features = st.multiselect("Optional: Select features to remove", 
                                             ## options are columns that aren't target
                                            options = df_clean.drop(columns=target).columns,
                                            ## default is no features removed
                                            default = [])
            ## remove features that user chooses 
            df_clean = df_clean.drop(columns=remove_features)

        ## since categorical variables need to be onehot encoded, if there are a lot of unique values this can create too many columns
        for col in df.select_dtypes(include='object'):
            ## if a categorical column a lot features, show warning that user might want to remove it
            if df[col].nunique() > 50:
                if col not in remove_features:
                    st.warning(f"The categorical column'{col}' has {df[col].nunique()} unique values — consider removing this feature before running the model.")
    
        ## create option in sidebar to subset data
        with st.sidebar:
            st.subheader("Optional: Subset the data")
            ## give multiple options to choose one of 
            subset_data = st.selectbox("Select a percentage of data to use",
                                    options=[.1, .25, .5, .75, 1.0],
                                    ## default to using all of the data
                                    index=4,
                                    ## show options as percentages for easier interpretability
                                    format_func = lambda x: f"{int(x*100)}%")

            ## subset data if user chooses
            if subset_data < 1.0:
                ## stratify incase of imbalanced classes
                df_clean, _ = train_test_split(df_clean, train_size=subset_data, stratify=df_clean[target], random_state=42)

    ## in main panel, let user preview data and see shape of data
    with st.expander("Preview Data"):
        st.markdown(f"There are {df_clean.shape[0]} rows and {df_clean.shape[1]} columns")
        st.dataframe(df_clean)

    ## in main panel, let user see target class distribution
    if target_selected:
        with st.expander("Target Class Distribution"):
            ## count frequency of classes
            class_counts = df_clean[target].value_counts().sort_index()
            ## initalize plot, plot graph, and label axes and title
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.bar(class_counts.index, class_counts.values, edgecolor='black')
            ax.set_title(f'Distribution of Target Variable ({target})', fontsize = 8)
            ax.set_xlabel(f'Target Classes ({target})', fontsize = 8)
            ax.set_ylabel('Frequency', fontsize = 8)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            st.pyplot(fig)


## run models button
run_models = False
## show button once data is uploaded and requirements are met
if upload_data and target_selected and nas_handled:
    run_models = st.button("Run Models")
## if requirements not met, show warning
else: 
    st.warning("Please upload data, select a target variable, and handle missing values (if needed)")

## if user clicks run models button, then run the models
if run_models:
    ## split into features and target
    X = df_clean.drop(columns=target)
    y = df_clean[target]
    
    ## categorical columns 
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()   

    ## onehot encode categorical columns 
    for col in categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded = encoder.fit_transform(X[[col]]) 
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([col]), index=X.index)
        X = pd.concat([X.drop(columns=[col]), encoded_df], axis=1)

    ## standardize numeric features, mainly for logistic regression
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    ## split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    ## initialize the three models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    ## split main panel into left and right columns 
    left, right = st.columns([2,1])
    ## initalize top features dictonary for each model
    model_top_features = {}
    ## initalize summary list for accuracy summary table
    summary = []

    ## define accuracy colors when displaying accuracies
    def get_accuracy_color(acc):
        if acc > 0.9:
            return "green"
        elif acc > 0.8:
            return "yellowgreen"
        elif acc > 0.6:
            return "orange"
        else:
            return "red"

    ## train models and display results in left panel
    with left:
        ## for each model
        for name, model in models.items():
            ## display model name and spinner in case training takes long time while fitting model 
            st.header(name)
            with st.spinner(f"Training {name}..."):
                model.fit(X_train, y_train)

            ## predictions from model from test set
            y_pred = model.predict(X_test)

            ## accuracy score
            acc = accuracy_score(y_test, y_pred)
            ## display accuracy, getting color from function defined previously
            st.subheader("Accuracy")
            color = get_accuracy_color(acc)
            ## display metric in the color defined, with a bigger font size, rounded to 2 decimal points
            st.markdown(f"<h3 style='color:{color}; font-size:28px'>{acc:.2%}</h3>", unsafe_allow_html=True)

            ## append accuracy to summary table
            summary.append({"Model": name, "Accuracy": acc})

            ## if logistic regression, use coefficients for model importance
            if hasattr(model, "coef_"):
                importances = model.coef_[0].abs()
                ## create df and sort
                feature_importance = pd.DataFrame({'Feature': X.columns, 'Coefficient': importances}).sort_values('Coefficient', ascending=False) 
                ## get only top 5 features
                top_features = feature_importance.head(5)
                ## add to dictionary
                model_top_features[name] = top_features

            ## if random forest or gradient boosting, using feature importances directly
            elif hasattr(model, "feature_importances_"):
                ## get feature importances 
                importances = model.feature_importances_
                ## create df, sort, get top 5, and add to dictionary
                feature_importance = pd.DataFrame({'Feature': X.columns, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
                top_features = feature_importance.head(5)
                model_top_features[name] = top_features

            ## get classification report and turn into df
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_dict).transpose()

            ## display classification report 
            st.subheader("Classification Report")
            st.dataframe(report_df.style.format(precision=2))

            ## plot confusion matrix to see how model performs visually
            st.subheader("Confusion Matrix")
            fig, ax = plt.subplots()
            disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax)
            st.pyplot(fig)

    ## define accuracy color function again, but for formatting df
    def table_accuracy_color(acc):
        if acc > 0.9:
            return "color: green"
        elif acc > 0.8:
            return "color: yellowgreen"
        elif acc > 0.6:
            return "color: orange"
        else:
            return "color: red"
    
    ## in the right panel
    with right:
        ## display the accuracy summary table 
        st.subheader("Model Accuracy Summary Ranking")
        ## sort descending and format by colors defined in previous function
        summary_df = pd.DataFrame(summary).sort_values(by = "Accuracy", ascending=False)
        styled_summary = summary_df.style.format({"Accuracy": "{:.2%}"}).applymap(table_accuracy_color, subset=["Accuracy"])
        st.dataframe(styled_summary)

        ## display top 5 features for each model
        for name, features in model_top_features.items():
            st.subheader(f"Top 5 Features for {name}")
            st.dataframe(features)


## Sources:
# https://docs.streamlit.io/get-started
# https://www.geeksforgeeks.org/understanding-feature-importance-in-logistic-regression-models/#1-coefficient-magnitude
# https://www.geeksforgeeks.org/feature-importance-with-random-forests/
# https://www.w3schools.com/python/python_ml_confusion_matrix.asp
