import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

from regressors import stats



### Basic functions to ease the EDA process ###

def detect_and_delete_duplicated(df, only_info=False):
    """
    Simple function to show and delete duplicate entries in a DataFrame

    Parameters:
    - df: The DataFrame to erase duplicate entries.
    - only_info: If you only want to know how many duplicates are.

    Return:
    - df: The DataFrame cleaned.
    """
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicate_count}")
    return df.drop_duplicates() if not only_info else None
    

def split_numerical_categorical_cols(df):
    """
    Simple function to get numerical and categorical cols of a dataframe splitted.

    Parameters:
    - df: The dataframe to analyze.

    Return:
    - Numerical_cols: Only the numerical cols of the initial dataframe.
    - Categorical_cols: Only the categorical cols of the initial dataframe.
    """

    numerical_cols = [colname for colname in df.columns if df[colname].dtype in ['int64', 'float64']]
    categorical_cols = [colname for colname in df.columns if df[colname].dtype in ['object']]
    return numerical_cols, categorical_cols


def detect_outliers(df, column, quant=0.25, add=1.5):
    """
    Function to find outlayers in a select column, with a modifiable quantile and addition.

    Parameters:
    - df: The DataFrame to analyze.
    - column: Column name to analyze.
    - quant (opt): set quantile.
    - add (opt): set addition to the superior and inferior limits calculation.

    Return:
    - Text indicating how many outlayers, in which column and their indexes.
    """

    # Calculate both quantiles, IQR
    Q3 = df[column].quantile(1-quant)
    Q1 = df[column].quantile(0+quant)
    IQR = Q3 - Q1

    # Calculate superior and inferior limits
    superior = Q3 + (add * IQR)
    inferior = Q1 - (add * IQR)
    out_sup = df[df[column] > superior].index
    out_inf = df[df[column] < inferior].index

    # Get outlayers and return the information
    outliers = list(set(out_sup).union(set(out_inf)))
    return (f'There are {len(outliers)} outlayers in the column {column} with the next indexs: {outliers}')


def get_columns_info(df):
    """
    Function to show initial information about the columns in the dataframe.
    You will get the shape, total nulls and total nulls per column,
    and a summary of numerical and categorical vars separately.

    Parameters:
    - df: The dataframe to analyze.
    """

    # Show initial and basic information
    print('Shape -> ', df.shape)
    print('Null values (number) per column -> ', df.isnull().sum())
    print('Null values per column -> ', df.isnull().any())

    # Get numerical and categorical columns, and create two separate dataframes
    numerical_cols, categorical_cols = split_numerical_categorical_cols(df)

    numerical_df = df[numerical_cols]
    numerical_summary = pd.DataFrame({
        'DataType': numerical_df.dtypes,
        'Min': numerical_df.min(),
        'Max': numerical_df.max(),
    })

    categorical_df = df[categorical_cols]
    categorical_summary = pd.DataFrame({
        'DataType': categorical_df.dtypes,
        'Unique': categorical_df.apply(lambda col: col.unique())
    })

    # Print the information and returns
    print("\nNumerical Summary:", numerical_summary)
    print("\nCategorical Summary:", categorical_summary)

    print("\nOutliers:\n")
    for col in numerical_cols: print(detect_outliers(df, col))



### Ploting functions to ease the EDA process ###

def plot_numerical_distributions(df, cols_per_row=4):
    """
    Function to plot histograms with boxplots for all numerical columns in the dataframe.

    Parameters:
    - df: The DataFrame to analyze.
    - cols_per_row: Number of columns (histogram + boxplot pairs) per row.
    """

    # Identify numerical columns in the dataframe and calculate rows number
    numerical_cols, _ = split_numerical_categorical_cols(df)
    num_cols = len(numerical_cols)
    rows = (num_cols + cols_per_row - 1) // cols_per_row

    # Set up the figure and flatten axes for easy indexing
    fig, axes = plt.subplots(rows * 2, cols_per_row, figsize=(14, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(numerical_cols):
        # Plot histogram with KDE
        sns.histplot(df[col], kde=True, ax=axes[2*i], color='#1D5F8A')
        axes[2*i].set_title(f'Distribution of {col}')
        axes[2*i].set_xlabel(col)
        axes[2*i].set_ylabel('Frequency')

        # Plot boxplot
        sns.boxplot(x=df[col], ax=axes[2*i + 1], color='#FFD43B')
        axes[2*i + 1].set_title(f'Boxplot of {col}')
        axes[2*i + 1].set_xlabel(col)

    # Hide any unused subplots and final show
    for j in range(2 * num_cols, len(axes)): axes[j].axis('off')
    plt.tight_layout()
    plt.show()


def get_predictions_and_initial_score(df, x_cols, y_col):
    """
    Function to get predictions and initial score of a model, used to evaluate.

    Parameters:
    - df: The DataFrame to analyze.
    - x_cols: Columns used to make predictions
    - y_col: Column to be predicted
    """
    x = df[x_cols].values
    y = df[y_col].values

    x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=42, test_size=0.2)
    sc_x = StandardScaler().fit(x)
    sc_y = StandardScaler().fit(y)

    x_train = sc_x.transform(x_train)
    x_test = sc_x.transform(x_test)
    y_train = sc_y.transform(y_train)
    y_test = sc_y.transform(y_test)

    model = LinearRegression()
    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    model.intercept_ = model.intercept_[0]
    model.coef_ = model.coef_.reshape(-1)

    y_test = y_test.reshape(-1)

    print("\n==== Summary ====\n")
    stats.summary(model, x_test, y_test, x_cols)

    residuals = np.subtract(y_test, y_pred.reshape(-1))
    plt.scatter(y_pred, residuals)
    plt.show()

    return x_train, x_test, y_train, y_test, y_pred, model


def plot_scatter_relation_1toN_separated_with_categorical(df, main_col, categorical_col, columns, num_cols=2):
    """
    Function to plot one variable with another ones with also a categorical column.

    Parameters:
    - df: The DataFrame to analyze.
    - main_col: Column to be compared with the rest.
    - categorical_col: Categorical column to show with the comparison.
    - columns: Each column to be compared with the main var.
    - num_cols: Number of columns per row, default is 2.
    """

    num_plots = len(columns)
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 6 * num_rows))  # Adjust figsize
    axes = axes.flatten()  # Flatten axes for easier indexing

    for i, col in enumerate(columns):
        sns.scatterplot(x=df[col], y=df[main_col], hue=df[categorical_col], ax=axes[i])
        axes[i].set_title(f'{main_col} vs. {col} (colored by {categorical_col})')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Charges')
        axes[i].legend(title=categorical_col)

    # Hide any unused subplots
    for i in range(num_plots, num_rows * num_cols): axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_boxplot_relation_1toN_separated(df, main_col, columns, num_cols=2):
    """
    Function to plot one variable with another ones with also a categorical column.

    Parameters:
    - df: The DataFrame to analyze.
    - main_col: Column to be compared with the rest.
    - categorical_col: Categorical column to show with the comparison.
    - columns: Each column to be compared with the main var.
    - num_cols: Number of columns per row, default is 2.
    """

    num_plots = len(columns)
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate rows needed

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 6 * num_rows))  # Adjust figsize
    axes = axes.flatten()  # Flatten axes for easier indexing

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], y=df[main_col], ax=axes[i], palette="Set3")
        axes[i].set_title(f'{main_col} vs. {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel(main_col)

    # Hide any unused subplots
    for i in range(num_plots, num_rows * num_cols): axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


### More functions to ease the EDA process ###

def calculate_elasticity(df, indep_col, dep_col):
    """
    Function to get elasticity of a model, used to evaluate.

    Parameters:
    - df: The DataFrame to analyze.
    - indep_col: Independent column to be compared with the dependent one.
    - dep_col: Dependend column to be compared with the independent one.

    Return:
    - elasticity: The elasticity of the model.
    """
    if indep_col == dep_col:
        #print("Columns must be different")
        return None
    
    x_independent_axis = df[[indep_col]]
    y_dependent_axis = df[dep_col]

    # Fit a linear regression model
    model = LinearRegression().fit(x_independent_axis, y_dependent_axis)

    # Get the coefficient (slope) from the model
    slope = model.coef_[0]

    # Calculate the average charges and age to get percentage change
    average_x_col = np.mean(df[indep_col])
    average_y_col = np.mean(df[dep_col])

    # Elasticity = (slope * average age) / average charges
    elasticity = (slope * average_x_col) / average_y_col
    print(f"{indep_col} elasticity of {dep_col} (dependent): {elasticity:.4f}")
    return elasticity


def regression_summary(y_true, y_pred, x_train, model=None, cv_folds=5):
    """
    Generates a summary of important regression metrics.
    
    Parameters:
    - y_true: Array of actual values
    - y_pred: Array of predicted values
    - x_train: Training feature matrix (used to calculate Adjusted R-squared)
    - model: Trained regression model (optional, used for cross-validation and AIC/BIC calculation)
    - cv_folds: Number of folds for cross-validation (default is 5)
    
    Returns:
    - summary: DataFrame containing regression metrics
    """
    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Mean Squared Error (MSE) and Root Mean Squared Error (RMSE)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    # R-squared and Adjusted R-squared
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)  # number of samples
    p = x_train.shape[1]  # number of features
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    
    # Cross-Validated R-squared (if model is provided)
    if model:
        cv_r2 = cross_val_score(model, x_train, y_true, cv=cv_folds, scoring='r2').mean()
    else:
        cv_r2 = None

    # AIC and BIC (if model has attributes for calculating log-likelihood)
    if model and hasattr(model, 'score'):
        residual_sum_of_squares = np.sum((y_true - y_pred) ** 2)
        aic = n * np.log(residual_sum_of_squares / n) + 2 * (p + 1)
        bic = n * np.log(residual_sum_of_squares / n) + np.log(n) * (p + 1)
    else:
        aic, bic = None, None

    # Print each metric
    print("Regression Model Summary:")
    print(f"Prediction shape --> {y_pred.shape}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R-squared: {r2:.4f}")
    print(f"Adjusted R-squared: {adj_r2:.4f}")
    print(f"Cross-Validated R-squared: {cv_r2:.4f}") if cv_r2 is not None else print("Cross-Validated R-squared: Not available")
    print(f"AIC: {aic:.4f}") if aic is not None else print("AIC: Not available")
    print(f"BIC: {bic:.4f}") if bic is not None else print("BIC: Not available")
        
    # Summarize results in a DataFrame
    summary = pd.DataFrame({
        'Metric': ['MAE', 'MSE', 'RMSE', 'MAPE', 'R-squared', 'Adjusted R-squared', 'Cross-Validated R-squared', 'AIC', 'BIC'],
        'Value': [mae, mse, rmse, mape, r2, adj_r2, cv_r2, aic, bic]
    })
    
    return summary