import pandas as  pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
import matplotlib.pyplot as plt
def load_data(filename):
    """Loads the insurance claim data from a txt file.

    Args:
        filename (str): The name of the txt file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.
    """
    insurance_data = pd.read_csv(f'docs/{filename}',delimiter='|')
    return insurance_data
def find_missing_values(df):
    """
    Finds missing values and returns a summary.

    Args:
        df: The DataFrame to check for missing values.

    Returns:
        A summary of missing values, including the number of missing values per column.
    """

    null_counts = df.isnull().sum()
    missing_value = null_counts
    percent_of_missing_value = 100 * null_counts / len(df)
    data_type=df.dtypes

    missing_data_summary = pd.concat([missing_value, percent_of_missing_value,data_type], axis=1)
    missing_data_summary_table = missing_data_summary.rename(columns={0:"Missing values", 1:"Percent of Total Values",2:"DataType" })
    missing_data_summary_table = missing_data_summary_table[missing_data_summary_table.iloc[:, 1] != 0].sort_values('Percent of Total Values', ascending=False).round(1)

    print(f"From {df.shape[1]} columns selected, there are {missing_data_summary_table.shape[0]} columns with missing values.")

    return missing_data_summary_table


def replace_missing_values(data):
  """
  Replaces missing values in a DataFrame with the mean for numeric columns and the mode for categorical columns.

  Args:
    data: The input DataFrame.

  Returns:
    The DataFrame with missing values replaced.
  """

  # Identify numeric and categorical columns
  numeric_columns = data.select_dtypes(include='number').columns
  categorical_columns = data.select_dtypes(include='object').columns

  # Replace missing values in numeric columns with the mean
  for column in numeric_columns:
    column_mean = data[column].mean()
    data[column] = data[column].fillna(column_mean)

  # Replace missing values in categorical columns with the mode
  for column in categorical_columns:
    column_mode = data[column].mode().iloc[0]
    data[column] = data[column].fillna(column_mode)

  return data

def histogramPlotForNumericalColumns(insurance_data):
    for column in insurance_data.select_dtypes(include='number').columns:
        print(insurance_data[column].value_counts())
        plt.figure(figsize=(20,6))
        plt.hist(insurance_data[column], bins=30)
        plt.title(f"Histogram of {column}")
        plt.show()

def barchartPlotForCategoricalColumns(insurance_data):
    for column in insurance_data.select_dtypes(include='object').columns:
        print(insurance_data[column].value_counts())
        insurance_data[column].value_counts().plot(kind='bar',figsize=(20,6))
        plt.title(f"Bar Chart of {column}")
        plt.show()

def get_outlier_summary(data):
    """
    Calculates outlier summary statistics for a DataFrame.

    Args:
        data : Input DataFrame.

    Returns:
        Outlier summary DataFrame.
    """

    outlier_summary = pd.DataFrame(columns=['Variable', 'Number of Outliers'])
    data = data.select_dtypes(include='number')

    for column_name in data.columns:
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]

        outlier_summary = pd.concat(
            [outlier_summary, pd.DataFrame({'Variable': [column_name], 'Number of Outliers': [outliers.shape[0]]})],
            ignore_index=True
        )
    non_zero_count = (outlier_summary['Number of Outliers'] > 0).sum()
    print(f"From {data.shape[1]} selected numerical columns, there are {non_zero_count} columns with outlier values.")

    return outlier_summary

def remove_outliers_winsorization(xdr_data):
    """
    Removes outliers from specified columns of a DataFrame using winsorization.

    Args:
        data: The input DataFrame.
        column_names (list): A list of column names to process.

    Returns:
        The DataFrame with outliers removed.
    """
    # data = xdr_data.select_dtypes(include='number')
    for column_name in xdr_data.select_dtypes(include='number').columns:
        q1 = xdr_data[column_name].quantile(0.25)
        q3 = xdr_data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        xdr_data[column_name] = xdr_data[column_name].clip(lower_bound, upper_bound)

    return xdr_data

def hypothesis_test_difference_between_columns(df, kpi_column, group_column):
  
    group_codes = df[group_column].unique()
   
    column_groups = [df[df[group_column] == group_code][kpi_column].dropna() for group_code in group_codes]

    t_stat, p_value = stats.f_oneway(*column_groups)
    print(f"T-statistic of {group_column}: {t_stat}")
    print(f"P-value of {group_column}: {p_value}")
     # Interpret the results
    alpha = 0.05  
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")



def ABhypothesisTesting(insurance_data,feature,metric1,metric2,kpi):
    group_a = insurance_data[insurance_data[feature] == metric1][kpi]
    group_b = insurance_data[insurance_data[feature] == metric2][kpi]

    # Perform an independent t-test
    t_stat, p_value = stats.ttest_ind(group_a.dropna(), group_b.dropna(),equal_var=False, nan_policy='omit')

    # Print the results
    print(f"T-statistic of {feature} values {metric1} and {metric2}: {t_stat}")
    print(f"P-value of {feature} values {metric1} and {metric2}: {p_value}")

    # Interpret the results
    alpha = 0.05  # significance level
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")


def chi_squared_test(df, categorical_column1, categorical_column2):
    """
    Performs a chi-squared test to determine if there's a significant association between two categorical variables.

    Args:
        df: The pandas DataFrame containing the data.
        categorical_column1: The first categorical column.
        categorical_column2: The second categorical column.

    Returns:
        chi2: The chi-squared test statistic.
        p_value: The p-value associated with the chi-squared test.
    """

    # Create a contingency table
    contingency_table = pd.crosstab(df[categorical_column1], df[categorical_column2])

    # Perform the chi-squared test
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    print(f"Chi-squared statistic of {categorical_column1} and {categorical_column2}:", chi2)
    print("P-value:", p_value)
    
    alpha = 0.05  # significance level
    if p_value < alpha:
        print("Reject the null hypothesis: There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")