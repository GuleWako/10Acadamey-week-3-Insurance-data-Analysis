import pandas as  pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind
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
