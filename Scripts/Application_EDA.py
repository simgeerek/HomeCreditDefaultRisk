import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)


pd.pandas.set_option("display.max_columns", None)
pd.pandas.set_option("display.max_rows", None)
pd.set_option("display.float_format", lambda x: '%.2f' % x)


application_train = pd.read_csv("HomeCredit/Dataset/application_train.csv")


categoric_variable = application_train.columns[application_train.columns.str.contains("RATING|FLAG|STATUS|TYPE|CODE|NOT|APPR|AMT_REQ|BUREAU_HOUR|BUREAU_DAY|BUREAU_WEEK|DEF|FONDKAPREMONT_MODE|WALLSMATERIAL_MODE|EMERGENCYSTATE_MODE" ,regex=True)]


numeric_variable= [col for col in application_train.columns if application_train[col].dtype != "O"
                  and col not in "TARGET"
                  and col not in "SK_ID_CURR"
                  and col not in categoric_variable]



def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


def plot_categorical(dataframe,cat_cols):
    for col in cat_cols:
         temp = dataframe[col].value_counts()
         df1 = pd.DataFrame({col: temp.index, 'Number of contracts': temp.values})
         fig, ax1 = plt.subplots(ncols=1, figsize=(12, 6))
         sns.set_color_codes("pastel")
         sns.barplot(ax=ax1, x=col, y="Number of contracts", data=df1)
         plt.show();


def plot_numerical(dataframe, num_cols, size=[8, 4], bins=50):
    for col in num_cols:
        plt.figure(figsize=size)
        plt.title("Distribution of %s" % col)
        sns.distplot(dataframe[col], kde=True,bins=bins)
        plt.show()



def target_summary_with_cats(dataframe,categorical_cols, target):
    for col in categorical_cols:
        print(pd.DataFrame({col: dataframe[col].value_counts(dropna=False),
                                "Count": len(dataframe[col]),
                                "Ratio": 100 * dataframe[col].value_counts() / len(dataframe),
                                "TARGET_MEAN": dataframe.groupby(col)[target].mean()}))
        sns.countplot(x=col, hue="TARGET", data=dataframe);
        plt.show()
        print("\n\n")


def target_summary_with_nums(dataframe,numeric_cols, target):
    for col in numeric_cols:
        print(col, end="\n\n")
        print(pd.DataFrame({"Count": dataframe.groupby(target)[col].count(),
                            "TARGET_MEDIAN": dataframe.groupby(target)[col].median()}))
        fig, ax = plt.subplots(nrows=1, figsize=(12, 8))
        plt.subplots_adjust(hspace=0.4, top=0.8)
        g1 = sns.distplot(dataframe[col].loc[dataframe[target] == 1], ax=ax, color="g")
        g1 = sns.distplot(dataframe[col].loc[dataframe[target] == 0], ax=ax, color='r')
        g1.set_title(col + " Distribuition", fontsize=15)
        g1.set_xlabel(col)
        g1.set_xlabel("Frequency")
        plt.show()
        print("\n\n")
    print("\n\n")



def EDA(dataframe,cat_cols,num_cols,target):
    a,b = dataframe.shape
    print("Number of Columns: {0}   Number of Rows: {1}".format(a,b), end="\n\n")
    print("****** The Number of Unique Values for Each Variable ******", end="\n\n")
    print(dataframe.nunique(axis=0), end="\n\n")
    print("****** Numeric Variables ******", end="\n\n")
    print(num_cols, end="\n\n")
    print("****** Categorical Variables ******", end="\n\n")
    print(cat_cols, end="\n\n")
    print("****** The Number of Missing Value for Each Variable ******", end="\n\n")
    print(missing_data(dataframe), end="\n\n")
    plot_categorical(dataframe, cat_cols)
    print("Graph of Categorical Variables was plotted.", end="\n\n")
    plot_numerical(dataframe, num_cols)
    print("Graph of Numeric Variables was plotted.",end="\n\n" )
    print("****** Examination of Categorical Variables According to Target ******",end="\n\n")
    target_summary_with_cats(dataframe, cat_cols, target)
    print("****** Examination of Numerical Variables According to Target ******",end="\n\n")
    target_summary_with_nums(dataframe, num_cols, target)


EDA(application_train,categoric_variable,numeric_variable,"TARGET")




















