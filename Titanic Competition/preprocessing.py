
# coding: utf-8

# ## Titanic: Machine Learning from Disaster
# **Predict survival on the Titanic**


# Data Processing and Visualization Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#  Data Modelling Libraries
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier,
                             GradientBoostingClassifier, ExtraTreesClassifier,
                             VotingClassifier)

from sklearn.model_selection import (GridSearchCV, cross_val_score, cross_val_predict,
                                     StratifiedKFold, learning_curve)


from sklearn.metrics import (confusion_matrix, accuracy_score)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import warnings
from collections import Counter

sns.set(style = 'white' , context = 'notebook', palette = 'deep')
warnings.filterwarnings('ignore', category = DeprecationWarning)
get_ipython().run_line_magic('matplotlib', 'inline')



# load the datasets using pandas's read_csv method
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# concat these two datasets, this will come handy while processing the data
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)

# separately store ID of test datasets,
# this will be using at the end of the task to predict.
TestPassengerID = test['PassengerId']


# shape of the data set
print(train.shape)


# first 5 records
print(train.head())


# Definitions of each features and quick thoughts:
#
# - PassengerId. Unique identification of the passenger. It shouldn't be necessary for the machine learning model.
#
# - Survived. Survival (0 = No, 1 = Yes). Binary variable that will be our target variable.
# - Pclass. Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd). Ready to go.
# - Name. Name of the passenger. We need to parse before using it.
# - Sex. Gender Categorical variable that should be encoded. We can use dummy variable to encode it.
# - Age. Age in years.
# - SibSp. Siblings / Spouses aboard the Titanic.
# - Parch. Parents / Children aboard the Titanic.
# - Ticket. Ticket number. Big mess.
# - Fare. Passenger fare.
# - Cabin. Cabin number.
# - Embarked. Port of Embarkation , C = Cherbourg, Q = Queenstown, S = Southampton. Categorical feature that should be encoded. We can use feature mapping or make dummy vairables for it.
#
# The main conclusion is that we already have a set of features that we can easily use in our machine learning model. But features like Name, Ticket, Cabin require an additional effort before we can integrate them.


# using info method we can get quick overview of the data sets
print(train.info())

# Descriptive Statistics
print(train.describe())


# Create table for missing data analysis
def find_missing_data(data):
    Total = data.isnull().sum().sort_values(ascending = False)
    Percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

    return pd.concat([Total,Percentage] , axis = 1 , keys = ['Total' , 'Percent'])

find_missing_data(train)
find_missing_data(dataset)


# checking only datasets set
sns.heatmap(dataset.isnull(), cbar = False ,
            yticklabels = False , cmap = 'viridis')


# Outlier detection

def detect_outliers(df,n,features):
    """
    Takes a dataframe df of features and returns a list of the indices
    corresponding to the observations containing more than n outliers according
    to the Tukey method.
    """
    outlier_indices = []

    # iterate over features(columns)
    for col in features:

        # 1st quartile (25%)
        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)
        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)
        IQR = Q3 - Q1

        # outlier step
        outlier_step = 1.5 * IQR

        # Determine a list of indices of outliers for feature col
        outlier_list_col = df[(df[col] < Q1 - outlier_step) |
                              (df[col] > Q3 + outlier_step )].index
        # append the found outlier indices for col to the list of outlier indices
        outlier_indices.extend(outlier_list_col)


    # select observations containing more than 2 outliers
    outlier_indices = Counter(outlier_indices)

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )
    return multiple_outliers

# detect outliers from Age, SibSp , Parch and Fare
Outliers_to_drop = detect_outliers(train,2,["Age","SibSp","Parch","Fare"])


# Show the outliers rows
print(train.loc[Outliers_to_drop])

# Drop outliers
train = train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)

# after removing outlier, let's re-concat the data sets
dataset =  pd.concat(objs=[train, test], axis=0).reset_index(drop=True)


# a custom function for age imputation
def AgeImpute(df):
    Age = df[0]
    Pclass = df[1]

    if pd.isnull(Age):
        if Pclass == 1: return 37
        elif Pclass == 2: return 29
        else: return 24
    else:
        return Age

# Age Impute
dataset['Age'] = dataset[['Age' , 'Pclass']].apply(AgeImpute, axis = 1)


# In[22]:


# age featured imputed; no missing age records
sns.heatmap(dataset.isnull(), yticklabels = False, cbar = False, cmap = 'summer')



# Explore SibSp feature vs Survived
# We'll use factorplot to analysis
Sib_Sur = sns.factorplot(x="SibSp",y="Survived",data=train,
                   kind="bar", size = 6 , palette = "Blues")

Sib_Sur.despine(left=True)
Sib_Sur = Sib_Sur.set_ylabels("survival probability")


dataset["Fare"].isnull().sum()
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())


print(dataset['Sex'].head()) # top 5
print(' ')
print(dataset['Sex'].tail()) # last 5


# convert Sex into categorical value 0 for male and 1 for female
sex = pd.get_dummies(dataset['Sex'], drop_first = True)
dataset = pd.concat([dataset,sex], axis = 1)

# After now, we really don't need to Sex features, we can drop it.
dataset.drop(['Sex'] , axis = 1 , inplace = True)


# let's see the percentage
train[["Sex","Survived"]].groupby('Sex').mean()

# Count
print(dataset.groupby(['Embarked'])['PassengerId'].count())

# Compare with other variables
dataset.groupby(['Embarked']).mean()


# count missing values
print(dataset["Embarked"].isnull().sum())

# Fill Embarked nan values of dataset set with 'S' most frequent value
dataset["Embarked"] = dataset["Embarked"].fillna("S")


# Counting passenger based on Pclass and Embarked
Embarked_Pc = sns.factorplot("Pclass", col="Embarked",  data=dataset,
                   size=5, kind="count", palette="muted", hue = 'Survived')

Embarked_Pc.despine(left=True)
Embarked_Pc = Embarked_Pc.set_ylabels("Count")

# create dummy variable
embarked = pd.get_dummies(dataset['Embarked'], drop_first = True)
dataset = pd.concat([dataset,embarked], axis = 1)

# after now, we don't need Embarked coloumn anymore, so we can drop it.
dataset.drop(['Embarked'] , axis = 1 , inplace = True)
dataset.head()

dataset['Name'].head(10)

# Get Title from Name
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]

# add dataset_title to the main dataset named 'Title'
dataset["Title"] = pd.Series(dataset_title)

# count
dataset["Title"].value_counts()

# Plot bar plot (titles and Age)
plt.figure(figsize=(18,5))
sns.barplot(x=dataset['Title'], y = dataset['Age'])

# Means per title
print(dataset.groupby('Title')['Age'].mean())

# Convert to categorical values Title
dataset["Title"] = dataset["Title"].replace(['Lady', 'the Countess',
                                             'Capt', 'Col','Don', 'Dr',
                                             'Major', 'Rev', 'Sir', 'Jonkheer',
                                             'Dona'], 'Rare')

dataset["Title"] = dataset["Title"].map({"Master":0, "Miss":1, "Ms" : 1 ,
                                         "Mme":1, "Mlle":1, "Mrs":1, "Mr":2,
                                         "Rare":3})

dataset["Title"] = dataset["Title"].astype(int)

# Drop Name variable
dataset.drop(labels = ["Name"], axis = 1, inplace = True)

# viz counts the title coloumn
sns.countplot(dataset["Title"]).set_xticklabels(["Master","Miss-Mrs","Mr","Rare"]);

# Let's see, based on title what's the survival probability
sns.barplot(x='Title', y='Survived', data=dataset);

# viz top 5
dataset.head()


# Create a family size descriptor from SibSp and Parch
dataset["Famize"] = dataset["SibSp"] + dataset["Parch"] + 1

# Drop SibSp and Parch variables
dataset.drop(labels = ["SibSp",'Parch'], axis = 1, inplace = True)

# drop some useless features
dataset.drop(labels = ["Ticket",'Cabin','PassengerId'], axis = 1,
             inplace = True)
