# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import matplotlib.pyplot as plt
import mglearn
import pandas as pd
import numpy  as np

#########################################################################
## Read data
#########################################################################
root_dir = "/Users/fenghan/Documents/UT/2018fall/machine learning in\
             practice/project/EHRData_24/"

encounter_df  = pd.read_csv(root_dir + "Patient_demographics.csv")
medication_df = pd.read_csv(root_dir + "Patient_medication_24_hours.csv")

Lab         = pd.read_csv(root_dir + "Patient_labs_24_hours.csv")
Lab.rename(columns={'Patient_SK': 'PATIENT_SK'}, inplace=True)
Lab_names = Lab.loc[:, 'PATIENT_SK':'LAB_PROCEDURE_NAME']
Lab_name_value = Lab.loc[:, 'PATIENT_SK':'RESULT_VALUE_NUM']

encounter_df.columns
medication_df.columns

#########################################################################
# Select some of the columns
#########################################################################
encounter_df = encounter_df[['PATIENT_SK', 'AGE_IN_YEARS', 'Trauma_IND', 
                             'RACE', 'GENDER', 'MARITAL_STATUS', 
                             'VASPRESOR_Class', 'Mortality']]
medication_df = medication_df[['PATIENT_SK', 'GENERIC_NAME']]
encounter_df = encounter_df.dropna()

encounter_df = encounter_df[encounter_df['GENDER'] != 'unknown']
encounter_df.shape
medication_df.shape

encounter_df = encounter_df.sort_values('PATIENT_SK')
medication_df = medication_df.sort_values('PATIENT_SK')

encounter_df.dropna().describe()
medication_df.dropna().describe()

print(encounter_df.Mortality.value_counts())
print(encounter_df.VASPRESOR_Class.value_counts())
print(encounter_df.Trauma_IND.value_counts())
print(encounter_df.GENDER.value_counts())
print(encounter_df.RACE.value_counts())
print(encounter_df.MARITAL_STATUS.value_counts())

encounter_df.groupby('GENDER')[['Mortality']].mean()
encounter_df.groupby('VASPRESOR_Class')[['Mortality']].mean()
encounter_df.groupby(['VASPRESOR_Class', 'GENDER'])[['Mortality']].aggregate('mean').unstack()

encounter_df.groupby(['VASPRESOR_Class', 'GENDER'])[['Mortality']].mean().unstack()

encounter_df.pivot_table('Mortality', index='GENDER', columns='VASPRESOR_Class')
age = pd.cut(encounter_df['AGE_IN_YEARS'], [0, 30, 50, 90])
encounter_df.pivot_table('Mortality', ['GENDER', age], 'VASPRESOR_Class')

encounter_df.groupby('Trauma_IND')[['Mortality']].mean()
encounter_df.groupby(['VASPRESOR_Class', 'Trauma_IND'])['Mortality'].aggregate('mean')

#encounter_df.groupby('Mortaltiy')[['Length_of_stay']].describe()


# Create dummy variables
patient_df_dummies = pd.get_dummies(Demographic, columns=
                     ['VASPRESOR_Class', 'RACE', 'GENDER','MARITAL_STATUS'])

Medication_dummies = pd.get_dummies(Medication, columns=['GENERIC_NAME'])
Medication_dummies_u = Medication_dummies.groupby('PATIENT_SK', 
                                                  as_index = False).max()

### Lab ###

# Get dummy
Lab_dummies = pd.get_dummies(Lab_names, columns=['LAB_PROCEDURE_NAME'])
Lab_dummies_u = Lab_dummies.groupby('PATIENT_SK',as_index = False).max()
# Total of 3808 patients have lab records

# Get the means of each lab test for each patient
Lab_means = Lab_name_value.groupby(['PATIENT_SK','LAB_PROCEDURE_NAME']).mean()   
# Set the index as a column
Lab_means_index = Lab_means.reset_index()
# Pivot the Lab_means_index table
Lab_means_pivot = Lab_means_index.pivot(index='PATIENT_SK', 
                                        columns='LAB_PROCEDURE_NAME',
                                         values='RESULT_VALUE_NUM')
# Set the index as PATIENT_SK
Lab_means_pivot = Lab_means_pivot.reset_index()

# Merge Lab dummy and Lab mean values
Lab_data = pd.merge(Lab_dummies_u,Lab_means_pivot,
                        how = 'left', on='PATIENT_SK')
Lab_data.isna().sum() # No missing values

# Merge Demographic and Medication table
patient_med_data = pd.merge(patient_df_dummies, Medication_dummies_u,
                        how = 'left', on='PATIENT_SK')
patient_med_data = patient_med_data.fillna(0)

# Merge patient_data and lab_data
patient_data = pd.merge(patient_med_data, Lab_data,
                        how = 'left', on='PATIENT_SK')
patient_data.isna().sum() # No missing values

# Impute missing lab record with the median of all the patients
def median_impute(x):
    x=x.fillna(x.median())
    return(x)
patient_data = patient_data.apply(median_impute, 1)
patient_data.isna().sum() # No missing values

# Get the column names
genericname = patient_data.columns.values
list(genericname)

# Drop vassapresser from medication
patient_data = patient_data.drop(columns=['PATIENT_SK','GENERIC_NAME_dopamine',
                           'GENERIC_NAME_phenylephrine',
                           'GENERIC_NAME_norepinephrine'])



# ----------------------------------------
# Visualize the summary statistics
# ----------------------------------------
# Bar plot by groups
import seaborn as sns
fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Mortality', data=encounter_df, hue='GENDER')
ax.set_ylim(0,2000)
plt.title("Impact of Gender on Motality")
plt.show()

fig, ax=plt.subplots(figsize=(8,6))
sns.countplot(x='Mortality', data=encounter_df, hue='VASPRESOR_Class')
ax.set_ylim(0,2000)
plt.title("Impact of VASPRESOR_Class on Mortality")
plt.show()

Alive = encounter_df[encounter_df['Mortality'] == 0] 
Death = encounter_df[encounter_df['Mortality'] == 1] 

# Continuous variable histogram by groups
# Define a function for an overlaid histogram
def overlaid_histogram(data1, data1_name, data1_color, data2, data2_name, data2_color, x_label, y_label, title):
    # Set the bounds for the bins so that the two distributions are
    # fairly compared
    max_nbins = 10
    data_range = [min(min(data1), min(data2)), max(max(data1), max(data2))]
    binwidth = (data_range[1] - data_range[0]) / max_nbins
    bins = np.arange(data_range[0], data_range[1] + binwidth, binwidth)

    # Create the plot
    _, ax = plt.subplots()
    ax.hist(data1, bins = bins, color = data1_color, alpha = 1, label = data1_name)
    ax.hist(data2, bins = bins, color = data2_color, alpha = 0.75, label = data2_name)
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.legend(loc = 'best')

# Call the function to create plot
overlaid_histogram(data1 = Alive['AGE_IN_YEARS']
                   , data1_name = 'Mortality=0'
                   , data1_color = '#539caf'
                   , data2 = Death['AGE_IN_YEARS']
                   , data2_name = 'Mortality=1'
                   , data2_color = '#7663b0'
                   , x_label = 'Age in years'
                   , y_label = 'Frequency'
                   , title = 'Distribution of Age By Mortality')

# -------------------------------------------------
# Prepare the data for prediction models
#--------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

X = patient_data.drop("Mortality",axis=1)
y = patient_data["Mortality"]

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
training_accuracy = []
test_accuracy = []

# Scaler for the training data
scaler_1 = MinMaxScaler().fit(X_train)
scaler_2 = StandardScaler().fit(X_train)
X_train_scaled_MinMax   = scaler_1.transform(X_train)
X_train_scaled_Standard = scaler_2.transform(X_train)
X_test_scaled_MinMax = scaler_1.transform(X_test)
X_test_scaledStandard = scaler_2.transform(X_test)

# -------------------------------------------------
# Fitting in Prediction Models
#--------------------------------------------------

#######################
### Random Forest #####

# Grid Search with Cross Validation for Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

forest = RandomForestClassifier(random_state=2) 
param_grid = { 
    'n_estimators': [50,100,200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [5,6,7,8],
    'criterion' :['gini', 'entropy']
}
CV_rfc = GridSearchCV(estimator=forest, param_grid=param_grid, cv= 10)
CV_rfc.fit(X_train, y_train)
CV_rfc.best_params_
CV_rfc.best_score_
# The Best paramiters:
# {'criterion': 'gini','max_depth': 8,
# 'max_features': 'auto','n_estimators': 200}

# Applied the best result from grid search
forest_select = RandomForestClassifier(random_state=2, max_features='auto', 
                                n_estimators= 200, max_depth=8, 
                                criterion='gini')
forest_select.fit(X_train, y_train)

print("Accuracy on training set: {:.3f}"
      .format(forest_select.score(X_train, y_train))) # 0.901
print("Accuracy on test set: {:.3f}".format(forest_select
      .score(X_test, y_test))) #0.840

# Feature importances computed from a random forest 
importances = forest_select.feature_importances_ 
std = np.std([tree.feature_importances_ for tree in forest_select.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
data_name = list(patient_data.columns.values)
print("Feature ranking:")
for f in range(30):
    print("%d. %s (%f)" % (f + 1, data_name[indices[f]], 
          importances[indices[f]]))
# Plot the feature importances of the forest
plt.figure()
importances = forest_select.feature_importances_ 
indices = np.argsort(importances)[::-1][:29,]
plt.title('Feature Importances') 
plt.barh(range(len(indices)), importances[indices], color='b', align='center') 
plt.yticks(range(len(indices)), [data_name[i] for i in indices],fontsize=6) 
plt.xlabel('Relative Importance') 
plt.show()

###################
#### Lasso ########
from sklearn.linear_model import Lasso
lasso = Lasso(random_state=0)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{'alpha': alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X_train, y_train)
clf.best_params_  # 0.0028
clf.best_score_   # 0.35

lasso = Lasso(alpha=0.0028, max_iter=100000).fit(X_train, y_train)
print("Training set score: {:.2f}".format(lasso.score(X_train, y_train)))
print("Test set score: {:.2f}".format(lasso.score(X_test, y_test)))
print("Number of features used: {}".format(np.sum(lasso.coef_ != 0)))
# Training set score: 0.44
# Test set score: 0.30
# Number of features used: 175


################################
#Neural Networks (Deep Learning)
################################
from sklearn.neural_network import MLPClassifier

parameters = {'solver': ['lbfgs'], 
              'max_iter': [1000], 
              'alpha': 10.0 ** -np.arange(1, 10), 
              'hidden_layer_sizes':np.arange(1, 2), 
              'random_state':[0,1,2,3,4,5,6,7,8,9]}

clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1)

clf.fit(X_train_scaled_MinMax, y_train)
print(clf.score(X_train_scaled_MinMax, y_train))
print(clf.best_params_)


mlp = MLPClassifier(max_iter=1000, random_state=0)
mlp.fit(X_train_scaled_MinMax, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score
      (X_train_scaled_MinMax, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score
      (X_test_scaled_MinMax, y_test)))
# Accuracy on training set: 0.99
# Accuracy on test set: 0.84

mlp_max = MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp_max.fit(X_train_scaled_MinMax, y_train)
print("Accuracy on training set: {:.2f}".format(mlp_max.score
      (X_train_scaled_MinMax, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp_max.score
      (X_test_scaled_MinMax, y_test)))
# Accuracy on training set: 0.93
# Accuracy on test set: 0.84

mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=1)
mlp.fit(X_train_scaled_MinMax, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score
      (X_train_scaled_MinMax, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score
      (X_test_scaled_MinMax, y_test)))
# Accuracy on training set: 0.91
# Accuracy on test set: 0.85

mlp = MLPClassifier(max_iter=1000, alpha=1,hidden_layer_sizes = 2,
                    random_state=0)
mlp.fit(X_train_scaled_MinMax, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score
      (X_train_scaled_MinMax, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score
      (X_test_scaled_MinMax, y_test)))
# Accuracy on training set: 0.92
# Accuracy on test set: 0.85


mlp = MLPClassifier(max_iter=1000, alpha=0.1, random_state=0)
mlp.fit(X_train_scaled_MinMax, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score
      (X_train_scaled_MinMax, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score
      (X_test_scaled_MinMax, y_test)))
# Accuracy on training set: 0.99
# Accuracy on test set: 0.83


## NN on standard scaler
mlp = MLPClassifier(max_iter=1000, alpha=1, random_state=1)
mlp.fit(X_train_scaled_Standard, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score
      (X_train_scaled_Standard, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score
      (X_test_scaled_Standard, y_test)))
# Accuracy on training set: 0.98
# Accuracy on test set: 0.85

mlp = MLPClassifier(max_iter=1000, alpha=10, random_state=1)
mlp.fit(X_train_scaled_Standard, y_train)
print("Accuracy on training set: {:.2f}".format(mlp_standard.score
      (X_train_scaled_Standard, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp_standard.score
      (X_test_scaled_Standard, y_test)))
# Accuracy on training set: 0.94
# Accuracy on test set: 0.85

mlp = MLPClassifier(max_iter=1000, alpha=10, hidden_layer_sizes = 2,
                    random_state=3)
mlp.fit(X_train_scaled_Standard, y_train)
print("Accuracy on training set: {:.2f}".format(mlp.score
      (X_train_scaled_Standard, y_train)))
print("Accuracy on test set: {:.2f}".format(mlp.score
      (X_test_scaled_Standard, y_test)))
# Accuracy on training set: 0.93
# Accuracy on test set: 0.84

# Heatmap of MLP
plt.figure(figsize=(20, 5))
plt.imshow(mlp.coefs_[0][:29,], interpolation='none', cmap='viridis')
plt.yticks(range(30), data_name[:29], fontsize = 6)
plt.xlabel("Columns in weight matrix")
plt.ylabel("Input feature")
plt.colorbar()


##########
# Logistic Regression
###########
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline


#Logistic regression data
#adjust C value for regulization(C=1)
logreg = LogisticRegression().fit(X_train_scaled_MinMax, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_MinMax, 
      y_train)))#Training set score: 0.922
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled_MinMax, 
      y_test)))#Test set score: 0.834

logreg100 = LogisticRegression(C=100).fit(X_train_scaled_MinMax, y_train)
print("Training set score:{:.3f}".format(logreg100.score(X_train_scaled_MinMax, 
      y_train)))#Training set score: 0.834
print("Test set score: {:.3f}".format(logreg100.score(X_test_scaled_MinMax, 
      y_test)))#Test set score: 0.800


logreg001 = LogisticRegression(C=0.01).fit(X_train_scaled_MinMax, y_train)
print("Training set score:{:.3f}".format(logreg001.score(X_train_scaled_MinMax, 
      y_train)))#Training set score: 0.853
print("Test set score: {:.3f}".format(logreg001.score(X_test_scaled_MinMax,
      y_test)))#Test set score: 0.840


#PLOT
plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.001")
xlims = plt.xlim()
plt.hlines(0, xlims[0], xlims[1])
plt.xlim(xlims)
plt.ylim(-5, 5)
plt.xlabel("Feature")
plt.ylabel("Coefficient magnitude")
plt.legend()

####Standard scaler
#Logistic regression data
#adjust C value for regulization(C=1)
logreg = LogisticRegression().fit(X_train_scaled_Standard, y_train)
print("Training set score: {:.3f}".format(logreg.score(X_train_scaled_Standard,
      y_train)))#Training set score: 0.973
print("Test set score: {:.3f}".format(logreg.score(X_test_scaled_Standard, 
      y_test)))#Test set score: 0.801

logreg100 = LogisticRegression(C=100).fit(X_train_scaled_Standard, y_train)
print("Training set score: {:.3f}".format(
        logreg100.score(X_train_scaled_Standard,
                        y_train)))#Training set score: 0.986
print("Test set score: {:.3f}".format(logreg100.score(X_test_scaled_Standard, 
      y_test)))#Test set score: 0.772

logreg001 = LogisticRegression(C=0.01).fit(X_train_scaled_Standard, y_train)
print("Training set score: {:.3f}".format(logreg001.score(
        X_train_scaled_Standard,y_train)))#Training set score: 0.936
print("Test set score: {:.3f}".format(logreg001.score(X_test_scaled_Standard, 
      y_test)))#Test set score: 0.838

#pipeline
# Standard Scaler
param_grid = {'logisticregression__C': [0.01, 0.1, 1, 10, 100]}

pipe = make_pipeline(StandardScaler(), LogisticRegression())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_))
# {'logisticregression__C': 0.01}
print("Best parameters: ", grid.best_params_)#Test set accuracy: 0.84
print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test)))
#Best cross-validation accuracy: 0.85

#MinMax Scaler
pipe = make_pipeline(MinMaxScaler(), LogisticRegression())
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)

#use the scaled training data to run our grid search using cross-validation
print("Best cross-validation accuracy: {:.2f}".format(grid.best_score_)) #0.85
print("Best parameters: ", grid.best_params_) #0.85
print("Test set accuracy: {:.2f}".format(grid.score(X_test, y_test))) #0.84

# Applied the best result from grid search
logreg01 = LogisticRegression(C=0.1).fit(X_train_scaled_Standard, y_train)
print("Training set score: {:.3f}".format(logreg01.score(
        X_train_scaled_Standard, y_train)))#Training set score: 0.955
print("Test set score: {:.3f}".format(
        logreg01.score(X_test_scaled_Standard, y_test)))#Test set score: 0.823

# Feature importances computed from a logitic Regression 
coefs=logreg01.coef_[0]
coefs_abs=np.absolute(logreg01.coef_[0])
top_thirty = np.argpartition(coefs_abs, -30)[-30:]
top_thirty_sorted=top_twenty[np.argsort(coefs[top_thirty])]
print(coefs[top_thirty_sorted])
print(X.columns[top_thirty_sortedt])
coef_dict = dict(zip(coefs[top_thirty_sortedt], X.columns[top_thirty_sortedt]))

##roc
roc=roc_auc_score(y_test, logreg01.predict_proba(X_test_scaled_Standard)[:,1])
cr=classification_report(y_test, logreg01.predict(X_test_scaled_Standard))


#---------------------------------
# Model Evaluation
#------------------------------

################################
#precision, recall, and f-score
################################
#f-score
from sklearn.metrics import f1_score
print("f1 score Random Forest: {:.2f}".format(f1_score(y_test, 
      forest_select.predict(X_test)))) #0.79
print("f1 score MLP standard scale: {:.2f}".format(f1_score(y_test, 
      mlp.predict(X_test_scaled_Standard)))) #0.81
print("f1 score logistic regression: {:.2f}".format(f1_score(y_test, 
      logreg01.predict(X_test_scaled_Standard)))) #0.79

#comprehensive summary of precision, recall, and f1-score
from sklearn.metrics import classification_report
print(classification_report(y_test, forest_select.predict(X_test), 
                            target_names=["Alive", "Death"]))
#reports precision, recall, and f-score with this class as the positive class
print(classification_report(y_test, mlp.predict(X_test_scaled_Standard), 
                            target_names=["Alive", "Death"]))
print(classification_report(y_test, logreg01.predict(X_test_scaled_Standard), 
                            target_names=["Alive", "Death"]))

##########################################
#  Precision-Recall curves
##########################################

from sklearn.metrics import precision_recall_curve
# Precision_recall_curve for Random forest
precision_rf,recall_rf,thresholds_rf = precision_recall_curve(
        y_test,forest_select.predict_proba(X_test)[:, 1])

plt.plot(precision_rf, recall_rf, label="Random Forest")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")

# Precision_recall_curve for Neural Network with standard scale
mlp_pred = mlp.predict_proba(X_test_scaled_Standard)[:, 1]
precision_mlp, recall_mlp, thresholds_mlp = precision_recall_curve(
        y_test, mlp_pred)

# Precision for logistic regression
precision_log,recall_log,thresholds_log = precision_recall_curve(
        y_test,logreg01.predict_proba(X_test_scaled_Standard)[:, 1])


# Precision curve of Random forest and Neural network
plt.plot(precision_log, recall_log, label="logreg")
close_default_log = np.argmin(np.abs(thresholds_log - 0.5))
plt.plot(precision_log[close_default_log], recall_log[close_default_log], 'o',
         c='k', markersize=10, label="threshold 0.5 rf", fillstyle="none", 
         mew=2)
plt.plot(precision_rf, recall_rf, label="rf")
close_default_rf = np.argmin(np.abs(thresholds_rf - 0.5))
plt.plot(precision_rf[close_default_rf], recall_rf[close_default_rf], 'v',
         c='k', markersize=10, label="threshold 0.5 rf", fillstyle="none", 
         mew=2)
plt.plot(precision_mlp, recall_mlp, label="mlp")
close_default_mlp = np.argmin(np.abs(thresholds_mlp - 0.5))
plt.plot(precision_mlp[close_default_mlp], recall_mlp[close_default_mlp], '^',
         c='k', markersize=10, label="threshold 0.5 mlp", fillstyle="none", 
         mew=2)
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc="best")
#Comparing precision recall curves of SVM and random forest

# Comparing two precision-recall curves provides
from sklearn.metrics import average_precision_score
ap_rf = average_precision_score(
        y_test, forest_select.predict_proba(X_test)[:, 1])
rf_pred = forest_select.predict_proba(X_test)[:, 1]
ap_mlp = average_precision_score(y_test, mlp.predict_proba(
        X_test_scaled_Standard)[:, 1])
ap_logreg = average_precision_score(
        y_test, logreg01.predict_proba(X_test_scaled_Standard)[:, 1])
print("Average precision of random forest: {:.3f}".format(ap_rf)) #0.862
print("Average precision of MLP: {:.3f}".format(ap_mlp)) #0.865
print("Average precision of Logreg: {:.3f}".format(ap_logreg)) #0.794


# %%
# ##### Receiver Operating Characteristics (ROC) and AUC

from sklearn.metrics import roc_curve
#ROC curve for Random Forest
fpr, tpr, thresholds = roc_curve(y_test, 
                                 forest_select.predict_proba(X_test)[:, 1])

fpr_log,tpr_log,thresholds_log = roc_curve(
        y_test,logreg01.predict_proba(X_test_scaled_Standard)[:, 1])

#Comparing ROC curves for Logistic Regression, random forest and MLP
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(
        y_test, mlp.predict_proba(X_test_scaled_Standard)[:, 1])

plt.plot(fpr_log, tpr_log, label="logreg")
close_default_log = np.argmin(np.abs(thresholds_log - 0.5))
plt.plot(fpr_log[close_default_log], tpr_log[close_default_log], 'o',
         c='k', markersize=10, label="threshold 0.5 rf", fillstyle="none", 
         mew=2)
plt.plot(fpr, tpr, label="ROC Curve RF")
plt.plot(fpr_mlp, tpr_mlp, label="ROC Curve MLP")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
close_default_rf = np.argmin(np.abs(thresholds - 0.5))
plt.plot(fpr[close_default_rf], tpr[close_default_rf], 'v', 
         markersize=10, label="threshold 0.5 RF", fillstyle="none", c='k', 
         mew=2)
close_default_mlp = np.argmin(np.abs(thresholds_mlp - 0.5))
plt.plot(fpr_mlp[close_default_mlp], tpr_mlp[close_default_mlp], '^', 
         markersize=10, label="threshold 0.5 MLP", fillstyle="none", c='k', 
         mew=2)
plt.legend(loc=4)

#AUC
from sklearn.metrics import roc_auc_score
from sklearn import metrics
rf_auc = roc_auc_score(y_test, forest_select.predict_proba(X_test)[:, 1])
mlp_auc = roc_auc_score(y_test, mlp_pred)
logreg_auc = roc_auc_score(y_test, logreg01.predict_proba(
        X_test_scaled_Standard)[:, 1])
print("AUC for Random Forest: {:.3f}".format(rf_auc)) #0.911
print("AUC for MLP: {:.3f}".format(mlp_auc)) #0.905
print("AUC for MLP: {:.3f}".format(logreg_auc)) #0.874
