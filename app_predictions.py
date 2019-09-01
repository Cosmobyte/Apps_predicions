
# coding: utf-8

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials,rand
from time import time

# Reading the data
app_data = pd.read_csv('University project/AppleStore.csv',index_col=0)
app_description = pd.read_csv('University project/appleStore_description.csv')


app_description.head()
#  to see if there is any null data in the dataset and what type of attribute each one it is
app_data.info() 
app_description.info()


#  Merging datasets
app_data = pd.merge(app_data,app_description, on=['size_bytes','id','track_name'])
app_data.head()

def random_color_generator(number_of_colors):
# generator of random colors
    color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)]
    return color
def make_plot(x_variable,y_variable):
# creates plots taking the x and y variable as parameters 
    y_axes = app_data[[x_variable, y_variable]].groupby(x_variable).mean()[y_variable].sort_values(ascending=False)
    ind = np.arange(len(y_axes))
    x_axes = y_axes.index
    width = 0.8 # the width of the bars
    fig, ax = plt.subplots()
    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(18.5, 10.5)
    rects = ax.bar(ind - width/2, y_axes, width, color=random_color_generator(100),align="center")
    ax.set_ylabel(y_variable)
    ax.set_title(x_variable)
    ax.set_xticks(ind)
    ax.set_xticklabels(x_axes,rotation=40, ha ='right')
    plt.show()
        
count_types = app_data['prime_genre'].value_counts()
ind = np.arange(len(count_types)) 
print(count_types)

columns = app_data.columns.values.tolist()
# Countplot for each type of Prime_genre
for i in range(len(columns)):
    print(columns[i])
    print(type(app_data[columns[i]].head(1).values[0]))
    if(isinstance(app_data[columns[i]].head(1).values[0],(np.integer,np.float))):
        print(i)
        make_plot('prime_genre',columns[i])
        
width = 0.8
genre_label= app_data['prime_genre'].unique()


fig, ax = plt.subplots()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
rects1 = ax.bar(ind - width/2, count_types, width, color=random_color_generator(100))

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Count')
ax.set_title('Prime genre')
ax.set_xticks(ind)
ax.set_xticklabels(genre_label,rotation=40,ha = 'right')

plt.show()




# Correlation between the user rating and the other features
app_data_new = app_data.drop(['id','currency','track_name'],axis=1)

app_data_new['app_desc_len'] = app_data_new['app_desc'].str.len()
app_data_new1 = app_data_new.drop(['app_desc','prime_genre'],axis = 1)
app_data_new1.head()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
sns.heatmap(app_data_new1.corr(),annot=True, fmt=".2f")



# dropping some of the of the attributes that are not usefull 
app_data_new2 = app_data_new1.drop(['rating_count_ver','user_rating_ver','sup_devices.num','ver','cont_rating'],axis=1)
app_data_new2.head()


# spliting user_rating into to classes 

target = app_data_new2['user_rating']
bins = [-1,4,6]
classes = [0,1]
d = dict(enumerate(classes, 1))

target = pd.cut(target, bins, labels=classes)
print(target.value_counts())



# dropping some other attributes
app_data_new2 = app_data_new2.drop(['user_rating','price','size_bytes','vpp_lic'],axis=1)
# normalizing the remaining attributes
sc_X = StandardScaler()
X_train = sc_X.fit_transform(app_data_new2)
print(X_train)
X_train.shape



from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import warnings

clf_rf = RandomForestClassifier()
clf_lgbm = LGBMClassifier()
clf_mlpc = MLPClassifier()

models = [clf_rf,
          clf_lgbm,
          clf_mlpc
          ]

def hyperopt_train_test(params):
# this function checks the parameters and finds out which classifier fits them, then returns the cross validation results
    t = params['type']
    del params['type']
    if  t=='randomforest':
        clf = RandomForestClassifier(**params)
    elif  t=='lgbm':
        clf = LGBMClassifier(**params)
    elif  t=='mlp':
        clf = MLPClassifier(**params)
    else:
        return 0
    return cross_val_score(clf, X_train, target).mean()

space_rf = hp.choice('classifier_type', [
    {
        'type': 'randomforest',
        'max_depth': hp.choice('max_depth', range(1,20)),
        'max_features': hp.choice('max_features', range(1,5)),
        'n_estimators': hp.choice('n_estimators', range(1,20)),
        'criterion': hp.choice('criterion', ["gini", "entropy"])
    }])
space_lgbm = hp.choice('classifier_type', [
    {
        'type': 'lgbm',
        'num_leaves': hp.choice('num_leaves', np.arange(10, 200, dtype=int)),
        'min_data_in_leaf':  hp.choice ('min_data_in_leaf',np.arange(10, 200, dtype=int)),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.choice ('max_bin', np.arange(64, 512, dtype=int)),
        'bagging_freq': hp.choice ('bagging_freq', np.arange(1, 5, dtype=int)),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
    }])
space_mlp = hp.choice('classifier_type', [
    {
        'type' : 'mlp',
        'hidden_layer_sizes' : 10 + hp.randint('hidden_layer_sizes', 40),
        'alpha' : hp.loguniform('alpha', -8*np.log(10), 3*np.log(10)),
        'activation' : hp.choice('activation', ['relu', 'logistic', 'tanh']),
        'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam'])
    }])
space = [space_rf,space_lgbm,space_mlp]
best=0
def f(params):
# objective function to minimize
    global best 
    acc = hyperopt_train_test(params.copy())
    if acc > best :
        print ('new best:', acc, 'using', params['type'])
        best = acc   
    return {'loss':1-best,'status': STATUS_OK}

besty=[]
for i in range(len(models)):
# using hypeopt this funciton finds the best hyperparaters for the classifiers given
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    besty.append(fmin(f,space[i],algo= tpe.suggest,return_argmin=False,max_evals=100, trials=Trials()))


from hyperopt import space_eval
best_params = besty
best_models = []
for i in range(len(models)):
# assigning the best hyperparameters for the models 
    if(best_params[i]['type']=="randomforest"):
        del best_params[i]['type']
        best_models.append(RandomForestClassifier(max_features = best_params[i]['max_features'],criterion=best_params[i]['criterion'],
                                             n_estimators= best_params[i]['n_estimators'],max_depth = best_params[i]['max_depth']))
    elif(best_params[i]['type']=="lgbm"):
        del best_params[i]['type']
        best_models.append(LGBMClassifier(num_leaves= best_params[i]['num_leaves'], min_data_in_leaf= best_params[i]['min_data_in_leaf'], 
                                          learning_rate= best_params[i]['learning_rate'],feature_fraction = best_params[i]['feature_fraction'],
                                          bagging_freq=best_params[i]['bagging_freq'], lambda_l1= best_params[i]['lambda_l1'],
                                          max_bin = best_params[i]['max_bin'], min_sum_hessian_in_leaf= best_params[i]['min_sum_hessian_in_leaf'],
                                          lambda_l2= best_params[i]['lambda_l2']))
    elif(best_params[i]['type']=="mlp"):
        del best_params[i]['type']
        best_models.append(MLPClassifier(hidden_layer_sizes= best_params[i]['hidden_layer_sizes'], solver= best_params[i]['solver'],
                                         alpha= best_params[i]['alpha'], activation= best_params[i]['activation']))
print(best_models)


models_table = pd.DataFrame(columns=['train_score', 'test_score'])
for i, model in enumerate(best_models):
# finding the accuracy and recall of each of the best models
    warnings.filterwarnings(action='ignore', category=DeprecationWarning)
    cv_result = cross_validate(model, X_train, target, cv=10, scoring='accuracy',return_train_score=True)
    models_table.loc[i, 'Classfier_name'] = model.__class__.__name__
    models_table.loc[i, 'train_score'] = cv_result['train_score'].mean()
    models_table.loc[i, 'test_score'] = cv_result['test_score'].mean()
    y_pred = cross_val_predict(model, X_train, target, cv=10)
    print(model.__class__.__name__)
    print(confusion_matrix(target, y_pred))
    print()
    print(model.__class__.__name__ + ' Recall')
    print(recall_score(target, y_pred, average="macro")) 
    print()
models_table

