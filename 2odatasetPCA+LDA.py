from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.pipeline import Pipeline
import numpy as np
import matplotlib.pyplot as plt

digits = load_digits()
print(digits.data.shape)
data = digits.data
y = digits.target


X_final = []
for i in range(len(digits.images)):
    X_final.append(digits.images[i].flatten())
X_final = np.array(X_final)

X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.3, random_state=42)

#kanonikopoiisi twn train kai twn test dedomenwn

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
scaler.fit(X_test)
X_test = scaler.transform(X_test)


def my_scorer(estimator, X_train, y=None):
    X_reduced = estimator.transform(X_train)
    X_preimage = estimator.inverse_transform(X_reduced)
    return -1 * mean_squared_error(X_train, X_preimage)

#upologismos tou kaluterou sindiasmou synartisis pirina kai parametrou gamma

param_grid = [{
    "gamma": [0.0001, 0.001, 0.01, 0.1],
    "kernel": ["rbf", "linear", "poly"],
}]

kpca = KernelPCA(fit_inverse_transform=True, n_jobs=-1)
grid_search = GridSearchCV(kpca, param_grid, cv=3, scoring=my_scorer)
grid_search.fit(X_train)
print(grid_search.best_params_)

#upologismos tis kaliteris ptwsis diastasis me KPCA

#steps = [ ('KPCA', KernelPCA(kernel='linear')), ('KNC', NearestCentroid( metric='euclidean'))]
steps = [ ('KPCA', KernelPCA(kernel='linear')), ('KNN',KNeighborsClassifier(n_neighbors=1, metric='euclidean'))]
pipeline = Pipeline(steps)
parameters = {'KPCA__gamma': [0.0001],'KPCA__n_components':np.arange(2, 300)}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=10, n_jobs=3)  # define a GridSearchCV object
grid.fit(X_train, y_train)
print('\n best parameters based on train data', grid.best_params_)
print('\n all the parameters:', grid.cv_results_)
print('Score on training_set: ', grid.score(X_train, y_train))
print('Score on test_set: ', grid.score(X_test, y_test))

# Plot tis meiwsis diastasts me to mean test score gia PCA

aa = parameters['KPCA__n_components']
bb = grid.cv_results_['mean_test_score']
plt.scatter(aa, bb)
plt.title('')
plt.xlabel('Number of KPCA Components', fontsize = 10)
plt.ylabel('Mean Test Score', fontsize = 10)
plt.show()

#upologismos tis kaliteris ptwsis diastasis me LDA

#steps = [ ('KPCA', KernelPCA(kernel='linear',gamma=0.0001,n_components=60)),('LDA', LDA()),('KNC',NearestCentroid( metric='euclidean'))]
steps = [ ('KPCA', KernelPCA(kernel='linear',gamma=0.0001,n_components=59)),('LDA', LDA()),('KNN',KNeighborsClassifier(n_neighbors=1, metric='euclidean'))]
pipeline = Pipeline(steps)
parameters = {'LDA__n_components':np.arange(2, 9)}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=3, verbose=10, n_jobs=3)
grid.fit(X_train, y_train)
print('\n best parameters based on train data', grid.best_params_)
print('\n all the parameters:', grid.cv_results_)
print('Score on training_set: ', grid.score(X_train, y_train))
print('Score on test_set: ', grid.score(X_test, y_test))

# Plot tis meiwsis diastasts me to mean test score gia LDA

aa = parameters['LDA__n_components']
bb = grid.cv_results_['mean_test_score']
plt.scatter(aa, bb)
plt.title('')
plt.xlabel('Number of LDA Components', fontsize = 10)
plt.ylabel('Mean Test Score', fontsize = 10)
plt.show()



#grafiki parastasi twn dedomenwn me KPCA stis 2 diastaseis

transformer = KernelPCA(n_components=2, kernel='linear')
X_transformed = transformer.fit_transform(X_train)
X_transformed.shape

colordict = {0:'red',
             1: 'blue',
             2:'yellow',
             3: 'cyan',
             4: 'magenta',
             5: 'orange',
             6: 'black',
             7: 'pink',
             8: 'green',
             9: 'grey'}

colormap = [colordict[x] for x in y_train]
plt.scatter(X_transformed[:,0],X_transformed[:,1], c = colormap)
plt.show()

#grafiki parastasi twn dedomenwn me kpca + lda stis 2 diastaseis

transformer = KernelPCA(n_components=9, kernel='linear')
y = transformer.fit_transform(X_train)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis(n_components = 2)
data = clf.fit_transform(y, y_train)

colordict = {0:'red',
             1: 'blue',
             2:'yellow',
             3: 'cyan',
             4: 'magenta',
             5: 'orange',
             6: 'black',
             7: 'pink',
             8: 'green',
             9: 'grey'}


colormap = [colordict[x] for x in y_train]
plt.scatter(data[:,0],data[:,1], c = colormap)
plt.show()