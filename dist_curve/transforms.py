# AUTOGENERATED! DO NOT EDIT! File to edit: 04_Univariate_Transforms.ipynb (unless otherwise specified).

__all__ = ['prepFeatures', 'trainOOBClassifier', 'trainKFoldClassifier', 'getOptimalTransform']

# Cell
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm.notebook import tqdm

# Cell
def prepFeatures(X):
    "Apply z-score normalization to nxd feature matrix"
    ss = StandardScaler(with_mean=True, with_std=True)
    Xz = ss.fit_transform(X)
    return Xz,ss

# Cell
def trainOOBClassifier(X,y, modelFactory=lambda: DecisionTreeClassifier(),n_estimators=100):
    """
    Train ensemble of <n_estimators> models predicting the probability that each
    instance came from the labeled positive, rather than the unlabeled mixture, set.

    Required Arguments:
        - X : ndarray shape (n,d) : feature matrix
        - y : ndarray shape (n,)  : positive v. unlabeled component assignments for each instance
    Optional Arguments:
        - modelFactory : lambda function returning sklearn-style model instance (has fit, fit_predict, predict_proba, ... functions) : default DicisionTreeRegressor
        - n_estimators : size of the ensemble : default 100

    Returns
        - transform_scores : ndarray (n,) : probability that each instance came from labeled positive set, calculating using out-of-bag scores
        - auc_pu : float : the AUROC of this non-traditional classifier
    """
    # z-score normalization is applied to the whole dataset prior to training
    X,ss = prepFeatures(X)
    clf = BaggingClassifier(n_jobs=-1,base_estimator=modelFactory(), n_estimators=n_estimators,
                            max_samples=X.shape[0],max_features=X.shape[1], bootstrap=True,
                            bootstrap_features=False, oob_score=True).fit(X,y)
    transform_scores = clf.oob_decision_function_[:,1]
    auc_pu = roc_auc_score(y, transform_scores)
    return transform_scores, auc_pu

# Cell
def trainKFoldClassifier(X,y, modelFactory=lambda: SVC(probability=True, degree=1),KFoldValue=10):
    """
    Train model using K-fold cross-validation
    Required Arguments:
        - X : ndarray shape (n,d) : feature matrix
        - y : ndarray shape (n,)  : positive v. unlabeled component assignments for each instance
    Optional Arguments:
        - modelFactory : lambda function returning sklearn-style model instance (has fit, fit_predict, predict_proba, ... functions) : default SVC
        - KFoldValue : number of folds to use in k-fold cross-validation : default 10

    Returns
        - transform_scores : ndarray (n,) : probability that each instance came from labeled positive set
        - auc_pu : float : the AUROC of this non-traditional classifier

    """
    transform_scores = np.zeros(y.shape, dtype=float)
    # z-score normalization applied globally rather than within each k-fold iteration
    X,ss = prepFeatures(X)
    kf = KFold(n_splits=KFoldValue, shuffle=False)
    for train_indices, val_indices in kf.split(X):
        X_train, y_train = X[train_indices], y[train_indices]
        X_val = X[val_indices]
        clf = modelFactory()
        clf.fit(X_train, y_train)
        transform_scores[val_indices] = clf.predict_proba(X_val)[:,1]
    auc_pu = roc_auc_score(y, transform_scores)
    return transform_scores, auc_pu

# Cell
def getOptimalTransform(X,y):
    """
    Train the 6 univariate transforms from (Zeiberg 2020) and return the transform scores and auc_pu for the best transform

    Required Arguments:
        - X : ndarray shape (n,d) : feature matrix
        - y : ndarray shape (n,)  : positive v. unlabeled component assignments for each instance
    Returns:
        - transform_scores : ndarray (n,) : probability that each instance came from labeled positive set
        - auc_pu : float : the AUROC of this non-traditional classifier
    """
    transform_scores, auc_pu = {},{}
    models = [("nn_1",lambda: MLPClassifier(hidden_layer_sizes=(1,1)), 100),
              ("nn_5",lambda: MLPClassifier(hidden_layer_sizes=(1,1)), 100),
              ("nn_25",lambda: MLPClassifier(hidden_layer_sizes=(1,1)), 100),
              ("rt",lambda: DecisionTreeClassifier(), 1000),
              ("svm_1",lambda: SVC(kernel="poly", degree=1, probability=True),10),
              ("svm_2",lambda: SVC(kernel="poly", degree=1, probability=True),10)]
    for model_name, model_factory, n in tqdm(models,total=len(models),desc="Training univariate transforms"):
        if "svm" in model_name:
            scores, auc = trainKFoldClassifier(X,y,modelFactory=model_factory,KFoldValue=n)
        else:
            scores, auc = trainOOBClassifier(X,y,modelFactory=model_factory, n_estimators=n)
        transform_scores[model_name] = scores
        auc_pu[model_name] = auc
    # Find the best transform
    best_auc = .5
    best_transform = "rt"
    for model_name, auc in auc_pu.items():
        if auc > best_auc:
            best_transform = model_name
            best_auc = auc
    return transform_scores[best_transform], auc_pu[best_transform]