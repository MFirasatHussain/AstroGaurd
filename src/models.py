from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_models():
    """Returns a dictionary of models."""
    models = {
        "DecisionTree": DecisionTreeClassifier(),
        "RandomForest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "AdaBoost": AdaBoostClassifier(n_estimators=50)
    }
    return models
