import pickle

from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score


def heart_disease(data):
    X = data.drop("target", axis=1)
    y = data['target']

    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(X, y)

    model = pickle.load(open("model/model_xgboost.pkl", 'rb'))

    y_pred = model.predict(X)
    accuracy = round(accuracy_score(y, y_pred) * 100, 2)

    df_final = X
    df_final['target'] = y

    return model, accuracy, df_final