from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor

class StiffnessModels:
    @staticmethod
    def get_random_forest():
        # Hyperparameters from the paper/notebook
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42
        )

    @staticmethod
    def get_catboost():
        return CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            verbose=0,
            allow_writing_files=False
        )

    @staticmethod
    def get_svm():
        return SVR(
            kernel='rbf',
            C=100,
            gamma='scale'
        )