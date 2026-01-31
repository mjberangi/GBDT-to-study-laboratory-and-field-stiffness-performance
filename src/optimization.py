import optuna
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from catboost import CatBoostRegressor

class BayesianOptimizer:
    def __init__(self, X, y, n_trials=15):
        self.X = X
        self.y = y
        self.n_trials = n_trials
        # 5-Fold Cross Validation
        self.kf = KFold(n_splits=5, shuffle=True, random_state=42)

    def optimize_svm(self):
        def objective(trial):
            # Hyperparameter search space
            C = trial.suggest_float('C', 0.1, 1000, log=True)
            gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
            epsilon = trial.suggest_float('epsilon', 0.01, 1.0)
            
            model = SVR(C=C, gamma=gamma, epsilon=epsilon)
            # Negative RMSE because Optuna minimizes
            score = cross_val_score(model, self.X, self.y, cv=self.kf, scoring='neg_root_mean_squared_error').mean()
            return -score 

        print("Optimizing SVM...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return SVR(**study.best_params)

    def optimize_rf(self):
        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 400)
            max_depth = trial.suggest_int('max_depth', 5, 30)
            min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
            
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
            score = cross_val_score(model, self.X, self.y, cv=self.kf, scoring='neg_root_mean_squared_error').mean()
            return -score

        print("Optimizing Random Forest...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return RandomForestRegressor(**study.best_params, random_state=42)

    def optimize_catboost(self):
        def objective(trial):
            params = {
                'iterations': trial.suggest_int('iterations', 100, 1000),
                'depth': trial.suggest_int('depth', 4, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10, log=True),
                'verbose': 0,
                'allow_writing_files': False,
                'random_state': 42
            }
            
            model = CatBoostRegressor(**params)
            score = cross_val_score(model, self.X, self.y, cv=self.kf, scoring='neg_root_mean_squared_error').mean()
            return -score

        print("Optimizing CatBoost...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=self.n_trials)
        return CatBoostRegressor(**study.best_params, verbose=0, allow_writing_files=False, random_state=42)