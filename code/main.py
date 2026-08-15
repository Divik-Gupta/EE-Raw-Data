from optuna.samplers import GPSampler
import optuna
from scipy.stats import qmc
from ucimlrepo import fetch_ucirepo
import pandas as pd
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier

#hyperparameter ranges used for BO and QMC (search space)
SEARCH_SPACE = {
    "n_estimators": (20, 300),
    "max_depth": (4, 30),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 15),
    "max_features": ["sqrt", "log2", None],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy", "log_loss"],
    "class_weight": [None, "balanced", "balanced_subsample"]
}


PARAM_ORDER = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "max_features",
    "bootstrap",
    "criterion",
    "class_weight"
]

INT_PARAMS = {
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf"
}

EVALUATIONS = 64
SEEDS = [42, 12, 456, 789, 2024]

#fetch dataset
adult = fetch_ucirepo(id=2)

#features and target variable
X = adult.data.features
y = adult.data.targets

y = y.squeeze()
y = y.map({
    "<=50K": 0,
    ">50K": 1,
    "<=50K.": 0,
    ">50K.": 1
})

#column type, for preprocessing
cat_cols = X.select_dtypes(include=["object", "category"]).columns
num_cols = X.select_dtypes(exclude=["object", "category"]).columns

#preprocessing
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), num_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
])

def log_result(method, evaluation, params, score, best_score, elapsed):
    #for storing data/results
    results.append({
        "method": method,
        "evaluation": evaluation,
        **params,
        "f1": score,
        "best_f1": best_score,
        "elapsed_time": elapsed
    })

dimensions = int(input("Dimension count: "))
active_params = PARAM_ORDER[:dimensions]
run_bo = input("Press any key for BO: ")
run_qmc = input("Press any key for QMC: ")

#for loop to perform all 5 trials on different seeds as there is randomness involved
for trial_num, seed in enumerate(SEEDS):
    print(f"\nTrial {trial_num+1} (seed={seed})")

    #split into train test (70-30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.3,
        random_state=seed,
        stratify=y
    )

    #preprocess only on the training data
    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    def evaluate(params):
        #train random forest with the input hyperparameters and return the f1-score on the test set
        #remains the same for both BO and QMC for fair comparison
        rf = RandomForestClassifier(**params, random_state=seed, n_jobs=-1).fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        return f1_score(y_test, y_pred)

    results = []

    if run_bo:

        #measure compute time
        start_time = time.perf_counter()
    
        sampler = GPSampler(seed=seed)

        study = optuna.create_study(
            direction="maximize",
            sampler=sampler
        )

        bo_best = [float("-inf")]

        def objective(trial):
            params = {}

            for param in active_params:

                values = SEARCH_SPACE[param]

                if param in INT_PARAMS:
                    params[param] = trial.suggest_int(param, *values)

                #elif param in FLOAT_PARAMS:
                #    params[param] = trial.suggest_float(param, *values)

                else:
                    params[param] = trial.suggest_categorical(param, values)

            score = evaluate(params)

            elapsed = time.perf_counter() - start_time

            bo_best[0] = max(bo_best[0], score)

            log_result(
                method="BO",
                evaluation=trial.number + 1,
                params=params,
                score=score,
                best_score=bo_best[0],
                elapsed=elapsed
            )

            return score

        study.optimize(objective, n_trials=EVALUATIONS)

    if run_qmc:
        
        start_time = time.perf_counter()

        sampler = qmc.Sobol(
            d=dimensions,
            scramble=True,
            seed=seed
        )
        samples = sampler.random_base2(m=6)    #64 evaluations
        print("Generated Sobol sequence...")

        def scale(sample):
            params = {}

            for i, param in enumerate(active_params):

                values = SEARCH_SPACE[param]

                if param in INT_PARAMS:

                    low, high = values
                    params[param] = int(round(low + sample[i] * (high - low)))

                #elif param in FLOAT_PARAMS:

                #    low, high = values
                #    params[param] = low + sample[i] * (high - low)

                else:

                    index = min(
                        int(sample[i] * len(values)),
                        len(values) - 1
                    )

                    params[param] = values[index]

            return params

        best = float("-inf")

        for i, sample in enumerate(samples, start=1):

            print(f"Evaluation {i}/{EVALUATIONS}", end=" ")
                
            params = scale(sample)

            print(f"Parameters: {params}", end=" ")

            score = evaluate(params)

            best = max(best, score)
            print(f"best f1: {best}", end=" ")

            elapsed = time.perf_counter() - start_time
            print(f"Current F1: {score:.4f}")
            
            log_result(
                method="QMC",
                evaluation=i,
                params=params,
                score=score,
                best_score=best,
                elapsed=elapsed
            )

    pd.DataFrame(results).to_csv(f"t{trial_num + 1}_{dimensions}d.csv", index=False)
