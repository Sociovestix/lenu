import pandas  # type: ignore
from lenu.data.lei import COL_ELF
from sklearn.model_selection import (  # type: ignore
    cross_validate,
    StratifiedShuffleSplit,
)


def evaluate_jurisdiction(jurisdiction_data, pipeline):

    X = jurisdiction_data
    y = jurisdiction_data[COL_ELF]

    crossvalidation = StratifiedShuffleSplit(n_splits=10, test_size=0.3, random_state=0)
    # try:
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=crossvalidation,
        scoring=[
            "accuracy",
            "balanced_accuracy",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
        ],
        return_train_score=True,
        return_estimator=True,
    )
    return pandas.Series(results)
