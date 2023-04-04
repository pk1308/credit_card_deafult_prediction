import argparse
import importlib
import os
import warnings
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from optuna import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier

from CreditCard.entity import OptunaTrainingArtifact
from CreditCard.entity.artifact_entity import MetricEvalArtifact
from CreditCard.logging import logger
from CreditCard.utils import write_yaml, read_yaml_as_dict, read_yaml, create_directories, load_numpy_array_data
from CreditCard.utils.model_eval import evaluate_classification_model

warnings.filterwarnings("ignore")


class ModelFactory:
    def __init__(self, model_factory_config_path: Path, data_to_train: np.ndarray = None):
        """Model factory class for training models using optuna hyperparameter tuning

        Args:
            model_factory_config_path (Path): model config file path
            data_to_train (np.ndarray, optional): data to train if not specified in config file. Defaults to None.
        """
        self.target_test = None
        self.target_train = None
        self.features_test = None
        self.features_train = None
        self.Model_factory_config_info = read_yaml(Path(model_factory_config_path))
        self.data_to_train = self.get_data_to_train(data_to_train)
        self.metrics = self.Model_factory_config_info.metrics
        self.report_path = Path(self.Model_factory_config_info.report_path)
        self.model_keys = self.Model_factory_config_info.models.keys()
        self.num_trails = self.Model_factory_config_info.num_trails
        create_directories([os.path.dirname(self.report_path)])

    def get_data_to_train(self, data_to_train: np.ndarray = None):
        """ class to get data to train from config file or from data_to_train argument

        Args:
            data_to_train (np.ndarray, optional): data to train if not specified. Defaults to None.

        Returns:
            _type_: _description_
        """
        if data_to_train is None:
            data_to_train = load_numpy_array_data(Path(self.Model_factory_config_info.data_path))
        features = data_to_train[:, :-1]
        target = data_to_train[:, -1].astype(int)
        self.features_train, self.features_test, self.target_train, self.target_test = train_test_split(features,
                                                                                                        target,
                                                                                                        test_size=0.2,
                                                                                                        random_state=1)
        return data_to_train

    @staticmethod
    def update_model_config_file(data_to_update: dict, report_file_path: Path):
        """Update model config file with new data
            data_to update is a dictionary with key as model name and value as dictionary of hyperparameters
            report_file_path is the path of the config file"""
            
        model_key = list(data_to_update.keys())[0]
        config = dict()
        new_file = False
        if os.path.exists(report_file_path):
            config = read_yaml_as_dict(Path(report_file_path))
        if model_key in config.keys():
            logger.info(f"Model {model_key} already exists in config file")
            refrence_data = config.get(model_key)
            if refrence_data["Model_Best_params_score"] < data_to_update[model_key]["Model_Best_params_score"]:
                logger.info(f"Updating model {model_key} in config file")
                config[model_key] = data_to_update[model_key]
                write_yaml(data=config, file_path=Path(report_file_path))
            else:
                logger.info(f"Model {model_key} already exists in config file with better score")            
            new_file = False
        else:
            config.update(data_to_update)
            write_yaml(data=config, file_path=Path(report_file_path))
            new_file = True
        
        logger.info(f"new_file created is {new_file}")
        return new_file

    def __lda_objective(self, trial: Trial):
        """Objective function for LDA model

        Args:
            trial (Trial): optuna trial object

        Returns:
            _type_: trail score
        """
        # define hyperparameter to tune
        solver = trial.suggest_categorical("solver", ["svd", "lsqr", "eigen"])
        shrinkage = trial.suggest_categorical("shrinkage", ["auto", None])
        tol = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
        if solver == "svd":
            shrinkage = None
        # define model
        model = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage, tol=tol)

        # define cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        # evaluate model
        scores = cross_val_score(model, self.features_train, self.target_train, scoring=self.metrics, cv=cv, n_jobs=-1)

        # return mean accuracy
        return scores.mean()

    def __get_lda_best_param(self, report_file_path: Path):
        """Get best hyperparameter for LDA model"""
        sampler = TPESampler(seed=1)
        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", study_name="LDA", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self.__lda_objective(trial), n_trials=self.num_trails)
        logger.info(f"""Best hyperparameter are {study.best_params} with a {self.metrics} of {study.best_value}""")
        training_artifact_file_name = self.Model_factory_config_info.models.LDA.training_artifact_file_name
        training_artifact_file_path = os.path.join(os.path.dirname(self.report_path), training_artifact_file_name)

        model_config_info = self.Model_factory_config_info.models.LDA
        lda_artifact = OptunaTrainingArtifact(Model_index=model_config_info.model_index,
                                              Model_module=model_config_info.model_module,
                                              Model_class=model_config_info.model_class,
                                              Model_Best_params=study.best_params,
                                              Model_Best_params_score=study.best_value,
                                              training_artifact_file_path=training_artifact_file_path)
        data_to_update = {model_config_info.Model_index: lda_artifact.dict()}

        self.update_model_config_file(data_to_update=data_to_update, report_file_path=report_file_path)
        study.trials_dataframe().to_csv(training_artifact_file_path)
        return study.best_params

    def __gbt_objective(self, trial: Trial):
        """ Objective function for GBT model

        Args:
            trial (Trial):  optuna trial object

        Returns:
            _type_: trail score
        """
        loss = trial.suggest_categorical("loss", ["log_loss", "exponential"])
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        subsample = trial.suggest_float("subsample", 0.1, 1.0)
        criterion = trial.suggest_categorical("criterion", ["friedman_mse", "squared_error"])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        min_weight_fraction_leaf = trial.suggest_float("min_weight_fraction_leaf", 0.0, 0.5)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_impurity_decrease = trial.suggest_float("min_impurity_decrease", 0.0, 0.5)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
        tol = trial.suggest_float("tol", 1e-4, 1e-1, log=True)
        ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.5)

        model = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
                                           subsample=subsample, criterion=criterion,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_depth=max_depth,
                                           min_impurity_decrease=min_impurity_decrease,
                                           max_features=max_features, tol=tol, ccp_alpha=ccp_alpha)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        cv_scores = cross_val_score(model, self.features_train, self.target_train, scoring=self.metrics, cv=cv,
                                    n_jobs=-1)

        return cv_scores.mean()

    def __get_gbt_best_param(self, report_file_path: Path):
        """Get best hyperparameter for GBT model"""
        model_config_info = self.Model_factory_config_info.models.GBT
        sampler = TPESampler(seed=1)
        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", study_name="GBT", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self.__gbt_objective(trial), n_trials=self.num_trails)
        logger.info(f"""Best hyperparameter are {study.best_params} with a {self.metrics} of {study.best_value}""")
        training_artifact_file_name = model_config_info.training_artifact_file_name
        training_artifact_file_path = os.path.join(os.path.dirname(self.report_path), training_artifact_file_name)

        gbt_artifact = OptunaTrainingArtifact(Model_index=model_config_info.model_index,
                                              Model_module=model_config_info.model_module,
                                              Model_class=model_config_info.model_class,
                                              Model_Best_params=study.best_params,
                                              Model_Best_params_score=study.best_value,
                                              training_artifact_file_path=training_artifact_file_path)
        data_to_update = {model_config_info.Model_index: gbt_artifact.dict()}

        self.update_model_config_file(data_to_update=data_to_update, report_file_path=report_file_path)
        study.trials_dataframe().to_csv(training_artifact_file_path)
        return study.best_params

    def __lgbm_objective(self, trial: Trial):
        """ Objective function for LGBM model

        Args:
            trial (Trial): optuna trial object

        Returns:
            _type_: trail score
        """
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        num_leaves = trial.suggest_int("num_leaves", 2, 256)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 1, 10)
        lambda_l1 = trial.suggest_float("lambda_l1", 0, 100, step=5)
        lambda_l2 = trial.suggest_float("lambda_l2", 0, 100, step=5)
        min_gain_to_split = trial.suggest_float("min_gain_to_split", 0, 100, step=5)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.2, 0.95, step=0.1)
        bagging_freq = trial.suggest_int("bagging_freq", 1, 10)
        feature_fraction = trial.suggest_float("feature_fraction", 0.2, 0.95, step=0.1)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        model = LGBMClassifier(objective="binary", n_estimators=n_estimators, learning_rate=learning_rate,
                               num_leaves=num_leaves, max_depth=max_depth,
                               min_data_in_leaf=min_data_in_leaf, lambda_l1=lambda_l1, lambda_l2=lambda_l2,
                               min_gain_to_split=min_gain_to_split,
                               bagging_fraction=bagging_fraction, bagging_freq=bagging_freq,
                               feature_fraction=feature_fraction)

        cv_scores = cross_val_score(model, self.features_train, self.target_train, scoring=self.metrics, cv=cv,
                                    n_jobs=-1)

        return cv_scores.mean()

    def __get_lgbm_best_param(self, report_file_path: Path):
        """Get best hyperparameter for LGBM model"""
        model_config_info = self.Model_factory_config_info.models.LGBM
        sampler = TPESampler(seed=1)
        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", study_name="LGBM", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self.__lgbm_objective(trial), n_trials=self.num_trails)
        logger.info(f"""Best hyperparameter are {study.best_params} with a {self.metrics} of {study.best_value}""")
        training_artifact_file_name = model_config_info.training_artifact_file_name
        training_artifact_file_path = os.path.join(os.path.dirname(self.report_path), training_artifact_file_name)

        lgbm_artifact = OptunaTrainingArtifact(Model_index=model_config_info.model_index,
                                              Model_module=model_config_info.model_module,
                                              Model_class=model_config_info.model_class,
                                              Model_Best_params=study.best_params,
                                              Model_Best_params_score=study.best_value,
                                              training_artifact_file_path=training_artifact_file_path)
        data_to_update = {model_config_info.Model_index: lgbm_artifact.dict()}

        self.update_model_config_file(data_to_update=data_to_update, report_file_path=report_file_path)
        study.trials_dataframe().to_csv(training_artifact_file_path)
        return study.best_params

    def __xgboost_objective(self, trial: Trial):
        """ Objective function for XGBoost model

        Args:
            trial (Trial): optuna trial object

        Returns:
            _type_: trail score
        """
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)
        min_child_weight = trial.suggest_int("min_child_weight", 1, 10)
        gamma = trial.suggest_loguniform('gamma', 1e-8, 1.0)
        subsample = trial.suggest_loguniform('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_loguniform('colsample_bytree', 0.5, 1.0)
        reg_alpha = trial.suggest_loguniform('reg_alpha', 1e-8, 1.0)
        reg_lambda = trial.suggest_loguniform('reg_lambda', 1e-8, 1.0)
        eval_metric = trial.suggest_categorical('eval_metric', ['auc', 'logloss'])

        model = XGBClassifier(max_depth=max_depth, learning_rate=learning_rate, n_estimators=n_estimators,
                              min_child_weight=min_child_weight,
                              gamma=gamma, subsample=subsample, colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                              reg_lambda=reg_lambda,
                              eval_metric=eval_metric)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        cv_score = cross_val_score(model, self.features_train, self.target_train, scoring=self.metrics, cv=cv,
                                   n_jobs=-1)

        return cv_score.mean()

    def __get_xgboost_best_param(self, report_file_path: Path):
        """Get best hyperparameter for XGBoost model"""
        model_config_info = self.Model_factory_config_info.models.XGBOOST
        sampler = TPESampler(seed=1)
        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", study_name="XGBOOST", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self.__gbt_objective(trial), n_trials=self.num_trails)
        logger.info(f"""Best hyperparameter are {study.best_params} with a {self.metrics} of {study.best_value}""")
        training_artifact_file_name = model_config_info.training_artifact_file_name
        training_artifact_file_path = os.path.join(os.path.dirname(self.report_path), training_artifact_file_name)

        lda_artifact = OptunaTrainingArtifact(Model_index=model_config_info.model_index,
                                              Model_module=model_config_info.model_module,
                                              Model_class=model_config_info.model_class,
                                              Model_Best_params=study.best_params,
                                              Model_Best_params_score=study.best_value,
                                              training_artifact_file_path=training_artifact_file_path)
        data_to_update = {model_config_info.Model_index: lda_artifact.dict()}

        self.update_model_config_file(data_to_update=data_to_update, report_file_path=report_file_path)
        study.trials_dataframe().to_csv(training_artifact_file_path)
        return study.best_params

    def __rf_objective(self, trial: Trial):
        """ Objective function for Random Forest model

        Args:
            trial (Trial): optuna trial object
        """
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        max_depth = trial.suggest_int("max_depth", 3, 10)
        max_features = "sqrt"
        max_leaf_nodes = trial.suggest_int("max_leaf_nodes", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)
        n_estimators = trial.suggest_int("n_estimators", 100, 1000)

        model = RandomForestClassifier(bootstrap=bootstrap, criterion=criterion, max_depth=max_depth,
                                       max_features=max_features,
                                       max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
                                       n_estimators=n_estimators)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        cv_score = cross_val_score(model, self.features_train, self.target_train, scoring=self.metrics, cv=cv,
                                   n_jobs=-1)
        return cv_score.mean()

    def __get_rf_best_param(self, report_file_path: Path):
        """Get best hyperparameter for Random Forest model"""
        model_config_info = self.Model_factory_config_info.models.RF
        sampler = TPESampler(seed=1)
        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", study_name="RF", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self.__rf_objective(trial), n_trials=self.num_trails)
        logger.info(f"""Best hyperparameter are {study.best_params} with a {self.metrics} of {study.best_value}""")
        training_artifact_file_name = model_config_info.training_artifact_file_name
        training_artifact_file_path = os.path.join(os.path.dirname(self.report_path), training_artifact_file_name)

        rf_artifact = OptunaTrainingArtifact(Model_index=model_config_info.model_index,
                                              Model_module=model_config_info.model_module,
                                              Model_class=model_config_info.model_class,
                                              Model_Best_params=study.best_params,
                                              Model_Best_params_score=study.best_value,
                                              training_artifact_file_path=training_artifact_file_path)
        data_to_update = {model_config_info.Model_index: rf_artifact.dict()}

        self.update_model_config_file(data_to_update=data_to_update, report_file_path=report_file_path)
        study.trials_dataframe().to_csv(training_artifact_file_path)
        return study.best_params

    def __catboost_objective(self, trial: Trial):
        """ Objective function for CatBoost model
        Args:
            trial (Trial): optuna trial object

        Returns:
            _type_: trail score
        """ 
        objective = trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"])
        colsample_bylevel = trial.suggest_float("colsample_bylevel", 0.01, 1.0, log=True)
        depth = trial.suggest_int("depth", 3, 12)
        boosting_type = trial.suggest_categorical("boosting_type", ["Ordered", "Plain"])
        bootstrap_type = trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"])
        bagging_temperature = None
        subsample = None

        if bootstrap_type == "Bayesian":
            bagging_temperature = trial.suggest_float("bagging_temperature", 0.0, 10.0)
        elif bootstrap_type == "Bernoulli":
            subsample = trial.suggest_float("subsample", 0.01, 1.0, log=True)

        model = CatBoostClassifier(objective=objective, colsample_bylevel=colsample_bylevel, depth=depth,
                                   boosting_type=boosting_type,
                                   bootstrap_type=bootstrap_type, bagging_temperature=bagging_temperature,
                                   subsample=subsample)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        cv_score = cross_val_score(model, self.features_train, self.target_train, scoring=self.metrics, cv=cv,
                                   n_jobs=-1)
        return cv_score.mean()

    def __get_catboost_best_param(self, report_file_path: Path):
        """Get best hyperparameter for CatBoost model"""
        model_config_info = self.Model_factory_config_info.models.CATBOOST
        sampler = TPESampler(seed=1)
        pruner = MedianPruner(n_warmup_steps=5)
        study = optuna.create_study(direction="maximize", study_name="CATBOOST", sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: self.__rf_objective(trial), n_trials=self.num_trails)
        logger.info(f"""Best hyperparameter are {study.best_params} with a {self.metrics} of {study.best_value}""")
        training_artifact_file_name = model_config_info.training_artifact_file_name
        training_artifact_file_path = os.path.join(os.path.dirname(self.report_path), training_artifact_file_name)

        cat_artifact = OptunaTrainingArtifact(Model_index=model_config_info.model_index,
                                              Model_module=model_config_info.model_module,
                                              Model_class=model_config_info.model_class,
                                              Model_Best_params=study.best_params,
                                              Model_Best_params_score=study.best_value,
                                              training_artifact_file_path=training_artifact_file_path)
        data_to_update = {model_config_info.Model_index: cat_artifact.dict()}

        self.update_model_config_file(data_to_update=data_to_update, report_file_path=report_file_path)
        study.trials_dataframe().to_csv(training_artifact_file_path)
        return study.best_params

    def initiate_model_params_search(self, ) -> Path:
        """ Initiate model hyperparameter search

        Returns:
            Path: path to model report file to train model

        """
        model_report_path = self.report_path

        if "LDA" in self.model_keys:
            self.__get_lda_best_param(report_file_path=model_report_path)
        if "GBT" in self.model_keys:
            self.__get_gbt_best_param(report_file_path=model_report_path)
        if "LGBM" in self.model_keys:
            self.__get_lgbm_best_param(report_file_path=model_report_path)
        if "XGBOOST" in self.model_keys:
            self.__get_xgboost_best_param(report_file_path=model_report_path)
        if "RF" in self.model_keys:
            self.__get_rf_best_param(report_file_path=model_report_path)
        if "CATBOOST" in self.model_keys:
            self.__get_catboost_best_param(report_file_path=model_report_path)

        return model_report_path

    @staticmethod
    def class_for_name(module_name, class_name):
        """
        This function is equivalent to
        from module_name import class_name
        return: class_name
        """
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise e

    @staticmethod
    def update_property_of_class(instance_ref, property_data: dict):
        """ Update property of class

        Args:
            instance_ref (_type_): instance of class to update property
            property_data (dict): dictionary of property to update

        Raises:
            Exception: _description_
            e: _description_

        Returns:
            _type_: updated instance of class
        """
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            logger.info(property_data)
            for key, value in property_data.items():
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise e

    def get_model_list(self, best_model_config_info):
        """ Get list of model instance

        Args:
            best_model_config_info (_type_):  path to best hyperparameter for each model

        Returns:
            _type_:  model list with best model with best hyperparameter
        """
        initial_model_list = list()
        for model_key, model_info in best_model_config_info.items():
            model_module = model_info.Model_module
            model_class = model_info.Model_class
            model_best_params = model_info.Model_Best_params
            model_class_ref = self.class_for_name(module_name=model_module, class_name=model_class)
            model_instance = model_class_ref()
            model_instance = self.update_property_of_class(instance_ref=model_instance, property_data=model_best_params)
            initial_model_list.append(model_instance)
        return initial_model_list

    def get_best_model(self, model_report_path: Path) -> list:
        """ Get best trained model with best hyperparameter

        Args:
            model_report_path (Path): path to model report file

        Returns:
            list: _description_
        """
        best_model_config_info = read_yaml(Path(model_report_path))
        base_model_list = self.get_model_list(best_model_config_info=best_model_config_info)
        train_model_list = [model.fit(self.features_train, self.target_train) for model in base_model_list]
        return train_model_list

    def get_best_evaluated_model(self, train_model_list, base_accuracy: float, base_report_dir: Path,
                                 eval_difference: float, eval_param: str, columns_trained_on: list):
        """ Get best evaluated model

        Args:
            train_model_list (_type_): trained model list
            base_accuracy (float): minimum accuracy to evaluate model
            base_report_dir (Path): path to report directory
            eval_difference (float): minimum difference between test accuracy and train accuracy
            eval_param (str): evaluation parameter
            columns_trained_on (list): list of columns trained on

        Returns:
            _type_:  report of best model
            object : best model object
        """

        base_metric_info: MetricEvalArtifact = evaluate_classification_model(estimators=train_model_list,
                                                                             x_train_eval=self.features_train,
                                                                             y_train=self.target_train,
                                                                             x_test_eval=self.features_test,
                                                                             y_test=self.target_test,
                                                                             base_accuracy=base_accuracy,
                                                                             report_dir=str(base_report_dir),
                                                                             eval_difference=eval_difference,
                                                                             eval_param=eval_param,
                                                                             experiment_id="base_model",
                                                                             columns=columns_trained_on)
        logger.info(f"base model info {base_metric_info.dict()}")
        base_model_object = base_metric_info.best_model
        report = base_metric_info.best_model_report
        return base_model_object, report


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", type=str, required=True)

    args = args_parser.parse_args()
    ModelFactory_config_path = Path(args.config)

    model_factory = ModelFactory(model_factory_config_path=ModelFactory_config_path)
    best_model_report = model_factory.initiate_model_params_search()
    model_list = model_factory.get_best_model(model_report_path=best_model_report)
    trained_model = model_factory.get_best_evaluated_model(train_model_list=model_list, base_accuracy=0.6,
                                                           base_report_dir=Path("reports"),
                                                           eval_difference=0.05, eval_param="accuracy")
