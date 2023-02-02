from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, ConfusionMatrixDisplay, roc_curve, auc
import matplotlib.pyplot as plt
from typing import Tuple, Union
import logging


def ml_scaling(input_train: np.ndarray, input_test: np.ndarray,
               input_val: np.ndarray = None, scaling_type: str = "standard") -> \
        Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    This function performs scaling of the input data for the machine learning model.


    :param scaling_type: str, default="standard". A string mentioning the type of desired scaling. "standard", "minmax",
    "maxabs", "robust" corresponds to StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler respectively.

    :param input_train: np.ndarray. A numpy array of input train data.

    :param input_test: np.ndarray. A numpy array of input test data.

    :param input_val: np.ndarray, default=None. A numpy array of input validation data.

    :return input_train_scaled: np.ndarray. A numpy array of scaled input train data.

    :return input_test_scaled: np.ndarray. A numpy array of scaled input test data.

    :return input_val_scaled: np.ndarray. A numpy array of scaled input validation data.
    It is returned only input_val is provided.
    """

    if scaling_type == "standard":
        scaler = StandardScaler()
    elif scaling_type == "minmax":
        scaler = MinMaxScaler()
    elif scaling_type == "maxabs":
        scaler = MaxAbsScaler()
    elif scaling_type == "robust":
        scaler = RobustScaler()
    else:
        logging.error(f"scaling_type is expected to be either 'standard', 'minmax', 'maxabs' or 'robust' "
                      f"but it is given as '{scaling_type}'")
        raise ValueError(f"scaling_type is expected to be either 'standard', 'minmax', 'maxabs' or 'robust' "
                         f"but it is given as '{scaling_type}'")

    input_train_scaled = scaler.fit_transform(input_train)
    input_test_scaled = scaler.transform(input_test)

    if input_val is not None:
        input_val_scaled = scaler.transform(input_val)
        logging.info(f"The input_train, input_test and input_val sets are successfully scaled using {scaler}")
        return input_train_scaled, input_test_scaled, input_val_scaled
    else:
        logging.info(f"The input_train and input_test sets are successfully scaled using {scaler}")
        return input_train_scaled, input_test_scaled


def ml_preprocessing(input_data: np.ndarray, output_data: np.ndarray,
                     scaling_type: str = "standard", data_spilt_type: str = "test_train_split",
                     train_size: float = 0.7) -> \
        Union[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
              Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    This function performs scaling and dataset split for cross validation for machine learning model.

    :param input_data: np.ndarray. A numpy array of full input data.

    :param output_data: np.ndarray. A numpy array of full output data.

    :param scaling_type: str, default="standard". A string mentioning the type of desired scaling. "standard", "minmax",
    "maxabs", "robust" corresponds to StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler respectively.

    :param data_spilt_type: str, default="test_train_split". A string mentioning how the cross validation sets
    to be split. if "test_train_split" then whole data set is split into two, train set and test set.
    if "test_train_val_split" then whole data set is split into three, train set, test set and validation set.

    :param train_size: float, default = 0.7. A float value indicating the fraction of train data. 1-train_size will be
    size of the test set when data_spilt_type == "test_train_split". if data_spilt_type == "test_train_val_split" then
    (1-train_size)/2 will be size of the test set and validation set each.

    :return input_train: np.ndarray. A numpy array of scaled input train data.

    :return input_test: np.ndarray. A numpy array of scaled input test data.

    :return output_train: np.ndarray. A numpy array of output train data.

    :return output_test: np.ndarray. A numpy array of output data.

    :return input_val: np.ndarray. A numpy array of scaled input validation data. It is returned only when
    data_spilt_type == "test_train_val_split".

    :return output_train: np.ndarray. A numpy array of output validation data. It is returned only when
    data_spilt_type == "test_train_val_split".
    """

    if not 0 < train_size < 1:
        logging.error(f"'train_size is expected to a float value with range (0,1) but given as {train_size}'")
        raise ValueError(f"'train_size is expected to a float value with range (0,1) but given as {train_size}'")

    if data_spilt_type == "test_train_split":

        input_train, input_test, output_train, output_test = train_test_split(input_data, output_data,
                                                                              test_size=(1 - train_size),
                                                                              random_state=0)

        logging.info(f"Test train split done successfully. Train data : {round(train_size * 100, 2)}% and "
                     f"test data: {round((1 - train_size) * 100, 2)}%")

        input_train, input_test = ml_scaling(scaling_type=scaling_type, input_train=input_train,
                                             input_test=input_test)

        return input_train, input_test, output_train, output_test

    elif data_spilt_type == "test_train_val_split":
        input_train, input_test_val, output_train, output_test_val = train_test_split(input_data, output_data,
                                                                                      test_size=(1 - train_size),
                                                                                      random_state=0)

        input_test, input_val, output_test, output_val = train_test_split(input_test_val, output_test_val,
                                                                          test_size=0.5, random_state=0)

        logging.info(f"Test train validation split done successfully. Train data : {train_size * 100}%, "
                     f"test data: {round(((1 - train_size) / 2) * 100, 2)}%, "
                     f"validation data: {round(((1 - train_size) / 2) * 100, 2)}%")

        input_train, input_test, input_val = ml_scaling(scaling_type=scaling_type,
                                                        input_train=input_train,
                                                        input_test=input_test, input_val=input_val)

        return input_train, input_test, input_val, output_train, output_test, output_val

    else:
        logging.error(f"The parameter 'data_spilt_type' is expected to be either 'test_train_split' or "
                      f"'test_train_val_split' but it is given as '{data_spilt_type}'")
        raise (ValueError(f"The parameter 'data_spilt_type' is expected to be either 'test_train_split' or "
                          f"'test_train_val_split' but it is given as '{data_spilt_type}'"))


def ml_results_preparation(output_pred: np.ndarray, output_actual: np.ndarray, model_name: str, show: bool = True) \
        -> None:
    """
    This function calculates the metrics accuracy, recall and precision and store them in a file in
    results folder. This function also stores confusion matrix picture to the results folder.

    :param output_pred: np.ndarray. A numpy array of model predictions of output.

    :param output_actual: np.ndarray. A numpy array of actual output.

    :param model_name: str. A string indicating the name of the model.

    :param show: bool. A boolean value indicating whether to show the pictures while running or not.
    """

    accuracy = accuracy_score(output_pred, output_actual)
    recall = recall_score(output_pred, output_actual)
    precision = precision_score(output_pred, output_actual)

    results_filepath = f'..\\results\\{model_name.replace(" ", "")}_results.txt'
    with open(results_filepath, 'w') as results:
        logging.info(f"Writing results of {model_name} model to {results_filepath}")
        results.write(f"Results of {model_name} model:\n\n")
        results.write(f"Accuracy score: {accuracy}\n")
        results.write(f"Recall score: {recall}\n")
        results.write(f"Precision score: {precision}\n")

    logging.info(f"Successfully written results of {model_name} model to {results_filepath}")

    ConfusionMatrixDisplay.from_predictions(y_true=output_actual, y_pred=output_pred, normalize='all')

    plt.title(f'Confusion matrix for {model_name} model')
    plt.savefig(f'..\\results\\{model_name.replace(" ", "")}_cm.png')
    logging.info(f'Successfully stored {model_name.replace(" ", "")}_cm.png')

    if show:
        plt.show()


def ml_2models_compare_ROC(output_actual: np.ndarray, output_pred_model1: np.ndarray, model1_name: str,
                           output_pred_model2: np.ndarray, model2_name: str) -> None:
    """
    This functions helps in comparison of two models using ROC and AUC metrics. It gives the image of ROC curves of two
    models which are compared.

    :param output_actual: np.ndarray. A numpy array of actual output.

    :param output_pred_model1: np.ndarray. A numpy array of model1's predictions of output.

    :param model1_name: str. A string indicating the name of the model1.

    :param output_pred_model2: np.ndarray. A numpy array of model2's predictions of output.

    :param model2_name: str. A string indicating the name of the model2.
    """

    false_pos_rate_model1, true_pos_rate_model1, thresholds_model1 = roc_curve(output_actual, output_pred_model1)
    auc_model1 = auc(false_pos_rate_model1, true_pos_rate_model1)

    false_pos_rate_model2, true_pos_rate_model2, thresholds_model2 = roc_curve(output_actual, output_pred_model2)
    auc_model2 = auc(false_pos_rate_model2, true_pos_rate_model2)

    plt.plot(false_pos_rate_model1, true_pos_rate_model1, label=f"ROC for {model1_name} (AUC = {round(auc_model1, 3)})")
    plt.plot(false_pos_rate_model2, true_pos_rate_model2, label=f"ROC for {model2_name} (AUC = {round(auc_model2, 3)})")
    plt.plot(np.linspace(0, 1, 50), np.linspace(0, 1, 50), label="Reference line (TPR = FPR)", linestyle='--')
    plt.legend()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves: {model1_name} vs {model2_name}")
    plt.savefig(f'..\\results\\{model1_name.replace(" ", "")}_vs_{model2_name.replace(" ", "")}_roc.png')
    logging.info(f'Successfully stored {model1_name.replace(" ", "")}_vs_{model2_name.replace(" ", "")}_roc.png')
    # plt.grid()
    plt.show()
