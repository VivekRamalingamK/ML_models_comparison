import sklearn.datasets as datasets
from src.ml_functions import ml_preprocessing, ml_results_preparation, ml_2models_compare_ROC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import logging

logging.basicConfig(filename="..\\logs\\ml.log", level=logging.INFO,
                    format="%(asctime)s: %(levelname)s:  %(message)s")


def main():

    # reading the data
    cancer_data = datasets.load_breast_cancer()
    input_data, output_data = cancer_data.data, cancer_data.target

    # data pre-processing
    input_train, input_test, output_train, output_test = ml_preprocessing(input_data=input_data,
                                                                          output_data=output_data,
                                                                          scaling_type='standard',
                                                                          data_spilt_type='test_train_split')

    # fitting of model1 - Random forest classifier
    model1 = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=0)
    model1.fit(input_train, output_train)

    # evaluation of model1
    output_test_pred_model1 = model1.predict(input_test)
    ml_results_preparation(output_pred=output_test_pred_model1,
                           output_actual=output_test, model_name="Random Forest Classifier")

    # fitting of model2 - Logistic Regression
    model2 = LogisticRegression(C=0.5, random_state=0)
    model2.fit(input_train, output_train)

    # evaluation of model2
    output_test_pred_model2 = model2.predict(input_test)
    ml_results_preparation(output_pred=output_test_pred_model2,
                           output_actual=output_test, model_name="Logistic Regression")

    # comparison of model1 and model2 using ROC
    ml_2models_compare_ROC(output_actual=output_test, output_pred_model1=output_test_pred_model1,
                           model1_name="Random Forest Classifier", output_pred_model2=output_test_pred_model2,
                           model2_name="Logistic Regression")


if __name__ == "__main__":
    main()
