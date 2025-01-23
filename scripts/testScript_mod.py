import numpy as np
import pandas as pd
from solutionScript import predict

############################### IMPORTANT ###############################
##  DO NOT MODIFY ANY OF THIS CODE!                                    ##
##  Modify the solutionScript.py with the predict() function.          ##
##  This is the code that will be used to test your predictive models. ##
############################### IMPORTANT ###############################

WINDOW_SIZE = 24 * 7
PREDICTION_HORIZON = 24


def main():
    inputData = pd.read_csv("validationData.csv").values
    # The final evaluation will be done just by changing the input data:
    # inputData = pd.read_csv("testData.csv").values

    realValues = []
    predictionValues = []
    for start in range(0, len(inputData) - WINDOW_SIZE - PREDICTION_HORIZON):
        end = start + WINDOW_SIZE
        windowToPredict = inputData[start:end]
        realValues.extend(inputData[:, 2][end : end + PREDICTION_HORIZON])

        predictionValues.extend(predict(windowToPredict))
        #print(realValues)
        #print(predictionValues)
        print(len(inputData) - WINDOW_SIZE - PREDICTION_HORIZON, len(realValues), len(predictionValues), end="\n")

    # Compute the RMSE (root mean squared error).
    rmse = np.nanmean(np.subtract(realValues, predictionValues) ** 2) ** 0.5

    print("Final prediction mean RMSE score:", rmse)


if __name__ == "__main__":
    main()