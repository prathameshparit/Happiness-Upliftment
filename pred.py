import joblib
import numpy as np

model = joblib.load('models/regressor.sav')


def Pred_regressor(inp):
    arr = np.array([inp])
    pred = model.predict(arr)

    return pred


loaded_model = joblib.load('models/classifier.sav')


def Pred_classifier(inp):
    pred_reg = loaded_model.predict(np.array([inp]))
    return pred_reg


# if __name__ == '__main__':
#     arr = [1, 3, 1, 2, 1, 1, 1, 4, 0, 0, 0, 2, 1, 1]
#     res = Pred_regressor(arr)
#     res_2 = Pred_classifier(arr)
#     print(res[0])
#     print(res_2)
