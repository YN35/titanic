from sklearn.metrics import accuracy_score

class Util():

    def __init__(self) -> None:
        pass

    def data_conv(self, data):
        data = (data > 0.5).astype(int)
        return data


    def accuracy_score(self, train, predict):
        """
        どのくらい答えに近いか評価するスコアを出す
        あっているほど数値が高いようにする
        コンペによって評価方法が違うからこれを変える
        """
        return accuracy_score(train, predict)
