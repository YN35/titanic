

class Util():

    def __init__(self) -> None:
        pass

    def data_conv(self, data):
        data = (data > 0.5).astype(int)
        return data