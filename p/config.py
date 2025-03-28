

class Config:

    def __init__(self):
        self.SEQ_LENGTH = 100
        self.engine_stockfish = \
            'D:/Work2/PyCharm/SmartEval2/github/src/healers/healers/dist' + \
            '/stockfish17-windows-x86-64-avx2.exe'
        self.white_model_name = "white.pth"
        self.black_model_name = "black.pth"
