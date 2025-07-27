import chess
import chess.engine


class Evaluate:

    def __init__(self):
        self.result = {}
        self.score = 0.0
        self.stockfish = \
            'D:/Work2/PyCharm/SmartEval2/thirdparty/stockfish17/' + \
            'stockfish-windows-x86-64-avx2.exe'
        self.lczero = \
            'D:/Work2/PyCharm/SmartEval2/thirdparty/lc0-v0.31.2-cpu-dnnl/' + \
            'lc0.exe'
        self.engine = {
            self.stockfish: chess.engine.SimpleEngine.popen_uci(self.stockfish),
            self.lczero: chess.engine.SimpleEngine.popen_uci(self.lczero),
        }

    def get_score(self, engine, state, depth=1):
        variants = self.engine[engine].analyse(
            board=state,
            limit=chess.engine.Limit(time=0.3),
            multipv=len(list(state.legal_moves)),
            game=object()
        )
        for variant in variants:
            if not self.result.__contains__(variant["pv"][0].__str__()):
                self.result[variant["pv"][0].__str__()] = {}
            if state.turn == chess.WHITE:
                score = variant["score"].white()
            else:
                score = variant["score"].black()
            if score.__str__()[0] == "#":
                score = 10 ** 5 - int(score.__str__()[1:])
            else:
                score = int(score.__str__())
            self.result[variant["pv"][0].__str__()][engine] = score
        best = variants[0]["pv"][0]
        return best

    def get_analysis(self, state):
        analysis = {}
        for move in self.result:
            analysis[move] = 3 * self.result[move][self.stockfish]
            analysis[move] += 2 * self.result[move][self.lczero]
            # analysis[move] /= 2
        best = None
        current = - 10 ** 10
        for move in analysis:
            if analysis[move] > current:
                current = analysis[move]
                best = move
        # print(self.result[best][self.stockfish], self.result[best][self.lczero])
        self.score = current
        return best


def main():
    e = Evaluate()
    board = chess.Board()
    while not board.is_game_over():
        e.get_score(engine=e.stockfish, state=board)
        e.get_score(engine=e.lczero, state=board)
        best = e.get_analysis(state=board)
        board.push(chess.Move.from_uci(best))
        print(board)
        print(best)
        e.result = {}
        if board.is_game_over():
            break
        best = e.get_score(engine=e.lczero, state=board)
        board.push(best)
        print(board)
        print(best)
        e.result = {}
    print(board.result())


if __name__ == "__main__":
    main()
