from sys import stdout

import chess

from evaluate import Evaluate


class Limits:
    def __init__(
            self,
            nodes: int,
            depth: int,
            time: int,
    ) -> None:
        self.limited = {"nodes": nodes, "depth": depth, "time0": time}


class Evaluate0:

    def __init__(self):
        pass

    @staticmethod
    def evaluate(board: chess.Board, color: chess.Color) -> float:
        return 0.0

    def evalu(self, board: chess.Board) -> float:
        e = self.evaluate(board, chess.WHITE) - \
             self.evaluate(board, chess.BLACK)
        if board.turn == chess.WHITE:
            return -e
        else:
            return e


class UCI:
    def __init__(self) -> None:
        self.out = stdout
        self.state = chess.Board()
        self.search = None
        self.thread: None
        self.e = Evaluate()

    def output(self, s) -> None:
        self.out.write(str(s) + "\n")
        self.out.flush()

    def stop(self) -> None:
        pass

    def quit(self) -> None:
        pass

    def uci(self) -> None:
        self.output("id name hybridsl")
        self.output("id author Aleksey Belyanin, xayam@yandex.ru")
        self.output("")
        self.output("uciok")

    def isready(self) -> None:
        self.output("readyok")

    def ucinewgame(self) -> None:
        pass

    def eval(self) -> None:
        score = 0.
        self.output(score)

    def process_command(self, inp: str) -> None:
        split = inp.split(" ")
        if split[0] == "quit":
            self.quit()
        elif split[0] == "stop":
            self.stop()
        elif split[0] == "ucinewgame":
            self.ucinewgame()
            # self.search.reset()
        elif split[0] == "uci":
            self.uci()
        elif split[0] == "isready":
            self.isready()
        elif split[0] == "setoption":
            pass
        elif split[0] == "position":
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            move_list = []
            move_idx = inp.find("moves")
            if move_idx >= 0:
                move_list = inp[move_idx:].split()[1:]
            if split[1] == "fen":
                position_idx = inp.find("fen") + len("fen ")
                if move_idx >= 0:
                    fen = inp[position_idx:move_idx]
                else:
                    fen = inp[position_idx:]
            self.state.set_fen(fen)
            for move in move_list:
                self.state.push_uci(move)
        elif split[0] == "print":
            print(self.state)
        elif split[0] == "go":
            self.e.result = {}
            self.e.get_score(engine=self.e.stockfish, state=self.state)
            self.e.get_score(engine=self.e.lczero, state=self.state)
            best = self.e.get_analysis(state=self.state)
            self.output("bestmove " + str(best))
        elif split[0] == "eval":
            self.output(self.e.score)


def main() -> None:
    uciLoop = UCI()

    while True:
        command = input()
        uciLoop.process_command(command)

        if command == "quit":
            break


if __name__ == "__main__":
    main()
