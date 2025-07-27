from sys import stdout
import chess

from h.model.barriers.chess.helpers import ITERATIONS
from h.model.barriers.chess.mctsearch import MCTS


class UCI:
    def __init__(self) -> None:
        self.out = stdout
        self.state = chess.Board()
        self.search = None
        self.thread: None

    def output(self, s) -> None:
        self.out.write(str(s) + "\n")
        self.out.flush()

    def stop(self) -> None:
        pass
        # self.search.STOP = True
        # if self.thread is not None:
        #     try:
        #         self.thread.join()
        #     except:
        #         pass

    def quit(self) -> None:
        pass
        # self.search.STOP = True
        # if self.thread is not None:
        #     try:
        #         self.thread.join()
        #     except:
        #         pass

    def uci(self) -> None:
        self.output("id name tesifaz")
        self.output("id author Aleksey Belyanin, xayam@yandex.ru")
        self.output("")
        self.output("option name Move Overhead type spin default 5 min 0 max 5000")
        self.output("option name Ponder type check default false")
        self.output("uciok")

    def isready(self) -> None:
        self.output("readyok")

    def ucinewgame(self) -> None:
        pass

    def eval(self) -> None:
        mcts = MCTS(state=self.state, iterations=ITERATIONS)
        _, score = mcts.mcts_best()
        self.output(score)

    def process_command(self, inp: str) -> None:
        splitted = inp.split(" ")
        if splitted[0] == "quit":
            self.quit()
        elif splitted[0] == "STOP":
            self.stop()
            self.search.reset()
        elif splitted[0] == "ucinewgame":
            self.ucinewgame()
            self.search.reset()
        elif splitted[0] == "uci":
            self.uci()
        elif splitted[0] == "isready":
            self.isready()
        elif splitted[0] == "setoption":
            pass
        elif splitted[0] == "position":
            # self.search.reset()
            fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
            movelist = []
            move_idx = inp.find("moves")
            if move_idx >= 0:
                movelist = inp[move_idx:].split()[1:]
            if splitted[1] == "fen":
                position_idx = inp.find("fen") + len("fen ")
                if move_idx >= 0:
                    fen = inp[position_idx:move_idx]
                else:
                    fen = inp[position_idx:]
            self.state.set_fen(fen)
            # self.search.hashHistory.clear()
            for move in movelist:
                self.state.push_uci(move)
                # self.search.hashHistory.append(self.search.get_hash(self.state))
        elif splitted[0] == "print":
            print(self.state)
        elif splitted[0] == "go":
            mcts = MCTS(state=self.state, iterations=ITERATIONS)
            bestmove, _ = mcts.mcts_best()
            stdout.write("bestmove " + str(bestmove) + "\n")
            stdout.flush()
        elif splitted[0] == "eval":
            return self.eval()
