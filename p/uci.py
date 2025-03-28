from sys import stdout

import chess
import torch

from p.agent import ChessAgent
from p.enviroment import Enviroment


class UCI:
    def __init__(self) -> None:
        self.out = stdout
        self.state = chess.Board()
        self.history = []
        self.game_data = []
        self.white_agent = ChessAgent(True)
        self.black_agent = ChessAgent(False)
        self.enviroment = Enviroment()
        self.process_command("position moves")

    def output(self, s) -> None:
        self.out.write(str(s) + "\n")
        self.out.flush()

    def stop(self) -> None:
        pass

    def quit(self) -> None:
        pass

    def uci(self) -> None:
        self.output("id name poler")
        self.output("id author Aleksey Belyanin, xayam@yandex.ru")
        self.output("uciok")

    def isready(self) -> None:
        self.output("readyok")

    def ucinewgame(self) -> None:
        self.history = []
        self.game_data = []
        self.process_command("position moves")

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
            current_history = [self.enviroment.board_to_tensor(self.state)]
            for move in move_list:
                self.state.push(chess.Move.from_uci(move))
                self.history.append(self.enviroment.board_to_tensor(self.state))
                self.game_data.append(
                    (current_history, 
                     self.enviroment.board_to_tensor(self.state)))
                current_history = [current_history[0]] + self.history[-100:]
            self.history = current_history[:]
        elif split[0] == "print":
            print(self.state)
        elif split[0] == "go":
            agent = self.white_agent if self.state.turn == chess.WHITE \
                else self.black_agent
            seq = torch.stack(self.history[-100:])
            next_pos = agent.predict_pos(seq)
            best_move = agent.pos2move(self.state, next_pos)
            self.output("bestmove " + str(best_move))
        elif split[0] == "view":
            self.output(self.state)
        # elif split[0] == "eval":
        #     pass
