import json
import math
import os.path
import sys

import torch.nn

import chess
import chess.engine
import chess.syzygy

from kan import *

from h.model.utils import utils_print


class Model:

    def __init__(self):
        self.sf = None
        self.job = None
        self.params = None
        self.listener = None
        self.process = None
        self.thread = None
        self.commands = None
        self.stop = None
        self.random = None
        self.model = None
        self.dataset = {}
        self.last_fen = None
        self.file_model = None
        self.file_formula1 = None
        self.file_formula2 = None
        self.pre_model_json = None
        self.model_json = None
        self.model_option = None
        self.lib_formula = None
        self.engine_stockfish = None
        self.syzygy_endgame = None
        self.epd_eval = None
        self.len_input = None
        self.count_limit = None
        self.device = None
        self.dtype = None
        self.formula1 = None

        self.model_config()

    def model_config(self):
        self.engine_stockfish = \
            'D:/Work2/PyCharm/SmartEval2/github/src/healers/healers/dist' + \
            '/stockfish-windows-x86-64-avx2.exe'
        self.syzygy_endgame = {
            "wdl345": "E:/Chess/syzygy/3-4-5-wdl",
            "wdl6": "E:/Chess/syzygy/6-wdl",
        }
        self.commands = {
            0: {"call": None, "desc": "Exit"},
            1: {"call": self.model_params, "desc": "Search hyperparameters"},
            2: {"call": self.model_finetune, "desc": "Fine-Tune model"},
            3: {"call": self.model_formula, "desc": "Save formula"},
            4: {"call": self.model_test, "desc": "Test model"},
            5: {"call": self.model_predict, "desc": "Make predict"},
        }
        self.stop = False
        self.random = random.SystemRandom(0)
        self.model_option = {
            "len_input": 64 * 12 + 4,
            "hidden_layers": [5, 5, 5],
            "len_output": 1,
            "grid": 5,
            "k": 3,
        }
        self.file_model = "model2.pth"
        self.file_formula1 = "model2_formula1.txt"
        self.file_formula2 = "model2_formula2.txt"
        self.pre_model_json = "pre_model2.json"
        self.model_json = "model2.json"
        self.lib_formula = [
            'x', 'x^2', 'x^3', 'x^4', 'exp',
            'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs',
        ]
        self.epd_eval = "dataset.epdeval"
        # self.count_limit = 48
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = torch.get_default_dtype()

        print(str(self.device).upper(), self.dtype)

        self.formula1= self.model_load()

    def start(self):
        while True:
            print("Available commands:")
            for key, value in self.commands.items():
                print(f"   {key}. {value['desc']}")
            try:
                command = int(input("Input command [default 0]: "))
            except ValueError:
                command = 0
            if command not in self.commands.keys():
                print("Error: Command not found")
                continue
            if command == 0:
                break
            self.job = self.commands[command]["call"]
            self.params = {}
            self.job()

    def model_save(self):
        torch.save(self.model.state_dict(), self.file_model)
        d = self.dataset["train_input"].tolist()
        tl = self.dataset["train_label"].tolist()
        for i in range(len(d)):
            d[i].append(tl[i])
        with open("dataset.json", "w") as f:
            json.dump(d, f)

    def model_formula(self):
        utils_print("self.model_formula() starting...")
        self.model_finetune(save_formula=True)

    def model_load(self):
        # print("Loading model...")
        if os.path.exists(self.model_json):
            with open(self.model_json, "r") as f:
                self.model_option = json.load(f)
        else:
            with open(self.model_json, "w") as f:
                json.dump(self.model_option, f)
        self.model = KAN(
            width=[
                self.model_option["len_input"],
                *self.model_option["hidden_layers"],
                self.model_option["len_output"]
            ],
            grid=self.model_option["grid"],
            k=self.model_option["k"], auto_save=False, seed=0,
            noise_scale=0.1,
            sp_trainable=False,
            sb_trainable=False,
        )
        if os.path.exists(self.file_model):
            self.model.load_state_dict(torch.load(self.file_model))
        if not os.path.exists(self.file_formula1):
            return None
        else:
            with open(self.file_formula1, encoding="UTF-8", mode="r") as p:
                f1 = str(p.read()).strip()
            return f1

    def model_finetune(self, save_formula=False):
        if not save_formula:
            utils_print("self.model_finetune() starting...")
        count = 0
        while True:
            count += 1
            utils_print(count)
            self.get_dataset(nums=1000, epoch=1000)
            utils_print(self.dataset["train_input"].shape)
            utils_print(self.dataset["train_label"].shape)
            utils_print(self.dataset["test_input"].shape)
            utils_print(self.dataset["test_label"].shape)
            result = self.model.fit(
                self.dataset,
                opt="LBFGS",
                # loss_fn=self.loss_fn,
                lamb=0.001,
                steps=5,
                update_grid=False,
                metrics=(
                     self.train_acc,
                     self.test_acc
                )
            )
            utils_print(result['train_acc'][-1], result['test_acc'][-1])
            model.model_save()
            self.dataset = {}
            if save_formula:
                self.model.auto_symbolic(lib=self.lib_formula)
                formula1 = self.model.symbolic_formula()[0][0]
                with open(self.file_formula1, encoding="UTF-8", mode="w") as p:
                    p.write(str(formula1).strip())
                break
            self.model_load()

    def get_best(self, state):
        moves = list(state.legal_moves)
        best_move = None
        best_value = - 10 ** 10
        result_scores = []
        result_moves = []
        for move in moves:
            state.push(move)
            inputs = torch.FloatTensor(
                [self.get_input(state)]
            ).type(self.dtype).to(self.device)
            score = self.model(inputs).detach().tolist()[0][0]
            print(state)
            print(score)
            result_scores.append(score)
            result_moves.append(move)
            state.pop()
        ##################
        # valid = []
        # valid_count = 0
        # for score in range(len(result_scores)):
        #     count = result_scores.count(result_scores[score])
        #     if count == 1:
        #         valid_count += 1
        #         valid.append(result_scores[score])
        #     else:
        #         valid.append(0)
        # result = []
        # for score in range(len(result_scores)):
        #     result.append([
        #         str(result_scores[score])[2:].ljust(18, "0"),
        #         result_moves[score].uci()
        #     ])
        #     print(result[-1][0], result[-1][1])
        # sys.exit()
        ##################
        mean_score = max(result_scores)  # / len(result_scores)
        delta = 10 ** 10
        best_move = None
        for score in range(len(result_scores)):
            if abs(result_scores[score] - mean_score) < delta:
                delta = abs(result_scores[score] - mean_score)
                best_move = result_moves[score]
        return best_move

    def get_data(self, nums=1000, epoch=1000):
        result_train = []
        result_label = []
        board = chess.Board()
        for num in range(nums):
            result = 0.
            board_copy = board.copy()
            inputs = self.get_input(board)
            for ep in range(epoch):
                while not board_copy.is_game_over():
                    moves = list(board_copy.legal_moves)
                    best_move = self.random.choice(moves)
                    # best_move = self.get_best(board_copy)
                    board_copy.push(best_move)
                if board_copy.result() == "1-0":
                    result += 1.
                elif board_copy.result() == "0-1":
                    result += - 1.
                else:
                    result += .001
            result_train.append(inputs)
            label = result / epoch
            result_label.append(label)
            if board.is_game_over():
                board = chess.Board()
            else:
                moves = list(board.legal_moves)
                best_move = self.random.choice(moves)
                board.push(best_move)
        return result_train, result_label

    def get_dataset(self, nums=1000, epoch=1000):
        train_inputs, train_labels = self.get_data(nums=nums, epoch=epoch)
        self.dataset['train_input'] = \
            torch.FloatTensor(train_inputs).type(self.dtype).to(self.device)
        self.dataset['train_label'] = \
            torch.FloatTensor(train_labels).type(self.dtype).to(self.device)
        test_inputs, test_labels = self.get_data(nums=nums, epoch=epoch)
        # test_inputs, test_labels = train_inputs, train_labels
        self.dataset['test_input'] = \
            torch.FloatTensor(test_inputs).type(self.dtype).to(self.device)
        self.dataset['test_label'] = \
            torch.FloatTensor(test_labels).type(self.dtype).to(self.device)

    def model_params(self):
        print("self.model_params() starting...")
        self.get_dataset()
        print(self.dataset['train_input'].shape)
        print(self.dataset['test_input'].shape)
        if os.path.exists(self.pre_model_json):
            with open(self.pre_model_json, "r") as f:
                self.model_option = json.load(f)
        hidden_layer1 = self.model_option["hidden_layer"]
        grid1 = self.model_option["grid"]
        k1 = self.model_option["k"]
        maxi = 10 ** 10
        maximum_layer = 10 ** 10
        maximum_grid = 10 ** 10
        maximum_k = 10 ** 10
        while True:
            self.model = KAN(
                width=[self.len_input, *hidden_layer1, 1],
                grid=grid1, k=k1, auto_save=False, seed=0)
            result = self.model.fit(
                self.dataset,
                opt="LBFGS",
                steps=5,
                metrics=(
                    self.train_acc,
                    self.test_acc
                )
            )
            if result['test_acc'][-1] < maxi:
                maxi = result['test_acc'][-1]
                maximum_layer = hidden_layer1
                maximum_grid = grid1
                maximum_k = k1
                with open(self.pre_model_json, "w") as f:
                    data = {"hidden_layer": maximum_layer,
                            "grid": maximum_grid,
                            "k": maximum_k}
                    json.dump(data, f)

            print(result['train_acc'], result['test_acc'])
            print(f"hidden_layer={maximum_layer}, grid={maximum_grid}, " +
                  f"k={maximum_k}, maxi_test_acc={maxi}")
            print(f"hidden_layer={hidden_layer1}, grid={grid1}, " +
                  f"k={k1}, test_loss={result['test_loss'][0]}")
            # if self.stop:
            #     self.model_save()
            break
            # hidden_layer1 = self.random.choice(list(range(5, 101)))
            # grid1 = self.random.choice(list(range(5, 51)))
            # k1 = self.random.choice(list(range(3, 26)))

    def loss_fn(self, x, y):
        return torch.max(torch.abs(x - y))

    def train_acc(self):
        return torch.mean(
            torch.abs(
                    self.model(self.dataset['train_input']) -
                    self.dataset['train_label']
            ).type(self.dtype)
        )

    def test_acc(self):
        return torch.mean(
            torch.abs(
                    self.model(self.dataset['test_input']) -
                    self.dataset['test_label']
            ).type(self.dtype)
        )

    @staticmethod
    def get_input(state):
        train_input = [[0.] * 64 for _ in range(12)]
        for piece in chess.PIECE_TYPES:
            for square in state.pieces(piece, chess.BLACK):
                train_input[piece - 1][square] = -piece
                for move in state.pseudo_legal_moves:
                    if move.from_square == square:
                        train_input[piece - 1][move.to_square] = -piece
        for piece in chess.PIECE_TYPES:
            for square in state.pieces(piece, chess.WHITE):
                train_input[piece + 5][square] = piece
                for move in state.pseudo_legal_moves:
                    if move.from_square == square:
                        train_input[piece + 5][move.to_square] = piece
        train_input = [j for i in train_input for j in i]
        if state.has_kingside_castling_rights(state.turn):
            train_input = [1.] + train_input
        else:
            train_input = [0.] + train_input
        if state.has_kingside_castling_rights(state.turn):
            train_input = [1.] + train_input
        else:
            train_input = [0.] + train_input
        if state.ep_square is None:
            train_input = [0.] + train_input
        else:
            train_input = [state.ep_square] + train_input
        train_input = [int(state.turn)] + train_input
        return train_input

    def model_test(self):
        print("self.model_test() starting...")
        count_win = 0
        count_draw = 0
        count_loss = 0
        for _ in range(1):
            board = chess.Board()
            while not board.is_game_over():
                best_move = self.get_best(board)
                board.push(best_move)
                print(board)
                print("")
                if board.is_game_over():
                    break
                moves = list(board.legal_moves)
                best_move = self.random.choice(moves)
                board.push(best_move)
                print(board)
                print(
                    f"count_win={count_win}, " +
                    f"count_loss={count_loss}, count_draw={count_draw}"
                )
                print("")
            if board.result() == "1-0":
                count_win += 1
            elif board.result() == "0-1":
                count_loss += 1
            else:
                count_draw += 1

    def model_predict(self):
        print("self.model_predict() starting...")
        # fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board = chess.Board()
        # board.set_fen(fen_start)
        print(board)
        formula0 = self.model_load()
        while not board.is_game_over():
            moves = list(board.legal_moves)
            index = 0
            bestmove = index
            score = - 10 ** 10
            for move in moves:
                board.push(move)
                inputs = self.get_input(state=board)
                variable_values = {
                    f"x_{i}": inputs[i - 1]
                    for i in range(self.model_option["len_input"], 0, -1)
                }
                formula = str(formula0)[:]
                # print(utils.ex_round(formula, 4))
                for _var, _val in variable_values.items():
                    formula = str(formula).replace(_var, str(_val))
                evaluate1 = eval(formula)
                # print(move.uci(), evaluate1)
                if evaluate1 > score:
                    bestmove = index
                    score = evaluate1
                index += 1
                board.pop()
            board.push(moves[bestmove])
            print(board)
            print(score)
            if board.is_game_over():
                break
            moves = list(board.legal_moves)
            bestmove = self.random.choice(moves)
            board.push(bestmove)
            print(board)
            print("")
        print(board.result())


if __name__ == "__main__":
    model = Model()
    model.start()
