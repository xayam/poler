import json
import os.path
# import sys
# from multiprocessing import Process
# from time import sleep
# import keyboard
import concurrent.futures as pool
import random
import sys
from time import sleep

import torch.nn
from pynput import keyboard
import chess
import chess.engine
import chess.syzygy

from kan import *

from h.model.utils import utils_progress, utils_print
import regmet


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
        self.formula2 = None

        self.model_config()

    def model_config(self):
        # Set this
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
            3: {"call": self.save_formula, "desc": "Save formula"},
            4: {"call": self.model_test, "desc": "Test model"},
            5: {"call": self.make_predict, "desc": "Make predict"},
        }
        self.stop = False
        self.random = random.SystemRandom(0)
        self.model_option = {
            "hidden_layer": [3, 3, 3],
            "grid": 5,
            "k": 3,
        }
        self.file_model = "model.pth"
        self.file_formula1 = "model_formula1.txt"
        self.file_formula2 = "model_formula2.txt"
        self.pre_model_json = "pre_model.json"
        self.model_json = "model.json"
        self.lib_formula = [
            'x', 'x^2', 'x^3', 'x^4', 'exp',
            'log', 'sqrt', 'tanh', 'sin', 'tan', 'abs',
        ]
        self.epd_eval = "dataset.epdeval"
        self.len_input = 65
        self.count_limit = 48
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = torch.get_default_dtype()

        print(str(self.device).upper(), self.dtype)

        self.formula1, self.formula2 = self.model_load()

    def loss_function(self, xx, yy):
        return torch.mean(
            yy - xx, dtype=self.dtype
        )

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
        # self.process = Process(target=self.executor)
        # self.listener = keyboard.Listener(on_release=self.key_release)
        # self.listener.start()
        # self.process.start()

    def key_release(self, key):
        if key == keyboard.Key.esc:
            print(key)
            self.stop = True
            return False

    def executor(self):
        if self.job is not None and self.params is not None:
            with pool.ThreadPoolExecutor(max_workers=1) as execute:
                execute.submit(self.job, **self.params)
                execute.shutdown()

    def model_save(self):
        torch.save(self.model.state_dict(), self.file_model)
        d = self.dataset["train_input"].tolist()
        tl = self.dataset["train_label"].tolist()
        for i in range(len(d)):
            d[i].append(tl[i])
        with open("dataset.json", "w") as f:
            json.dump(d, f)

    def save_formula(self):
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
            width=[self.len_input, *self.model_option["hidden_layer"], 65],
            grid=self.model_option["grid"],
            k=self.model_option["k"], auto_save=False, seed=0,
            noise_scale=0.1,
            sp_trainable=False,
            sb_trainable=False,
        )
        if os.path.exists(self.file_model):
            self.model.load_state_dict(torch.load(self.file_model))
        if not os.path.exists(self.file_formula1) or \
           not os.path.exists(self.file_formula2):
            return None, None
        else:
            with open(self.file_formula1, encoding="UTF-8", mode="r") as p:
                f1 = str(p.read()).strip()
            with open(self.file_formula2, encoding="UTF-8", mode="r") as p:
                f2 = str(p.read()).strip()
            return f1, f2

    def model_finetune(self, save_formula=False):
        if not save_formula:
            utils_print("self.model_finetune() starting...")
        count = 0

        while True:
            count += 1
            utils_print(count)
            self.load_data()
            utils_print(self.dataset["train_input"].shape)
            utils_print(self.dataset["test_input"].shape)
            result = self.model.fit(
                self.dataset,
                opt="LBFGS",
                # loss_fn=torch.nn.,
                lamb=0.001,
                steps=20,
                update_grid=False,
                metrics=(
                     self.train_acc,
                     self.test_acc
                )
            )
            utils_print("")
            utils_print(result['train_acc'][-1], result['test_acc'][-1])
            model.model_save()
            self.dataset = {}
            if save_formula:
                self.model.auto_symbolic(lib=self.lib_formula)
                formula1 = self.model.symbolic_formula()[0][0]
                with open(self.file_formula1, encoding="UTF-8", mode="w") as p:
                    p.write(str(formula1).strip())
                break
            # if count == 10:
            #     break

    def acc(self, formula1, formula2, X, y):
        batch = X.shape[0]
        correct = 0
        for i in range(batch):
            logit1 = np.array(
                formula1.subs('x_1', X[i, 0]).
                subs('x_2', X[i, 1]).
                subs('x_3', X[i, 2]).
                subs('x_4', X[i, 3]).
                subs('x_5', X[i, 4])
            ).astype(
                np.float32)
            logit2 = np.array(
                formula2.subs('x_1', X[i, 0]).
                subs('x_2', X[i, 1]).
                subs('x_3', X[i, 2]).
                subs('x_4', X[i, 3]).
                subs('x_5', X[i, 4])
            ).astype(
                np.float32)
            correct += (logit2 < logit1) == y[i]
        return correct / batch

    def get_data_rook(self):
        square_to_index = [
            56, 57, 58, 59, 60, 61, 62, 63,
            48, 49, 50, 51, 52, 53, 54, 55,
            40, 41, 42, 43, 44, 45, 46, 47,
            32, 33, 34, 35, 36, 37, 38, 39,
            24, 25, 26, 27, 28, 29, 30, 31,
            16, 17, 18, 19, 20, 21, 22, 23,
             8,  9, 10, 11, 12, 13, 14, 15,
             0,  1,  2,  3,  4,  5,  6,  7
        ]
        dataset_input = []
        dataset_label = []
        board = chess.Board()
        count = 0
        while True:
            fen = ["1"] * 64
            random_arr = [i for i in range(64)]
            bk = self.random.choice(random_arr)
            fen[bk] = "k"
            wk = self.random.choice(random_arr)
            fen[wk] = "K"
            wr = self.random.choice(random_arr)
            fen[wr] = "R"
            if bk == wk or bk == wr or wk == wr:
                continue
            output = ""
            for i in range(8):
                output += "".join(fen[i * 8: i * 8 + 8]) + "/"
            color = self.random.choice(["w", "b"])
            output = output[:-1] + " " + color + " - - 0 1"
            output = output.replace("11111111", "8")
            output = output.replace("1111111", "7")
            output = output.replace("111111", "6")
            output = output.replace("11111", "5")
            output = output.replace("1111", "4")
            output = output.replace("111", "3")
            output = output.replace("11", "2")
            try:
                board.set_fen(output)
            except:
                continue
            c = 0
            while not board.is_game_over():
                board1 = self.board_to_input65(board)
                move = self.random.choice(list(board.legal_moves))
                board.push(move)
                board2 = self.board_to_input65(board)
                dataset_input.append(board1)
                dataset_label.append(board2)
                c += 1
                if c > 100:
                    break
            count += 1
            if count > 100:
                break
        return dataset_input, dataset_label

    def board_to_input65(self, board):
        input65 = [0.0 for _ in range(65)]
        for sq in range(1, 65):
            piece = board.piece_at(sq - 1)
            if piece is None:
                continue
            if piece.color == chess.BLACK and piece.piece_type == chess.KING:
                input65[sq] = 0.25
            if piece.color == chess.WHITE and piece.piece_type == chess.KING:
                input65[sq] = 0.5
            if piece.color == chess.WHITE and piece.piece_type == chess.ROOK:
                input65[sq] = 0.75
        input65[0] = float(int(board.turn))
        return input65

    def load_data(self):
        train_inputs, train_labels = self.get_data_rook()
        self.dataset['train_input'] = \
            torch.FloatTensor(train_inputs).type(self.dtype).to(self.device)
        self.dataset['train_label'] = \
            torch.FloatTensor(train_labels).type(self.dtype).to(self.device)
        test_inputs, test_labels = self.get_data_rook()
        self.dataset['test_input'] = \
            torch.FloatTensor(test_inputs).type(self.dtype).to(self.device)
        self.dataset['test_label'] = \
            torch.FloatTensor(test_labels).type(self.dtype).to(self.device)

    def model_params(self):
        print("self.model_params() starting...")
        self.load_data()
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

    def train_acc(self):
        return torch.mean((torch.round(self.model(self.dataset['train_input'])) ==
                           self.dataset['train_label']).type(self.dtype))

    def test_acc(self):
        return torch.mean((torch.round(self.model(self.dataset['test_input'])) ==
                           self.dataset['test_label']).type(self.dtype))

    def get_train(self, state1, state2):
        return self.get_input(state1) + self.get_input(state2)

    def get_input(self, state):
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
        return train_input[:self.len_input // 2]

    def get_data(self, fen_generator, get_score, count_limit):
        count = 0
        count2 = 0
        dataset = {}
        train_inputs = []
        train_labels = []
        test_inputs = []
        test_labels = []
        board = chess.Board()
        for fen in fen_generator(get_score, count_limit):
            scores = []
            boards = []
            try:
                board.set_fen(fen)
                score = get_score(board)
                if score is None:
                    continue
            except chess.engine.EngineError:
                continue
            except chess.IllegalMoveError:
                continue
            scores.append(score)
            boards.append(board.copy())
            count += 1
            moves = board.legal_moves
            for move in moves:
                try:
                    board.push(move)
                    score = get_score(board)
                    if score is None:
                        board.pop()
                        continue
                except chess.engine.EngineError:
                    break
                except chess.IllegalMoveError:
                    break
                scores.append(score)
                boards.append(board.copy())
                count2 += 1
                utils_progress(
                    f"{str(count).rjust(4, ' ')} | " +
                    f"{str(count2).rjust(4, ' ')} | " +
                    f"{str(scores[-1]).rjust(5, ' ')} " +
                    f"| {board.fen()}")
                board.pop()
            if count % 2 == 0:
                for i in range(1, len(boards)):
                    test_input = self.get_train(
                        state1=boards[0], state2=boards[i]
                    )
                    test_inputs.append(test_input)
                    test_labels.append([scores[i] - scores[0]])
                    test_input = self.get_train(
                        state1=boards[i], state2=boards[0]
                    )
                    test_inputs.append(test_input)
                    test_labels.append([scores[0] - scores[i]])
            else:
                for i in range(1, len(boards)):
                    train_input = self.get_train(
                        state1=boards[0], state2=boards[i]
                    )
                    train_inputs.append(train_input)
                    train_labels.append([scores[i] - scores[0]])
                    train_input = self.get_train(
                        state1=boards[i], state2=boards[0]
                    )
                    train_inputs.append(train_input)
                    train_labels.append([scores[0] - scores[i]])
        # print()
        # min_len = min(len(test_inputs), len(train_inputs),
        #               len(test_labels), len(train_labels),
        #               )
        # test_inputs = test_inputs[:min_len]
        # test_labels = test_labels[:min_len]
        # train_inputs = train_inputs[:min_len]
        # train_labels = train_labels[:min_len]
        dataset['train_input'] = \
            torch.FloatTensor(train_inputs).type(self.dtype).to(self.device)
        dataset['train_label'] = \
            torch.FloatTensor(train_labels).type(self.dtype).to(self.device)
        dataset['test_input'] = \
            torch.FloatTensor(test_inputs).type(self.dtype).to(self.device)
        dataset['test_label'] = \
            torch.FloatTensor(test_labels).type(self.dtype).to(self.device)
        return dataset

    def get_wdl(self, fen_position):
        with chess.syzygy.open_tablebase(
                self.syzygy_endgame["wdl345"]
        ) as tablebase:
            board = chess.Board(fen_position)
            result = tablebase.get_wdl(board)
        if result is None:
            with chess.syzygy.open_tablebase(
                    self.syzygy_endgame["wdl6"]
            ) as tablebase:
                board = chess.Board(fen_position)
                result = tablebase.get_wdl(board)
        return result

    def set_piece(self, state, piece):
        while True:
            pos = self.random.choice(list(range(64)))
            row, col = divmod(pos, 8)
            sq = chess.square(col, row)
            if state.piece_at(sq) is not None:
                continue
            state.set_piece_at(sq, chess.Piece.from_symbol(piece))
            break
        return state

    def get_score(self, state, depth=10):
        with chess.engine.SimpleEngine.popen_uci(self.engine_stockfish) as sf:
            result = sf.analyse(
                state,
                chess.engine.Limit(depth=depth),
            )
            score = result['score'].white().score()
            return score

    def get_best(self, state, depth=20):
        self.sf = chess.engine.SimpleEngine.popen_uci(
            self.engine_stockfish
        )
        variants = self.sf.analyse(
            board=state,
            limit=chess.engine.Limit(depth=depth),
            multipv=1,
            game = object()
        )
        bestmove = variants[0]["pv"][0]
        self.sf.quit()
        return bestmove

    def get_fen_epd(self, get_score, count_limit):
        with open(self.epd_eval, mode="r") as f:
            dataevals = f.readlines()
        fens = []
        for _ in range(count_limit):
            dataeval = str(self.random.choice(dataevals)).strip()
            spl = dataeval.split(" ")
            fen = " ".join(spl[:-1])
            fens.append(fen)
        return fens

    def get_fen_random(self, get_score, count_limit):
        board = chess.Board()
        count = 0
        endgames = []
        pieces = ['P', 'p', 'N', 'n', 'B', 'b', 'R', 'r', 'Q', 'q']
        for _ in range(count_limit):
            board.clear()
            for king in ['K', 'k']:
                board = self.set_piece(state=board, piece=king)
            c = self.random.choice([1, 2, 3, 4])
            for _ in range(c):
                piece = self.random.choice(pieces)
                board = self.set_piece(state=board, piece=piece)
            board.turn = chess.WHITE
            fen_positions = board.fen()
            if board.is_valid() and fen_positions not in endgames:
                if get_score(fen_positions) is not None:
                    count += 1
                    endgames.append(fen_positions)
            board.turn = chess.BLACK
            fen_positions = board.fen()
            if board.is_valid() and fen_positions not in endgames:
                if get_score(fen_positions) is not None:
                    count += 1
                    endgames.append(fen_positions)
            board.turn = chess.WHITE
            for s in [chess.A6, chess.B6, chess.C6, chess.D6,
                      chess.E6, chess.F6, chess.G6, chess.H6]:
                board.ep_square = s
                fen_positions = board.fen()
                if board.is_valid() and fen_positions not in endgames:
                    if get_score(fen_positions) is not None:
                        count += 1
                        endgames.append(fen_positions)
            board.turn = chess.BLACK
            for s in [chess.A3, chess.B3, chess.C3, chess.D3,
                      chess.E3, chess.F3, chess.G3, chess.H3]:
                board.ep_square = s
                fen_positions = board.fen()
                if board.is_valid() and fen_positions not in endgames:
                    if get_score(fen_positions) is not None:
                        count += 1
                        endgames.append(fen_positions)
            utils_progress(f"{count}/{count_limit} | {fen_positions}")
        return endgames

    def model_test(self):
        print("self.model_test() starting...")
        self.load_data()
        y_pred = self.model(self.dataset['test_input'])
        for i in range(len(self.dataset["test_input"])):
            print(
                y_pred[i], self.dataset["test_label"][i]
            )
        # formula1, formula2 = self.model.symbolic_formula()[0]
        # print('train acc of the formula:',
        #       self.acc(
        #           formula1, formula2, self.dataset['train_input'],
        #           self.dataset['train_label'])
        #       )
        # print('test acc of the formula:',
        #       self.acc(
        #           formula1, formula2, self.dataset['test_input'],
        #           self.dataset['test_label'])
        #       )
        # y_true = self.dataset["train_label"]
        # variables = []
        # for data in self.dataset["train_input"]:
        #     variables.append({
        #         f"x_{i}": data[i - 1].numpy().item(0)
        #         for i in range(self.len_input, 0, -1)
        #     })
        # y_pred = []
        # for data in variables:
        #     formula1 = str(self.formula1)
        #     for key, value in data.items():
        #         formula1 = str(formula1).replace(key, str(value))
            # formula2 = str(self.formula2)
            # for key, value in data.items():
            #     formula2 = str(formula2).replace(key, str(value))
            # y_pred.append(eval(formula1))
        # print(y_pred)
        #
        # regmet.RegressionMetrics(y_true, y_pred)

    def make_predict(self):
        print("self.model_predict() starting...")
        fen_start = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        board1 = chess.Board()
        board1.set_fen(fen_start)
        print(board1)
        while True:
            moves = list(board1.legal_moves)
            index = 0
            bestmove = index
            score = 10 ** 6
            for move in moves:
                board1.push(move)
                board2 = chess.Board()
                board2.set_fen(board1.fen())
                board1.pop()
                inputs = self.get_train(state1=board1, state2=board2)
                variable_values = {
                    f"x_{i}": inputs[i - 1]
                    for i in range(self.len_input, 0, -1)
                }
                formula = self.model_load()
                for _var, _val in variable_values.items():
                    formula = str(formula).replace(_var, str(_val))
                evaluate = eval(formula)
                if evaluate < score:
                    bestmove = index
                    score = evaluate
                index += 1
            board1.push(moves[bestmove])
            print(board1)
            print("")
            if board1.is_game_over():
                break
            board1.push(random.choice(list(board1.legal_moves)))
            print(board1)
            print("")
            if board1.is_game_over():
                break
        print(board1.result())


if __name__ == "__main__":
    model = Model()
    model.start()
