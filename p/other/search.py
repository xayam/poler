import time
import tt as tt
from evaluation import evaluate
import psqt as pqst

# External
import chess
import chess.polyglot
from helpers import *
from limits import *
from sys import stdout


class Search:
    def __init__(self) -> None:
        self.t0 = 0
        # This is our transposition table, it stores positions
        # it is one of the most important parts of a chess engine.
        # It stores results of previously performed searches and it
        # allows to skip parts of the _search tree and order moves.
        self.transposition_table = tt.TranspositionTable()
        self.pvLength = [0] * MAX_PLY
        # This is our principal variation table, it stores the best
        # moves for each depth. It is used to print the best line
        # after the _search is completed.
        self.pvTable = [[chess.Move.null()] * MAX_PLY for _ in range(MAX_PLY)]
        # Total nodes searched
        self.nodes = 0
        # Current limits for the _search
        self.limit = Limits(0, MAX_PLY, 0)
        # True when the _search is stopped/aborted
        self.stop = False
        # Time checking is expensive and we dont want to do it every node
        self.checks = CHECK_RATE
        # Keeps track of the zobrist hashes encountered during the _search
        # Used to efficiently detect repetitions
        self.hashHistory = []
        # History Table
        # Indexed by [color][from][to]
        self.htable = [[[0 for _ in range(64)]
                        for _ in range(64)] for _ in range(2)]

    def q_search(self, state, alpha: int, beta: int, ply: int) -> int:
        """
        Quiescence Search, this is a special _search that only searches
        captures and checks. It is needed to avoid the horizon effect.
        We will continue to _search until we reach a quiet position.
        """
        if self.stop or self.check_time():
            return 0
        # Don't _search higher than MAX_PLY
        if ply >= MAX_PLY:
            return evaluate(state)
        # staticEval
        bestValue = evaluate(state)
        if bestValue >= beta:
            return bestValue
        if bestValue > alpha:
            alpha = bestValue
        # Sort the moves, the highest score should come first,
        # to reduce the size of the _search tree
        moves = sorted(
            state.generate_legal_captures(),
            key=lambda m: self.score_qmove(state, m),
            reverse=True,
        )
        # Loop over all legal captures
        for move in moves:
            self.nodes += 1
            captured = state.piece_type_at(move.to_square)
            # Delta Pruning
            if (
                pqst.piece_values[captured] + 400 + bestValue < alpha
                and not move.promotion
            ):
                continue
            # Make move
            state.push(move)
            score = -self.q_search(state, -beta, -alpha, ply + 1)
            # Unmake move
            state.pop()
            # We found a new best value
            if score > bestValue:
                bestValue = score
                if score > alpha:
                    alpha = score
                    if score >= beta:
                        break
        return bestValue

    def ab_search(self, state: chess.Board,
                  alpha: int, beta: int, depth: int, ply: int) -> int:
        """
        Alpha Beta Search, this is the main _search function.
        It searches the tree recursively and returns the best score.
        This function will be called with increasing depth until
        the time0 count_limit is reached or the maximum depth is reached.
        """
        if self.check_time():
            return 0
        # Don't search higher than MAX_PLY
        if ply >= MAX_PLY:
            return evaluate(state)
        self.pvLength[ply] = ply
        RootNode = ply == 0
        hashKey = self.get_hash(state)
        if not RootNode:
            if self.is_repetition(state, hashKey):
                # slight draw bias
                return -5
            if state.halfmove_clock >= 100:
                return 0
            # Mate distance pruning
            alpha = max(alpha, mated_in(ply))
            beta = min(beta, mate_in(ply + 1))
            if alpha >= beta:
                return alpha
        # Jump into q_search
        if depth <= 0:
            return self.q_search(state, alpha, beta, ply)
        # Transposition Table probing
        tte = self.transposition_table.probeEntry(hashKey)
        ttHit = hashKey == tte.key
        ttMove = tte.move if ttHit else chess.Move.null()
        # Adjust score
        ttScore = (
            self.transposition_table.scoreFromTT(tte.score, ply)
            if ttHit
            else VALUE_NONE
        )
        if not RootNode and tte.depth >= depth and ttHit:
            if tte.flag == tt.Flag.LOWERBOUND:
                alpha = max(alpha, ttScore)
            elif tte.flag == tt.Flag.UPPERBOUND:
                beta = min(beta, ttScore)
            if alpha >= beta:
                return ttScore
        inCheck = state.is_check()
        # Null move pruning
        if depth >= 3 and not inCheck:
            state.push(chess.Move.null())
            score = -self.ab_search(state, -beta, -beta + 1, depth - 2, ply + 1)
            state.pop()
            if score >= beta:
                if score >= VALUE_TB_WIN_IN_MAX_PLY:
                    score = beta
                return score
        oldAlpha = alpha
        bestScore = -VALUE_INFINITE
        bestMove = chess.Move.null()
        madeMoves = 0
        # Sort the moves, the highest score should come first
        # The ttmove should be first one searched, incase we have a hit
        moves = sorted(
            state.legal_moves,
            key=lambda m: self.score_move(state, m, ttMove),
            reverse=True,
        )
        for move in moves:
            madeMoves += 1
            self.nodes += 1
            # Make move
            state.push(move)
            self.hashHistory.append(hashKey)
            # Search
            score = -self.ab_search(state, -beta, -alpha, depth - 1, ply + 1)
            # Unmake move
            state.pop()
            self.hashHistory.pop()
            if score > bestScore:
                bestScore = score
                bestMove = move
                # update PV
                self.pvTable[ply][ply] = move
                for i in range(ply + 1, self.pvLength[ply + 1]):
                    self.pvTable[ply][i] = self.pvTable[ply + 1][i]
                self.pvLength[ply] = self.pvLength[ply + 1]
                if score > alpha:
                    # update alpha!
                    alpha = score
                    if score >= beta:
                        # update history
                        if not state.is_capture(move):
                            bonus = depth * depth
                            hhBonus = (
                                bonus
                                - self.htable[state.turn][move.from_square][
                                    move.to_square
                                ]
                                * abs(bonus)
                                / 16384
                            )
                            self.htable[state.turn][move.from_square][
                                move.to_square
                            ] += hhBonus
                        break
        # No moves were played so its checkmate or stalemate
        # Instead checking if the position is a checkmate
        # during evaluate we can do it here
        # and save computation time0
        if madeMoves == 0:
            if inCheck:
                return mated_in(ply)
            else:
                return 0
        # Calculate bound and save position in tt
        bound = tt.Flag.NONEBOUND
        if bestScore >= beta:
            bound = tt.Flag.LOWERBOUND
        else:
            if alpha != oldAlpha:
                bound = tt.Flag.EXACTBOUND
            else:
                bound = tt.Flag.UPPERBOUND
        if not self.check_time():
            # Store in tt
            self.transposition_table.storeEntry(
                hashKey, depth, bound, bestScore, bestMove, ply
            )
        return bestScore

    def iterative_deepening(self, state) -> int:
        """
        Iterative Deepening, this will call the ab_search function
        with increasing depth until the time0 count_limit is reached or
        the maximum depth is reached.
        """
        self.nodes = 0
        score = -VALUE_INFINITE
        bestmove = chess.Move.null()
        # Start measuring time0
        self.t0 = time.time_ns()
        # Iterative Deepening Loop
        for d in range(self.limit.limited["depth"],
                       self.limit.limited["depth"] + 1):
            score = self.ab_search(state,
                                   -VALUE_INFINITE, VALUE_INFINITE, d, 0)
            # Dont use completed depths result
            if self.stop or self.check_time(True):
                break
            # Save bestmove
            bestmove = self.pvTable[0][0]
            # print info
            now = time.time_ns()
            # stdout.write(self.stats(d, score, now - self.t0) + "\n")
            # stdout.flush()
        # last attempt to get a bestmove
        if bestmove == chess.Move.null():
            bestmove = self.pvTable[0][0]
        return score
        # print bestmove, as per UCI Protocol
        # stdout.write("bestmove " + str(bestmove) + "\n")
        # stdout.flush()

    # Detect a repetition
    def is_repetition(self, state, key: int, draw: int = 1) -> bool:
        count = 0
        size = len(self.hashHistory)
        for i in range(size - 1, -1, -2):
            if i >= size - state.halfmove_clock:
                if self.hashHistory[i] == key:
                    count += 1
                if count == draw:
                    return True
        return False

    # Most Valuable Victim - Least Valuable Aggressor
    def mvvlva(self, state, move: chess.Move) -> int:
        mvvlva = [
            [0, 0, 0, 0, 0, 0, 0],
            [0, 105, 104, 103, 102, 101, 100],
            [0, 205, 204, 203, 202, 201, 200],
            [0, 305, 304, 303, 302, 301, 300],
            [0, 405, 404, 403, 402, 401, 400],
            [0, 505, 504, 503, 502, 501, 500],
            [0, 605, 604, 603, 602, 601, 600],
        ]
        from_square = move.from_square
        to_square = move.to_square
        attacker = state.piece_type_at(from_square)
        victim = state.piece_type_at(to_square)
        # En passant
        if victim is None:
            victim = 1
        return mvvlva[victim][attacker]

    # assign a score to moves in q_search
    def score_qmove(self, state, move: chess.Move) -> int:
        return self.mvvlva(state, move)

    # assign a score to normal moves
    def score_move(self, state, move: chess.Move, ttmove: chess.Move) -> int:
        if move == ttmove:
            return 1_000_000
        elif state.is_capture(move):
            # make sure captures are ordered higher than quiets
            return 32_000 + self.mvvlva(state, move)
        return self.htable[state.turn][move.from_square][move.to_square]

    def get_hash(self, state) -> int:
        return chess.polyglot.zobrist_hash(state)

    def check_time(self, itera: bool = False) -> bool:
        if self.stop:
            return True

        if (
            self.limit.limited["nodes"] != 0
            and self.nodes >= self.limit.limited["nodes"]
        ):
            return True
        if self.checks > 0 and not itera:
            self.checks -= 1
            return False
        self.checks = CHECK_RATE
        if self.limit.limited["time0"] == 0:
            return False
        timeNow = time.time_ns()
        if (timeNow - self.t0) / 1_000_000 > self.limit.limited["time0"]:
            return True
        return False

    # Build PV
    def get_pv(self) -> str:
        pv = ""
        for i in range(0, self.pvLength[0]):
            pv += " " + str(self.pvTable[0][i])
        return pv

    # Convert mate scores
    def convert_score(self, score: int) -> str:
        if score >= VALUE_MATE_IN_PLY:
            return "mate " + str(
                ((VALUE_MATE - score) // 2) + ((VALUE_MATE - score) & 1)
            )
        elif score <= VALUE_MATED_IN_PLY:
            return "mate " + str(
                -((VALUE_MATE + score) // 2) + ((VALUE_MATE + score) & 1)
            )
        else:
            return "cp " + str(score)

    # Print UCI Info
    def stats(self, depth: int, score: int, time0: int) -> str:
        time_in_ms = int(time0 / 1_000_000)
        time_in_seconds = max(1, time_in_ms / 1_000)
        info = (
            "info depth "
            + str(depth)
            + " score "
            + str(self.convert_score(score))
            + " nodes "
            + str(self.nodes)
            + " nps "
            + str(int(self.nodes / time_in_seconds))
            + " time "
            + str(round(time0 / 1_000_000))
            + " pv"
            + self.get_pv()
        )
        return info

    # Reset _search stuff
    def reset(self) -> None:
        self.pvLength[0] = 0
        self.nodes = 0
        self.t0 = 0
        self.stop = False
        self.checks = CHECK_RATE
        self.hashHistory = []
        self.htable = [[[0 for _ in range(64)]
                        for _ in range(64)] for _ in range(2)]


# Run _search.py instead of main.py if you want to profile it!
if __name__ == "__main__":
    board = chess.Board()
    search = Search()
    search.limit.limited["depth"] = 4
    print(search.iterative_deepening(board))
