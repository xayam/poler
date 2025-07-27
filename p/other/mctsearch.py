import io
import pstats
import math
import random
from collections import deque
import time
import cProfile
from typing import Tuple

import chess
from h.model.barriers.chess.mcnode import MCTSNode
from h.model.barriers.chess.helpers import ITERATIONS


class MCTS:
    def __init__(self, state: chess.Board,
                 # search,
                 iterations: int,
                 exploration_constant: float = math.sqrt(2),
                 depth_limit: int = None,
                 cnn=None):
        """ Initialize the MCTS object
        :param state: The initial state of the game
        :param iterations: The number of iterations to run the algorithm
        :param exploration_constant: The exploration constant to use in
               the UCB1 algorithm
        :param depth_limit: The depth count_limit to use in the algorithm
        :param use_opening_book: Whether to use the opening book """
        self.state = state
        # self.search = None
        self.iterations = iterations  # The number of iterations to perform
        # The exploration constant, sqrt(2) by default
        self.exploration_constant = exploration_constant
        self.root = MCTSNode(self.state)
        self.current_node = self.root
        self.depth_limit = depth_limit
        self.cnn = cnn

    def set_current_node(self):
        """ Set the current node to the one corresponding to the given state

         :param state: The state to set the current node to"""
        # First look in the children of the current node
        for child in self.current_node.children:
            if child.state == self.state:
                self.current_node = child
                # print(self.current_node.state())
                return
        # If it's not in the children of the node, look in the entire tree
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            if node.state == self.state and node != self.root and \
                    node != self.current_node:
                self.current_node = node
                # print(self.current_node.state())
                return
            queue.extend(node.children)
        if not self.current_node.state == self.state:
            self.current_node = MCTSNode(self.state)

    def _select(self, node: MCTSNode, depth: int) -> MCTSNode:
        """ Select the next node to explore using the UCB1 algorithm
         :param node: The node to select from
         :param depth: The depth of the node
         :return: The selected node """
        while not node.state.is_game_over():
            if node.not_fully_expanded():
                return node
            if self.depth_limit and depth >= self.depth_limit:
                return node
            children = node.children
            children = [child for child in children if child.alpha <= node.beta]
            if len(children) == 0:
                return node
            # if node.state.turn == chess.WHITE:
            node = max(children,
                       key=lambda c: c.ucb1(self.exploration_constant))
            # else:
            #     node = min(children,
            #                key=lambda c: c.ucb1(self.exploration_constant))
            depth += 1
        return node

    @staticmethod
    def _expand(node: MCTSNode) -> MCTSNode:
        """ Expand the selected node by creating new children
         :param node: The node to expand
         :return: The new child node """
        next_state = node.state.copy()
        moves = list(next_state.legal_moves)
        moving = random.choice(moves)
        next_state.push(moving)
        if next_state.is_game_over():
            return node
        new_node = MCTSNode(next_state, parent=node,
                            alpha=node.alpha, beta=node.beta,
                            move=moving, cnn=None)
        node.children.append(new_node)
        return new_node

    @staticmethod
    def _simulate(node: MCTSNode) -> int:
        """ Simulate the game to a terminal state and return the result
         :param node: The node to simulate from
         :return: The result of the simulation """
        state = node.state.copy()
        r = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}
        while not state.is_game_over():
            moves = list(state.legal_moves)
            moving = random.choice(moves)
            state.push(moving)
            if state.is_game_over():
                return r[state.result()]
        return r[state.result()]

    @staticmethod
    def _backpropagate(node: MCTSNode, res: int):
        """ Backpropagate the result of the
            simulation from the terminal node to the root node
         :param node: The terminal node
         :param res: The result of the simulation"""
        while node is not None:
            node.visits += 1
            node.wins += res
            # print(node.alpha, node.beta)
            node = node.parent

    def select_move(self) -> Tuple[str, float]:
        """ Perform the MCTS algorithm and select the best move
         :return: The best move """
        self.set_current_node()
        for _ in range(self.iterations):
            node = self._select(self.current_node, 0)
            if node.not_fully_expanded():
                node = self._expand(node)
            res = self._simulate(node)
            self._backpropagate(node, res)
        if not self.current_node.children:
            return "", 0.0
        # if self.current_node.state.turn == chess.WHITE:
        best_child = max(self.current_node.children,
                         key=lambda c: c.ucb1(self.exploration_constant))
        # else:
        #     best_child = min(self.current_node.children,
        #                      key=lambda c: c.ucb1(self.exploration_constant))
        self.current_node = best_child
        self.set_current_node()
        return best_child.move, best_child.ucb1(self.exploration_constant)

    def mcts_best(self):
        _move, _score = self.select_move()
        return _move, _score


if __name__ == "__main__":
    chess_state = chess.Board()
    move = ""
    while not chess_state.is_game_over():
        mcts = MCTS(state=chess_state,
                    iterations=ITERATIONS,
                    depth_limit=None)
        move, score = mcts.mcts_best()
        print()
        print(chess_state)
        move = str(move).strip()
        if move:
            m = chess.Move.from_uci(move)
            chess_state.push(m)
        else:
            break
        print(move, score)
    result = chess_state.result()
    if not move:
        result = "1/2-1/2"
    print()
    print(result)
