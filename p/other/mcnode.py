import math
import chess

from h.model.barriers.chess.helpers import MAX_PLY, VALUE_INFINITE
from h.model.barriers.chess.limits import Limits
from h.model.barriers.chess.evaluation import evaluate


class MCTSNode:
    """ This is a node in the Monte Carlo Search Tree. """

    def __init__(self, state: chess.Board, parent=None,
                 move=None, alpha=-float("inf"),
                 beta=float("inf"), cnn=None):
        """ Create a new node

         :param state: The current state
         :param parent: The parent node
         :param move: The move that led to the current state
         :param alpha: The alpha value
         :param beta: The beta value"""
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.wins = 0
        self.alpha = alpha
        self.beta = beta
        self.cnn = cnn

    def not_fully_expanded(self) -> bool:
        """ Check if the node has been fully expanded
         :return: True if the node has not been fully expanded, False otherwise"""
        return len(self.children) < len(list(self.state.generate_legal_moves()))

    def ucb1(self, exploration_constant: float) -> float:
        """ Apply the UCT formula (Upper Confidence Bound applied to Trees)
         :param exploration_constant: The exploration constant to use
         :param search
         :return: The UCT value"""
        if self.visits == 0:
            return float('inf')
        e = evaluate(self.state)
        return e
        # ret = self.wins / self.visits + exploration_constant * \
        #     math.sqrt(math.log(self.parent.visits) / self.visits)
