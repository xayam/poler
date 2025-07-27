import pprint

import numpy

DTYPE = numpy.complex64

EMPTY_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
WHITE_PAWN_FIELD = numpy.asarray(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
BLACK_PAWN_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=DTYPE)
WHITE_KNIGHT_FIELD = numpy.asarray(
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
BLACK_KNIGHT_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=DTYPE)
WHITE_BISHOP_FIELD = numpy.asarray(
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
BLACK_BISHOP_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=DTYPE)
WHITE_ROOK_FIELD = numpy.asarray(
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
BLACK_ROOK_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=DTYPE)
WHITE_QUEEN_FIELD = numpy.asarray(
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
BLACK_QUEEN_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=DTYPE)
WHITE_KING_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=DTYPE)
BLACK_KING_FIELD = numpy.asarray(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=DTYPE)

GATE_X = numpy.asarray([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ], dtype=DTYPE)
GATE_Y = numpy.asarray([
    [0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0,-1j],
    [0, 0, 0, 0, 0, 0,  0,  0,  0,  0,-1j,  0],
    [0, 0, 0, 0, 0, 0,  0,  0,  0,-1j,  0,  0],
    [0, 0, 0, 0, 0, 0,  0,  0,-1j,  0,  0,  0],
    [0, 0, 0, 0, 0, 0,  0,-1j,  0,  0,  0,  0],
    [0, 0, 0, 0, 0, 0,-1j,  0,  0,  0,  0,  0],
    [ 0, 0, 0, 0, 0,1j, 0,  0,  0,  0,  0,  0],
    [ 0, 0, 0, 0,1j, 0, 0,  0,  0,  0,  0,  0],
    [ 0, 0, 0,1j, 0, 0, 0,  0,  0,  0,  0,  0],
    [ 0, 0,1j, 0, 0, 0, 0,  0,  0,  0,  0,  0],
    [ 0,1j, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0],
    [1j, 0, 0, 0, 0, 0, 0,  0,  0,  0,  0,  0], ], dtype=DTYPE)
GATE_Z = numpy.asarray([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1], ], dtype=DTYPE)
GATE_HADAMARD = (1 / 12 ** 0.5) * numpy.asarray([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1,-1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0,-1, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0,-1, 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0, 0,-1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0,-1, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,-1], ], dtype=DTYPE)


print(WHITE_PAWN_FIELD * GATE_X)

BLACK = False
WHITE = True


class Field:
    def __init__(self, square):
        self.position = square
        self.links = {}

    def add_link(self, from_link, to_link):
        self.links[from_link] = to_link



class GameState:

    def __init__(self, board):
        self.board = None
        self.state = None
        self.fields = {}
        self.figures = {}
        self.init(board=board)

    def init(self, board):
        self.clear()
        self.board = board
        for square in range(64):
            self.add_field(square)

    def clear(self):
        self.board = None
        self.state = None
        self.fields = {}

    def add_field(self, square):
        self.fields[square] = Field(square=square)
        self.figures = {}
