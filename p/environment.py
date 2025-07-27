
import chess
import torch

from p.config import Config


class Environment(Config):
    def __init__(self):
        Config.__init__(self)

    def board_to_tensor(self, board):
        tensor = torch.zeros(16, 8, 8)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece.piece_type - 1 + (6 if piece.color else 0)
                row, col = 7 - square // 8, square % 8
                tensor[channel, row, col] = 1
        tensor[12] = int(board.has_queenside_castling_rights(chess.WHITE))
        tensor[13] = int(board.has_kingside_castling_rights(chess.WHITE))
        tensor[14] = int(board.has_queenside_castling_rights(chess.BLACK))
        tensor[15] = int(board.has_kingside_castling_rights(chess.BLACK))
        return tensor
