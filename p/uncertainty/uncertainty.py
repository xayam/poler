import chess
import random
from collections import Counter

class UncertaintyEngine:
    def __init__(self):
        self.board = chess.Board()
    
    def calculate_uncertainty(self, board):
        """Оценка неопределенности позиции"""
        
        # 1. Количество легальных ходов (больше = неопределеннее)
        move_count = len(list(board.legal_moves))
        mobility_uncertainty = move_count / 50.0  # Нормализация
        
        # 2. Разброс оценок ходов (чем больше разброс = неопределеннее)
        evaluations = []
        for move in board.legal_moves:
            board.push(move)
            eval_score = self.simple_evaluation(board)
            evaluations.append(eval_score)
            board.pop()
        
        if len(evaluations) > 1:
            evaluation_variance = self.calculate_variance(evaluations)
        else:
            evaluation_variance = 0
        
        # 3. Сложность позиции (атаки, связки, открытые линии)
        positional_complexity = self.positional_complexity(board)
        
        # 4. Непредсказуемость (насколько ходы нестандартны)
        unpredictability = self.unpredictability_factor(board)
        
        # Комбинируем факторы
        total_uncertainty = (
            0.3 * mobility_uncertainty +
            0.3 * evaluation_variance +
            0.2 * positional_complexity +
            0.2 * unpredictability
        )
        
        return total_uncertainty
    
    def simple_evaluation(self, board):
        """Простая оценка позиции"""
        if board.is_checkmate():
            return -10000 if board.turn else 10000
        if board.is_stalemate() or board.is_insufficient_material():
            return 0
            
        piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        
        score = 0
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                value = piece_values[piece.piece_type]
                if piece.color == chess.WHITE:
                    score += value
                else:
                    score -= value
        return score
    
    def calculate_variance(self, values):
        """Рассчитываем дисперсию оценок"""
        if len(values) <= 1:
            return 0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return min(variance / 10000, 1.0)  # Нормализация
    
    def positional_complexity(self, board):
        """Оценка сложности позиции"""
        complexity = 0
        
        # Атаки на короля
        king_square = board.king(not board.turn)
        if king_square is not None:
            attackers = board.attackers(board.turn, king_square)
            complexity += len(attackers) * 0.1
        
        # Открытые линии
        complexity += len(board.attacks(chess.E4)) * 0.01
        
        # Связки и вилки
        # Упрощенная оценка
        complexity = min(complexity, 1.0)
        return complexity
    
    def unpredictability_factor(self, board):
        """Фактор непредсказуемости"""
        # Здесь можно использовать базу данных "типичных ходов"
        # и избегать их
        
        # Пока простая реализация - случайность как фактор
        return random.random() * 0.3  # До 30% непредсказуемости
    
    def uncertainty_minimax(self, board, depth, maximizing_player):
        """Минимакс по неопределенности"""
        
        if depth == 0 or board.is_game_over():
            return self.calculate_uncertainty(board)
        
        legal_moves = list(board.legal_moves)
        
        if maximizing_player:
            max_uncertainty = float('-inf')
            best_move = None
            
            for move in legal_moves:
                board.push(move)
                uncertainty = self.uncertainty_minimax(board, depth - 1, False)
                board.pop()
                
                if uncertainty > max_uncertainty:
                    max_uncertainty = uncertainty
                    best_move = move
            
            if depth == 3:  # Верхний уровень - возвращаем ход
                return best_move
            return max_uncertainty
        else:
            min_uncertainty = float('inf')
            for move in legal_moves:
                board.push(move)
                uncertainty = self.uncertainty_minimax(board, depth - 1, True)
                board.pop()
                min_uncertainty = min(min_uncertainty, uncertainty)
            return min_uncertainty
    
    def get_best_move(self, board, depth=3):
        """Получить лучший ход по неопределенности"""
        self.board = board.copy()
        return self.uncertainty_minimax(self.board, depth, True)

# Пример использования
def play_game():
    board = chess.Board()
    engine = UncertaintyEngine()
    
    print("Игра начинается!")
    print(board)
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Наш движок
            move = engine.get_best_move(board, depth=3)
            print(f"Ход движка: {move}")
        else:
            # Случайный ход для демонстрации
            move = list(board.legal_moves)[0]
            print(f"Ход противника: {move}")
        
        board.push(move)
        print(board)
        print("-" * 40)
        
        if len(board.move_stack) > 20:  # Ограничение для демонстрации
            break
    
    print("Игра завершена!")

# Запуск игры
if __name__ == "__main__":
    play_game()