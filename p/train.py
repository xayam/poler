
import torch
import chess

from p.agent import ChessAgent, ChessRandomAgent, ChessEngineAgent
from p.enviroment import Enviroment


class Train:
    def __init__(self):
        self.enviroment = Enviroment()

    def train_step(self, agent, states, rewards):
        if agent.model is None:
            return 0.0
        agent.optimizer.zero_grad()
        predicted_pos = agent.model(states.unsqueeze(0))
        loss = torch.mean((predicted_pos - states) ** 2) * rewards
        loss.backward()
        agent.optimizer.step()
        return loss.item()

    def play_episode(self, white, black, shift=2, depth=10, level=0):
        board = chess.Board()
        history = [self.enviroment.board_to_tensor(board)]
        while not board.is_game_over():
            current_agent = white if board.turn else black
            if isinstance(current_agent, ChessRandomAgent):
                move = current_agent.get_random(board=board)
            elif isinstance(current_agent, ChessEngineAgent):
                best = True if shift == 0 else False
                move = current_agent.get_move(
                    board=board, best=best, shift=shift,
                    depth=depth, skill_level=level,
                )
            else:
                seq = torch.stack(history[-self.enviroment.SEQ_LENGTH:])
                next_pos = current_agent.predict_pos(seq)
                move = current_agent.pos2move(board, next_pos)
            board.push(move)
            # print(board)
            history.append(self.enviroment.board_to_tensor(board))
        result = board.result()
        white_reward = 1.0 if result == "1-0" \
            else -1.0 if result == "0-1" else -0.01
        black_reward = -white_reward
        loss_white = self.train_step(white, torch.stack(history[:-1]), white_reward)
        loss_black = self.train_step(black, torch.stack(history[:-1]), black_reward)
        return loss_white, loss_black, result

    def plan(self):
        shift = 2
        while True:  # range(1, 7):
            yield [
                {
                    "white": ChessAgent(is_white=True),
                    "black": ChessEngineAgent(is_white=False),
                    "info": "EngineShift",
                    "shift": shift,
                    "depth": 10,
                    "level": 0
                },
                {
                    "white": ChessEngineAgent(is_white=True),
                    "black": ChessAgent(is_white=False),
                }]

    def fit(self, epoches=100):
        plan = self.plan()
        count = 0
        results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0}
        for task in plan:
            for i in range(2):
                count += 1
                white_agent = task[i]["white"]
                black_agent = task[i]["black"]
                losses1, losses2 = 0, 0
                for episode in range(epoches):
                    loss_white, loss_black, result = \
                        self.play_episode(
                            white=white_agent, black=black_agent,
                            shift=task[0]["shift"], depth=task[0]["depth"],
                            level=task[0]["level"],
                        )
                    results[result] += 1
                    losses1 += abs(loss_white)
                    losses2 += abs(loss_black)
                    if white_agent.model is not None:
                        white_agent.model_save()
                    if black_agent.model is not None:
                        black_agent.model_save()
                    print(task[0]["info"] + str(task[0]["shift"]) +
                          f": {epoches * i + episode}, r={result}, "
                          f"i={i}, c={count}, "
                          f"{losses1 / (episode + 1)} | "
                          f"{losses2 / (episode + 1)} | {results}")
                del white_agent
                del black_agent


plan = []
# plan = [[
#     {
#         "white": ChessAgent(is_white=True),
#         "black": ChessEngineAgent(is_white=False, model=False),
#         "info": "Engine",
#         "shift": 10,
#     },
#     {
#         "white": ChessEngineAgent(is_white=True, model=False),
#         "black": ChessAgent(is_white=False),
#     }]] * 20
# plan.append([
#     {
#         "white": ChessAgent(is_white=True),
#         "black": ChessRandomAgent(is_white=False, model=False),
#         "info": "Random",
#         "shift": 10,
#     },
#     {
#         "white": ChessRandomAgent(is_white=True, model=False),
#         "black": ChessAgent(is_white=False),
#     }])

# plan.append([
#     {
#         "white": ChessAgent(is_white=True),
#         "black": ChessAgent(is_white=False),
#         "info": "Self",
#         "depth": 10,
#     },
#     {
#         "white": ChessAgent(is_white=True),
#         "black": ChessAgent(is_white=False),
#     }])


if __name__ == "__main__":
    Train().fit()
