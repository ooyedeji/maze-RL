import os
import random
import numpy as np
from maze import Maze, MazeStatus
from agent import Agent
from model import Linear_QNet, QTrainer
from utils import plot

GRIDS = [grid for grid in os.listdir("grids") if grid.startswith("grid_")]
AVERAGING_WINDOW = 50
EARLY_STOP = 200
GAMMA = 0.90
LR = 0.001
HIDDEN_LAYER_SIZES = (128,)
BATCH_SIZE = 1_000


def qtrain(random_state=42):
    random.seed(random_state)
    get_grid_path = lambda: "grids/" + random.choice(GRIDS)

    # Track board metrics
    scores = []
    mean_scores = []
    win_streak = 0
    max_streak = 0

    # Initialize maze board and AI agent
    maze = Maze(grid_path=get_grid_path(), random_initial=True)
    model = Linear_QNet(len(maze.get_state()), HIDDEN_LAYER_SIZES, 4)
    trainer = QTrainer(model, lr=LR, gamma=GAMMA)
    agent = Agent(trainer=trainer, batch_size=BATCH_SIZE)

    while True:
        # Get current state
        state = maze.get_state()

        # Get action index from model
        action_id = agent.get_action(state)

        # Advance board with action
        reward, status = maze.play_step(agent.directions[action_id])
        next_state = maze.get_state()

        # Memorize outcomes
        agent.memorize(state, action_id, status, reward, next_state)

        # Train on short memory
        agent.train_short_memory()

        if status != MazeStatus.RUNNING:
            # Train on long memory
            agent.train_long_memory()

            # Update scores
            scores.append(maze.total_reward)
            mean_score = np.mean(scores[-AVERAGING_WINDOW:])
            mean_scores.append(mean_score)
            if status == MazeStatus.WIN:
                win_streak += 1
            else:
                win_streak = 0
            max_streak = win_streak if win_streak > max_streak else max_streak
            streaks = [win_streak, max_streak]

            # Print progress
            print(
                f"Episode: {agent.n_episode:>5}",
                f"-- Score: {scores[-1]:.1f}",
                f"-- mean_score: {mean_scores[-1]:.1f}",
                f"-- Record: {max(scores):.1f}",
                f"-- win_streak: {win_streak}",
                f"-- max_streak: {max_streak}",
            )
            plot(scores=scores, mean_scores=mean_scores, streaks=streaks)

            # Reset board
            maze.grid_path = get_grid_path()
            # maze.grid_path = "grids/" + "default.txt"
            maze.reset()

            # Update agent episode count
            agent.n_episode += 1

            # Save best model
            if win_streak > 0 and win_streak % AVERAGING_WINDOW == 0:
                agent.trainer.save(f"models/model_0.pth")

        # Stop if when exploration is concluded and win_streak >= EARLY_STOP
        if agent.epsilon == agent.epsilon_end and win_streak >= EARLY_STOP:
            print("QNet training completed.")
            agent.trainer.save(f"models/model_1.pth")
            break


if __name__ == "__main__":
    qtrain()
