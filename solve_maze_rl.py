import torch
import pygame
from maze import Maze, MazeStatus
from agent import Agent

# Load the saved model weights
model = torch.load("models/gold.pth", weights_only=False)
model.eval()

# Initialize maze board and AI agent
maze = Maze(grid_path="grids/default.txt", random_initial=False)
agent = Agent(model=model)

while True:
    # Get current state
    state = maze.get_state()

    # Get action from model
    action_id = agent.get_action(state, explore=False)

    # Advance game with action
    reward, status = maze.play_step(agent.directions[action_id])

    if status != MazeStatus.RUNNING:
        pygame.time.wait(3_000)
        break

print(f"Score: {maze.total_reward:.2f}")
print(f"# of iterations: {maze.iteration}")
pygame.quit()
