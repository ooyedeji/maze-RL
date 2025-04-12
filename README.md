# Search-RL: Reinforcement Learning and Graph Algorithms for Maze Navigation

Search-RL is a project that combines reinforcement learning (RL) and traditional graph-finding algorithms to solve maze navigation problems. The RL agent learns optimal policies using Q-learning, while graph algorithms like Breadth-First Search (BFS), Depth-First Search (DFS), and A* Search provide alternative solutions for comparison.

---

## Features

- **Reinforcement Learning Agent**:
  - Implements Q-learning with a neural network for decision-making.
  - Supports training with customizable hyperparameters.
  - Tracks metrics like scores, win streaks, and average performance.
  - Saves the best-performing models during training.

- **Graph-Finding Algorithms**:
  - **Breadth-First Search (BFS)**: Explores all possible paths level by level to find the shortest path.
  - **Depth-First Search (DFS)**: Explores paths deeply before backtracking.
  - **A* Search**: Uses a heuristic (Manhattan distance) to prioritize paths likely to reach the goal faster.

- **Maze Environment**:
  - Customizable maze grids with walls, goals, and starting positions.
  - Real-time visualization of the maze and agent's progress using `pygame`.

- **Metrics Visualization**:
  - Plots scores, mean scores, and win streaks over episodes.
  - Provides insights into the agent's learning progress.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/ooyedeji/maze-RL.git
   cd search-RL

2. Install dependencies:
   ```bash
   pip install -r requirements.txt

3. Ensure you have Python 3.8 or higher installed.

---

## Usage

### Training and running the RL Agent

To train the reinforcement learning agent, run the `train.py` script:
```bash
python train.py
```

Then, maze may be solved by specifying the appropriate model path in `solve_maze_rl.py` and running as follows:
```python
python solve_maze_rl.py

```

During training, the script will display metrics such as the episode number, score, mean score, win streak, and maximum streak. A plot of scores and win streaks will also be generated.

### Running Graph-Finding Algorithms

You can use the provided graph-finding algorithms (BFS, DFS, and A*) to solve the maze. These algorithms are integrated into the `maze.py` file and can be called as follows:

```python
from maze import Maze
from graph_pathfinding import Search

maze = Maze(grid_path="grids/default.txt", random_initial=False)

print("Running BFS...")
search = Search(maze=maze, algorithm="BFS")
search.run()

print("Running DFS...")
search = Search(maze=maze, algorithm="DFS")
search.run()

print("Running A*...")
search = Search(maze=maze, algorithm="AStar")
search.run()

```

### Customizing the Maze

- Maze grids are stored in the `grids/` directory.
- Each grid file should be a text file where:
  - `#` represents walls.
  - `O` represents the starting position of the agent.
  - `%` represents the goal.

Example grid file (`grids/grid_01.txt`):
```
######
#O   #
# ## #
#    #
#  ###
#  % #
######
```

You can create your own maze files and place them in the `grids/` directory.

---

## Example Output

### RL Agent Training
During training, the RL agent will display metrics like:
```
Episode:   100 -- Score: 12.5 -- mean_score: 10.2 -- Record: 15.0 -- win_streak: 5 -- max_streak: 10
```

## Configuration

You can customize the training parameters in `train.py`:
- `GAMMA`: Discount factor for future rewards.
- `LR`: Learning rate for the neural network.
- `HIDDEN_LAYER_SIZES`: Size of the hidden layers in the neural network.
- `BATCH_SIZE`: Number of experiences used for training in each batch.
- `EARLY_STOP`: Number of consecutive wins required to stop training.

---

## Dependencies
- `pygame`: For visualizing the maze environment.
- `numpy`: For numerical computations.
- `matplotlib`: For plotting training metrics.
