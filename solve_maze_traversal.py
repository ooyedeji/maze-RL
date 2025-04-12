import pygame
from maze import Maze
from graph_pathfinding import Search

# Initialize maze board and search model
maze = Maze(grid_path="grids/default.txt", random_initial=False)
search = Search(maze=maze, algorithm=0)

search.run()

pygame.time.wait(3_000)
print(f"# of iterations: {maze.iteration}")
pygame.quit()
