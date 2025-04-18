import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np


def flood_orthogonal(image: np.ndarray, seed_point: tuple[int], gradient=0.95):
    assert image.ndim == len(seed_point)
    seed_x, seed_y = seed_point
    rows, cols = image.shape
    seed_value = image[seed_x, seed_y]

    filled_image = np.zeros_like(image)
    queue = [(seed_x, seed_y, 0)]
    visited = set()

    while queue:
        x, y, n = queue.pop(0)

        if (x, y) in visited:
            continue

        filled_image[x, y] = gradient**n
        visited.add((x, y))

        # Expand frontier to neighbors (no diagonals)
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]

        for i, j in neighbors:
            if 0 <= i < rows and 0 <= j < cols and image[i, j] == seed_value:
                queue.append((i, j, n + 1))

    return filled_image


def plot(scores, mean_scores, streaks):
    clear_output(wait=True)
    plt.clf()
    plt.title(f"Agent training (Win streak: {streaks[0]} | max: {streaks[1]})")

    plt.plot(scores, label="Score")
    plt.plot(mean_scores, label="Mean Score")
    record = [max(scores[: i + 1]) for i in range(len(scores))]
    plt.plot(record, label="Record")

    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.legend(loc="lower right")

    plt.pause(0.1)
