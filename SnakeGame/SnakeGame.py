# This is a simple Snake game implemented in Python using Tkinter for the GUI.

import tkinter as tk
import numpy as np
import random
import json

# Constants
GRID_SIZE = 20       # Number of cells in the grid (20x20)
CELL_SIZE = 20       # Size of each cell in pixels
SPEED = 100          # Delay between moves in milliseconds

# Directions
DIRECTIONS = {
    'Up': np.array([-1, 0]),
    'Down': np.array([1, 0]),
    'Left': np.array([0, -1]),
    'Right': np.array([0, 1]),
}

# Opposite directions to prevent the snake from reversing
OPPOSITE_DIRECTIONS = {
    'Up': 'Down',
    'Down': 'Up',
    'Left': 'Right',
    'Right': 'Left',
}

# Grid encoding
EMPTY_CELL = 0
SNAKE_CELL = 1
FOOD_CELL = 2

# Input encoding
INPUT_ENCODING = {
    'Up': 'U',
    'Down': 'D',
    'Left': 'L',
    'Right': 'R',
    'None': 'N',  # Represents no input
}

class SnakeGame:
    def __init__(self, master):
        self.master = master
        self.master.title("Snake Game")
        self.canvas = tk.Canvas(master, width=GRID_SIZE*CELL_SIZE, height=GRID_SIZE*CELL_SIZE)
        self.canvas.pack()

        # Initialize game state
        self.reset_game()

        # Bind keys
        self.master.bind("<Key>", self.on_key_press)

        # Start game loop
        self.update()

    def reset_game(self):
        self.snake = [np.array([GRID_SIZE // 2, GRID_SIZE // 2])]  # Start in the middle
        self.direction = 'Right'  # Initial direction
        self.next_direction = 'Right'
        self.place_food()
        self.game_over = False

        # Initialize recording
        self.frames = []
        self.inputs = []

        # Initialize current input
        self.current_input = INPUT_ENCODING['None']

        # Record initial frame
        self.record_frame()

    def place_food(self):
        while True:
            position = np.array([random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)])
            if not any((position == segment).all() for segment in self.snake):
                self.food = position
                break

    def on_key_press(self, event):
        key = event.keysym
        if key in DIRECTIONS:
            if key != OPPOSITE_DIRECTIONS[self.direction]:
                self.next_direction = key
                # Record input
                self.current_input = INPUT_ENCODING[key]

    def move_snake(self):
        self.direction = self.next_direction
        new_head = self.snake[0] + DIRECTIONS[self.direction]

        # Check for collision with walls
        if (new_head < 0).any() or (new_head >= GRID_SIZE).any():
            self.game_over = True
            self.save_data()
            return

        # Check for collision with self
        if any((new_head == segment).all() for segment in self.snake):
            self.game_over = True
            self.save_data()
            return

        # Move snake
        self.snake.insert(0, new_head)

        # Check for food consumption
        if (new_head == self.food).all():
            self.place_food()
        else:
            self.snake.pop()

    def update(self):
        if not self.game_over:
            self.move_snake()
            self.draw()

            # Record frame after moving the snake
            self.record_frame()

            # Prepare for the next input
            self.current_input = INPUT_ENCODING['None']

            self.master.after(SPEED, self.update)
        else:
            self.canvas.create_text(
                GRID_SIZE*CELL_SIZE/2,
                GRID_SIZE*CELL_SIZE/2,
                text="Game Over",
                fill="red",
                font=("Arial", 24)
            )

    def draw(self):
        self.canvas.delete("all")

        # Draw snake
        for segment in self.snake:
            x = segment[1] * CELL_SIZE
            y = segment[0] * CELL_SIZE
            self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="green")

        # Draw food
        x = self.food[1] * CELL_SIZE
        y = self.food[0] * CELL_SIZE
        self.canvas.create_rectangle(x, y, x + CELL_SIZE, y + CELL_SIZE, fill="red")

    def record_frame(self):
        frame = np.full((GRID_SIZE, GRID_SIZE), EMPTY_CELL, dtype=int)
        # Mark snake positions
        for segment in self.snake:
            frame[segment[0], segment[1]] = SNAKE_CELL  # Snake body
        # Mark food position
        frame[self.food[0], self.food[1]] = FOOD_CELL  # Food
        # Append frame and input to the lists
        self.frames.append(frame.tolist())
        self.inputs.append(self.current_input)

    def save_data(self):
        # Load existing data if the file exists
        try:
            with open('snake_game_data.json', 'r') as f:
                existing_data = json.load(f)
        except FileNotFoundError:
            existing_data = []

        # Prepare the new data
        new_data = []
        for frame, input_key in zip(self.frames, self.inputs):
            new_data.append({
                'frame': frame,
                'input': input_key
            })

        # Append new data to existing data
        existing_data.extend(new_data)

        # Save the combined data back to the file
        with open('snake_game_data.json', 'w') as f:
            json.dump(existing_data, f)
        print("Game data appended to 'snake_game_data.json'.")

if __name__ == "__main__":
    root = tk.Tk()
    game = SnakeGame(root)
    root.mainloop()
