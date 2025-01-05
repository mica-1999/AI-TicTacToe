Tic-Tac-Toe AI with Q-Learning
This project implements a Tic-Tac-Toe game with an AI that uses Q-Learning, a reinforcement learning technique, to improve its strategy through self-play.

Features
AI Learning: The AI improves over time by playing against itself.
Human vs AI: Play against the AI and watch it adapt.
AI vs AI: Observe how two AI players compete.
Reinforcement Learning:
Rewards for making strategic moves (e.g., blocking opponent or winning).
Penalties for missing strategic opportunities.
Thread-Safe Q-Learning: Supports concurrent updates to the Q-table.
How It Works
Q-Learning Fundamentals:

A Q-table stores the AI's knowledge about the game states and actions.
The AI chooses actions based on an exploration-exploitation strategy.
Exploration: Random moves to discover new strategies.
Exploitation: Select moves based on the highest Q-values.
Learning Process:

Rewards for winning, blocking opponent moves, and strategic plays.
Penalties for missing winning opportunities or allowing the opponent to win.
Q-values are updated after every move using the Q-learning formula.
Self-Play:

The AI trains by playing 3,000 games against itself in a separate thread.
Results and learning progress are logged.
Game Modes:

Human vs AI: Play as "X" or "O".
AI vs AI: Watch two AI players compete.
Manual Play: Use the console to input your moves.
Files
learning_thread.log: Logs the AI's training progress.
Q-Table: Stored in memory during execution and updated dynamically.
Requirements
Python: Version 3.6 or higher.
Dependencies:
numpy
Install dependencies with:

bash
Copy code
pip install numpy
How to Run
Clone the repository:
bash
Copy code
git clone <repository-url>
cd <repository-folder>
Run the program:
bash
Copy code
python tic_tac_toe_ai.py
Choose a game mode and follow the on-screen instructions.
Code Overview
Main Components
Board Management:

The game board is represented as a 3x3 NumPy array.
Q-Table:

A dictionary mapping (state, action) pairs to Q-values.
Default Q-values are initialized with slight randomness to encourage exploration.
Reinforcement Learning:

Updates Q-values after each move using:
python
Copy code
Q(s, a) = Q(s, a) + α * (reward + γ * max(Q(s', a')) - Q(s, a))
Where:
α is the learning rate.
γ is the discount factor.
AI Strategy:

Chooses moves based on exploration rate (ε):
Random moves with probability ε.
Best-known move (highest Q-value) otherwise.
Decays ε over time to focus on exploitation as training progresses.
Game Logic:

Checks for winners, draws, and valid moves.
Prioritizes blocking opponents or taking advantageous moves.
Multithreading:

Runs self-play games in the background using a separate thread.
Thread-safe updates ensure consistency.
Customization
Hyperparameters:
Adjust learning rate (learning_rate), discount factor (discount_factor), and exploration rate (exploration_rate) to fine-tune AI learning.
Training Games:
Modify the number of self-play games in the learning_thread function.
Future Enhancements
Save and load Q-table for persistent learning across sessions.
Implement a graphical interface for easier gameplay.
Extend AI to learn other board games (e.g., Connect Four).
Credits
Developed by [Micael Ribeiro].
Inspired by reinforcement learning principles.
