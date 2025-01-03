import numpy as np
import random
from collections import defaultdict

# Initialize the Q-table
q_table = defaultdict(float)  # Default value of 0.0 for any (state, action) pair

# Hyperparameters for Q-learning
learning_rate = 0.1       # α: How much we update Q-values
discount_factor = 0.9     # γ: How much future rewards matter
exploration_rate = 0.2    # ε: Probability of choosing a random action

# Initialize the Tic-Tac-Toe board using NumPy
board = np.full((3, 3), ' ')  # A 3x3 grid

def update_q_table(state, action, reward, next_state, done):
    """Update Q-values using the Q-learning formula."""
    current_q = q_table[(state, action)]
    if done:
        # If the game is over, there's no next state
        q_table[(state, action)] = current_q + learning_rate * (reward - current_q)
    else:
        # Non-terminal state
        next_max_q = max(q_table[(next_state, next_action)] 
                         for next_action in get_empty_positions(np.array(next_state).reshape(3, 3)))
        q_table[(state, action)] = current_q + learning_rate * (reward + discount_factor * next_max_q - current_q)

def get_empty_positions(board):
    """Get a list of all empty positions on the board."""
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == ' ']

def is_winner(board, symbol):
    """Check if the given symbol has won."""
    for i in range(3):
        # Check rows and columns
        if all(board[i, :] == symbol) or all(board[:, i] == symbol):
            return True
    # Check diagonals
    if all(board.diagonal() == symbol) or all(np.fliplr(board).diagonal() == symbol):
        return True
    return False

def exploration_move(board, symbol):
    """Choose a move based on exploration-exploitation strategy."""
    # Get the list of empty positions
    empty_positions = get_empty_positions(board)
    
    # Choose a random number between 0 and 1 to decide whether to explore or exploit
    if random.uniform(0, 1) < exploration_rate:
        # Exploration: Choose a random move
        move = random.choice(empty_positions)
    else:
        # Exploitation: Choose the best move based on Q-values
        # Evaluate all empty positions and pick the one with the highest Q-value
        max_q_value = -float('inf')
        best_move = None
        for move in empty_positions:
            # Get the Q-value for the current state and action
            state = tuple(board.flatten())  # Flatten the board to create a state tuple
            q_value = q_table[(state, move)]
            
            if q_value > max_q_value:
                max_q_value = q_value
                best_move = move
        
        move = best_move
    
    # Perform the selected move on the board
    board[move[0], move[1]] = symbol
    return move

def is_draw(board):
    """Check if the game is a draw."""
    return not get_empty_positions(board)

def print_board(board):
    """Print the Tic-Tac-Toe board."""
    print("-" * 9)
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def is_opponent_one_move_from_win(board, current_player):
    """Check if the opponent can win in one move."""
    # Determine the opponent's symbol
    opponent = 'X' if current_player == 'O' else 'O'
    
    # Check all empty spots on the board
    for i in range(3):
        for j in range(3):
            if board[i, j] == ' ':
                # Simulate placing the opponent's move
                board[i, j] = opponent
                
                # Check if this results in a win for the opponent
                if is_winner(board, opponent):
                    board[i, j] = ' '  # Undo the move
                    return True, (i, j)
                
                # Undo the move
                board[i, j] = ' '
    
    # If no winning move is found
    return False, None

def blocked_opponent_move(board, move, current_player, opponent_win_cond):
    """Check if the current move blocks the opponent from winning."""
    # If there was no winning move condition, no need to check
    if opponent_win_cond is None:
        return False
    
    # Get the row and column of the current move
    row, col = move
    
    # Check if the current move matches the opponent's winning move coordinates
    if (row, col) == opponent_win_cond:
        return True
    
    return False

# Play a single game with self-play
def self_play():
    """Let the AI play against itself."""
    board = np.full((3, 3), ' ')  # Reset the board
    current_player = 'X'

    # Start tracking the state before the move
    state = tuple(board.flatten())  # Flatten the board for easy state tracking
    
    while True:
        winning_move, winning_move_coords = is_opponent_one_move_from_win(board, current_player)
        move = exploration_move(board, current_player)
        
        # After making the move, get the next state
        next_state = tuple(board.flatten())
        print_board(board)

        # Check if the game is over (win or draw)
        if is_winner(board, current_player):
            reward = 10  # Win
            done = True
        elif is_draw(board):
            reward = 0  # Draw
            done = True
        elif winning_move: #If a winning move existed, verifies if player blocked it
            # Check if the move blocks an opponent's winning move
            if blocked_opponent_move(board, move, current_player, winning_move_coords):
                reward = 5  # Reward for blocking the opponent's win
            else:
                reward = -5  # Penalty for not blocking the opponent's win
        else: 
            reward = 0 # normal move
            done = False

        # Update the Q-table based on the move made
        update_q_table(state, move, reward, next_state, done)

        if done:
            print(f"{current_player} wins!" if reward == 10 else "It's a draw!")
            break
        
        # Switch players
        current_player = 'X' if current_player == 'O' else 'O'
        state = next_state  # Update the state for the next move

# AI vs Human
def simulate_ai_vs_player():
    """Simulate AI vs player, after 20 self-play games."""
    board = np.full((3, 3), ' ')  # Reset the board
    current_player = 'X'  # Human starts as 'X'
    winning_move = False
    done = False

    while True:
        # Print current board state
        print_board(board)
        
        # Player's turn
        if current_player == 'X':
            valid_move = False
            while not valid_move:
                try:
                    # Prompt player for input
                    row, col = map(int, input("Enter your move (row col): ").split())
                    row -= 1  # Convert to 0-based indexing
                    col -= 1  # Convert to 0-based indexing
                    if board[row, col] == ' ':  # Check if the cell is empty
                        move = (row, col)
                        board[row, col] = 'X'  # Make the move
                        valid_move = True
                        # Switch to AI's turn
                    else:
                        print("Invalid move, cell already occupied. Try again.")
                except (ValueError, IndexError):
                    print("Invalid input. Please enter row and column as two integers separated by a space.")


        # AI's turn
        if current_player == 'O':
            # Flatten the board to create a state tuple
            state = tuple(board.flatten())

            # Check if the state is new or known
            if (state, None) in q_table:
                print("State is known.")

                # Print rewards for all possible moves from the Q-table
                print("Possible moves and rewards:")
                empty_positions = get_empty_positions(board)
                for move in empty_positions:
                    # Get the action (move) rewards from the Q-table
                    if (state, move) in q_table:
                        reward = q_table[(state, move)]
                    else:
                        reward = 0  # If there's no reward for this move, assume it as 0
                    # Print the move and its reward
                    print(f"Move {move} - Reward: {reward}")
            else:
                print("State is new.")

            winning_move, winning_move_coords = is_opponent_one_move_from_win(board, current_player)
            move = exploration_move(board, current_player)
            # After making the move, get the next state
            next_state = tuple(board.flatten())
            print(f"AI chooses move {move}")

        # Check if the game is over (win or draw)
        if is_winner(board, current_player):
            reward = 10  # Win
            done = True
        elif is_draw(board):
            reward = 0  # Draw
            done = True
        elif winning_move: #If a winning move existed, verifies if player blocked it
            # Check if the move blocks an opponent's winning move
            if current_player == 'O' and blocked_opponent_move(board, move, current_player, winning_move_coords):
                reward = 5  # Reward for blocking the opponent's win
                done = False
            else:
                reward = -5  # Penalty for not blocking the opponent's win
                done = False
        else: 
            reward = 0 # normal move
            done = False

        if (current_player == 'O'):
            # Update the Q-table based on the move made
            update_q_table(state, move, reward, next_state, done)
        
        # Switch players
        current_player = 'X' if current_player == 'O' else 'O'

        if done:
            print(f"{current_player} wins!" if reward == 10 else "It's a draw!")
            break
 

# Play multiple self-play games
def test_self_play(num_games):
    """Let the AI play multiple games against itself."""
    for game in range(num_games):
        print(f"Game {game + 1}:")
        self_play()
        print("\n" + "="*20 + "\n")

# Test self-play for 20 games
test_self_play(40000)

while True:
     simulate_ai_vs_player()