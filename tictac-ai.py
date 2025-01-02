import random
import numpy as np
from collections import defaultdict

# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2

# Q-Table (default dictionary to handle state-action pairs)
Q_table = defaultdict(lambda: defaultdict(lambda: 0.01))

# Initialize the Tic-Tac-Toe board using NumPy
board = np.full((3, 3), ' ')  # A 3x3 grid

def print_board(board):
    """Print the Tic-Tac-Toe board."""
    print("-" * 9)
    for row in board:
        print(" | ".join(row))
        print("-" * 9)

def board_to_tuple(board):
    """Convert the board to a tuple for storing in the Q-table."""
    return tuple(tuple(row) for row in board)

def get_valid_moves(board):
    """Return a list of valid moves (i.e., empty spots)."""
    return [(i, j) for i in range(3) for j in range(3) if board[i][j] == ' ']

def choose_action(board, player, epsilon):
    """Choose an action using epsilon-greedy approach."""
    state = board_to_tuple(board)
    valid_moves = get_valid_moves(board)
    random.shuffle(valid_moves)
    
    # Epsilon-greedy approach
    if random.random() < epsilon:
        print("Exploring: Random move")
        return random.choice(valid_moves)
    
    # Exploiting: Choose the best action based on learned values
    if state in Q_table:
        best_move = max(valid_moves, key=lambda move: Q_table[state].get(move, 0))
        print("Exploiting: Best learned move")
        return best_move
    else:
        print("Never seen state: Random move")
        return random.choice(valid_moves)


def update_q_table(board, action, reward, next_board, done):
    """Update the Q-table based on the reward and next state."""
    state = board_to_tuple(board)
    next_state = board_to_tuple(next_board)
    
    # Q-learning update rule
    if done:
        Q_table[state][action] = Q_table[state].get(action, 0) + alpha * reward
    else:
        future_rewards = max(Q_table[next_state].values(), default=0)
        Q_table[state][action] = Q_table[state].get(action, 0) + alpha * (reward + gamma * future_rewards)

def verify_win(board, player):
    """Check if the current player has won the game."""
    board = np.array(board)  # Convert list to a NumPy array
    # Check rows, columns, and diagonals for a winning combination
    for row in board:
        if np.all(row == player):
            return True
    for col in range(3):
        if np.all(board[:, col] == player):
            return True
    if np.all(np.diagonal(board) == player) or np.all(np.diagonal(np.fliplr(board)) == player):
        return True
    return False

def verify_draw(board):
    """Check if the board is full and no player has won."""
    return np.all(board != ' ')

def random_move(board):
    """Generate a random valid move for the board."""
    available_moves = [(i, j) for i in range(3) for j in range(3) if board[i, j] == ' ']
    return random.choice(available_moves)

def is_opponent_winning_next_turn(board):
    # Convert to NumPy array
    board = np.array(board)
    
    # Get a list of all valid moves
    valid_moves = get_valid_moves(board)

    # Simulate each valid move for the opponent
    for move in valid_moves:
        # Make a copy of the board to simulate the move
        simulated_board = np.array([row[:] for row in board])  # Convert to NumPy array
        # Apply the opponent's move
        simulated_board[move[0]][move[1]] = 'O'
        # Check if this move results in a win for the opponent
        if verify_win(simulated_board, 'O'):
            print("Opponent has a win chance")
            return True

    # If no winning move is found, return False
    return False

def is_ai_setting_up_win(board):
    """Check if the AI can set up a winning move on the next turn."""
    valid_moves = get_valid_moves(board)
    
    for move in valid_moves:
        # Make a copy of the board to simulate the AI's move
        simulated_board = [row[:] for row in board]
        # Apply the AI's move (assuming 'X' for AI)
        simulated_board[move[0]][move[1]] = 'X'
        
        # Check if this move results in a win for the AI
        if verify_win(simulated_board, 'X'):
            return True
    
    return False

def is_blocking_opponent(board, ai_move):
    # Make a copy of the board to simulate the AI's move
    simulated_board = [row[:] for row in board]
    # Apply the AI's move (let's assume AI is 'X')
    simulated_board[ai_move[0]][ai_move[1]] = 'X'
    
    # Now check if the opponent still has a winning move
    if is_opponent_winning_next_turn(simulated_board):
        return True  # The AI blocks the winning move
    return False  # The AI does not block the winning move

def play_game(epsilon):
    """Game Loop."""
    board = np.full((3, 3), ' ')  # Empty 3x3 board
    print_board(board)
    current_player = 'O'  # X starts the game
    done = False
    ai_last_move = None  # Initialize this variable outside the loop

    while not done:
        if current_player == 'X':  # AI's turn
            ai_last_move = choose_action(board, 'X',epsilon)
            board[ai_last_move[0], ai_last_move[1]] = 'X'
        else:  # Player's turn
            move = random_move(board)
            board[move[0], move[1]] = 'O'

        print_board(board)

        if verify_win(board, 'X'):  # AI wins
            reward = 10
            done = True
        elif verify_win(board, 'O'):  # Player wins
            reward = -10
            done = True
        elif verify_draw(board):  # Draw
            reward = 1
            done = True
        elif is_opponent_winning_next_turn(board):  # Opponent has a chance to win
            if ai_last_move and is_blocking_opponent(board, ai_last_move):  # AI blocks the winning move
                reward = 3  # Reward for blocking
            else:  # AI fails to block the opponent's winning move
                reward = -5  # Penalize AI for not blocking
            done = False
        elif is_ai_setting_up_win(board):  # AI sets up a winning move
            reward = 2  # Reward for setting up a win
            done = False
        else:  # Neutral move
            reward = 0  # No reward or penalty
            done = False
        
        # Only update the Q-table if the AI made a move
        if ai_last_move:  # Ensure ai_last_move was set (i.e., AI made a move)
            next_board = board.copy()
            update_q_table(board, ai_last_move, reward, next_board, done)

        # Switch players
        current_player = 'O' if current_player == 'X' else 'X'

    return done

# Train the AI by playing many games
for i in range(30000):
    epsilon = max(0.1, epsilon * 0.995)  # Slower decay rate
    play_game(epsilon)
    

# Now, let's allow the user to play against the trained AI.

def ai_move(board):
    """AI chooses the best move based on the current state of the board."""
    empty_cells = get_valid_moves(board)
    state = board_to_tuple(board)
    random.shuffle(empty_cells)
    
    # Print whether the current state exists in the Q-table
    if state in Q_table:
        print("AI has learned about this state before")
        print("Rewards for valid moves in this state:")
        
        # Print rewards for each valid move
        for move in empty_cells:
            reward = Q_table[state].get(move, 0)
            print(f"Move: {move}, Reward: {reward}")
    else:
        print("Never seen state")
    
    # If there are valid moves, choose the one with the highest Q-value
    best_move = max(empty_cells, key=lambda move: Q_table[state].get(move, 0))  # Get the move with the highest Q-value
    return best_move

def user_play(board):
    """Allow the user to play against the trained AI."""
    
    reward = 0  # Default reward if the game isn't finished
    done = False  # Game completion flag
    
    while not done:
        print_board(board)
        
        # User's turn
        while True:
            try:
                row = int(input("Enter row (1-3): ")) - 1
                col = int(input("Enter column (1-3): ")) - 1
                if board[row, col] == ' ':
                    board[row, col] = 'X'  # User plays as 'X'
                    break
                else:
                    print("This spot is already taken.")
            except (ValueError, IndexError):
                print("Invalid input, please enter valid row and column values.")
        
        # Check for game outcome after user's move
        if verify_win(board, 'X'):
            print_board(board)
            print("You win!")
            reward = -1
            done = True
        elif verify_draw(board):
            print_board(board)
            print("It's a draw!")
            reward = 0
            done = True

        if done:  # If game is over, skip AI's turn
            break
        
        # AI's turn
        row, col = ai_move(board)
        board[row, col] = 'O'
        print(f"AI played at row {row + 1}, col {col + 1}")
        
        # Check for game outcome after AI's move
        if verify_win(board, 'O'):
            print_board(board)
            print("AI wins!")
            reward = 1
            done = True
        elif verify_draw(board):
            print_board(board)
            print("It's a draw!")
            reward = 0
            done = True

    # Update Q-table after the game ends
    update_q_table(board, (row, col), reward, board.copy(), done)


# Start the game after 
for i in range(50):
    board = np.full((3, 3), ' ')
    user_play(board)
