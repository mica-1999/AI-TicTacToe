import numpy as np
import random
from collections import defaultdict

# Initialize the Q-table
q_table = defaultdict(lambda: random.uniform(-0.01, 0.01))  # Add slight randomness to unvisited Q-values

# Hyperparameters for Q-learning
learning_rate = 0.1       # α: How much we update Q-values
discount_factor = 0.9     # γ: How much future rewards matter
exploration_rate = 0.4    # ε: Probability of choosing a random action
exploration_rate_min = 0.01
exploration_rate_decay = 0.999  # Decay rate per game

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
    """Check if the either can win in one move."""
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
                    print("Player " + opponent + " has a winning move next turn \n")
                    return True, (i, j)
                
                # Undo the move
                board[i, j] = ' '
    
    # If no winning move is found
    return False, None

def check_move_condition(board, move, current_player, opponent_win_cond):
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

def get_human_move(board):
    """Prompt the human player for a move."""
    while True:
        try:
            row, col = map(int, input("Enter your move (row col): ").split())
            row -= 1  # Convert to 0-based indexing
            col -= 1  # Convert to 0-based indexing
            if board[row, col] == ' ':
                return (row, col)
            else:
                print("Invalid move, cell already occupied. Try again.")
        except (ValueError, IndexError):
            print("Invalid input. Please enter row and column as two integers separated by a space.")

def play_game(player1_type, player2_type):
    """Play a game with configurable player types (AI or Human)."""
    board = np.full((3, 3), ' ')  # Reset the board
    current_player = 'X'
    state = tuple(board.flatten()) # Flatten the board for easy state tracking
    done = False
    
    while True:
        winning_move_block, winning_move_block_coords = is_opponent_one_move_from_win(board, current_player)
        winning_move_current, winning_move_current_coords = is_opponent_one_move_from_win(board, 'X' if current_player == 'O' else 'O')
        
        # Determine the type of the current player
        if current_player == 'X' and player1_type == "Human" or current_player == 'O' and player2_type == "Human":
            move = get_human_move(board)
        else:
            move = exploration_move(board, current_player)

        # Make the move on the board
        board[move] = current_player
        next_state = tuple(board.flatten())
        print_board(board)

        # Check if the game is over (win or draw)
        if is_winner(board, current_player):
            reward = 10 # Win
            print(f"{current_player} wins!")
            done = True
        elif is_draw(board):
            reward = 0 # Draw
            print("It's a draw!")
            done = True
        elif winning_move_current: # If a winning move existed for current player and he didnt took it
            if not check_move_condition(board,move,current_player,winning_move_current_coords):
                reward = -5
                print(f"\n Penalty of  {reward} to {current_player} for not taking the winning move")
        elif winning_move_block: # If a winning move existed, verifies if player blocked it
            if check_move_condition(board,move,current_player,winning_move_block_coords):
                reward = 5 # Reward for blocking an opponent winning move
                print(f"\n Reward of  {reward} to {current_player} for blocking the opponent winning move")
            else:
                reward = -5 # Reward for not blocking an opponent winning move
                print(f"\n Penalty of  {reward} to {current_player} for not blocking the opponent winning move")
        else:
            reward = 0 # Normal move
       
        
        update_q_table(state,move,reward,next_state,done)
        state = next_state # Update the state for the next move

        if done:
            break

        # Switch players
        current_player = 'X' if current_player == 'O' else 'O'

def run_self_play_and_human_mode():
    """Run AI-vs-AI for 3000 games, then switch to AI-vs-Human mode forever."""
    print("Starting 3000 AI-vs-AI games...")
    ai_wins = 0
    human_wins = 0
    draws = 0
    global exploration_rate
    for game in range(30000):
        play_game(player1_type="AI", player2_type="AI")
        exploration_rate = max(exploration_rate * exploration_rate_decay, exploration_rate_min)
        

    print(f"AI-vs-AI games completed. Results: AI wins: {ai_wins}, Draws: {draws}")
    print("Switching to AI-vs-Human mode...")

    while True:
        play_game(player1_type="Human", player2_type="AI")

# Run:
run_self_play_and_human_mode()