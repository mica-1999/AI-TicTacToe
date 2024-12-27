import random

# Initialize Q-table: It maps board states to action values
Q_table = {}
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.2  # Exploration rate
# Board TicTac
board = [[' ', ' ', ' '],
         [' ', ' ', ' '],
         [' ', ' ', ' ']]


def print_board(board):
    """Print the Tic-Tac-Toe board."""
    print("-" * 9)  # Print a separator between rows
    for row in board:
        print(" | ".join(row))  # Join the cells with ' | ' for separation
        print("-" * 9)  # Print a separator between rows

# Convert board state into a tuple of string representations for easier storage in Q_table
def board_to_tuple(board):
    return tuple(tuple(row) for row in board)

def get_valid_moves(board):
    """Return a list of valid moves (i.e., empty spots)."""
    valid_moves = []
    for i in range(3):
        for j in range(3):
            if board[i][j] == ' ':
                valid_moves.append((i, j))
    return valid_moves

def choose_action(board, player):
    """Choose an action using epsilon-greedy approach."""
    if random.random() < epsilon:
        return random.choice(get_valid_moves(board))  # Explore random move
    else:
        # Exploit the best known move based on Q-table
        state = board_to_tuple(board)
        if state in Q_table:
            # Choose the move with the highest Q-value
            valid_moves = get_valid_moves(board)
            best_move = max(valid_moves, key=lambda move: Q_table[state].get(move, 0))
            return best_move
        else:
            return random.choice(get_valid_moves(board))  # No Q-values yet, explore
        
def update_q_table(board, action, reward, next_board, done):
    """Update the Q-table based on the reward and next state."""
    state = board_to_tuple(board)
    next_state = board_to_tuple(next_board)
    
    if state not in Q_table:
        Q_table[state] = {move: 0 for move in get_valid_moves(board)}
    
    if done:
        Q_table[state][action] = Q_table[state].get(action, 0) + alpha * (reward)
    else:
        future_rewards = max(Q_table.get(next_state, {}).values(), default=0)
        Q_table[state][action] = Q_table[state].get(action, 0) + alpha * (reward + gamma * future_rewards)

def verify_input(board,X,Y):
    "Verify input"
    if 0 <= X < 3 and 0 <= Y < 3:  # Check if the coordinates are within the valid range
        if board[X][Y] == ' ':  # Check if the cell is empty
            return True
    return False

def verify_Win(board,Player):
    """Check if the current player has won the game"""
    
    # Check rows
    for row in board:
        if row[0] == row[1] == row[2] == Player:
            return True
    
    # Check columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] == Player:
            return True

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] == Player:
        return True
    if board[0][2] == board[1][1] == board[2][0] == Player:
        return True
    
    return False

def verify_Draw(board):
    """Check if the board is full and no player has won"""
    for row in board:
        if ' ' in row:  # If there's any empty cell
            return False  # Not a draw, because there's still space to play
    return True  # Board is full and no winner, it's a draw

def play_game():
    "Game Loop"
    board = [[' ' for _ in range(3)] for _ in range(3)]  # Empty 3x3 board
    print_board(board);
    current_Player = 'X' # X Starts the game
    done = False


    #Game Running
    while not done:
        if current_Player == 'X':  # AI's turn
            move = choose_action(board, 'X')
            board[move[0]][move[1]] = 'X'
        else:  # Player's turn
            move = choose_action(board, 'O')
            board[move[0]][move[1]] = 'O'
        

        if verify_Win(board, 'X'):
            print("AI wins!")
            reward = 1
            done = True
        elif verify_Win(board, 'O'):
            print("Player wins!")
            reward = -1
            done = True
        elif verify_Draw(board):
            print("It's a draw!")
            reward = 0
            done = True
        else:
            reward = 0
        
        next_board = [row[:] for row in board]  # Copy board for the next state
        update_q_table(board, move, reward, next_board, done)
        
        # Switch players
        current_Player = 'O' if current_Player == 'X' else 'X'
    print_board(board);
    return done

# Train the AI by playing many games
for i in range(10000):
    play_game()



# Now let's allow the user to play against the trained AI.

def ai_move(board):
    """AI chooses a move based on the current state of the board (random move for simplicity)."""
    empty_cells = get_valid_moves(board)
    return random.choice(empty_cells)  # Simple random move

def user_play(board):
    """Allow the user to play against the trained AI."""
    while True:
        print_board(board)
        while True:
            try:
                # Get user input (row, col)
                row = int(input("Enter row (1-3): ")) - 1
                col = int(input("Enter column (1-3): ")) - 1
                if board[row][col] == ' ':
                    board[row][col] = 'X'  # User plays as 'X'
                    break
                else:
                    print("This spot is already taken.")
            except (ValueError, IndexError):
                print("Invalid input, please enter valid row and column values.")

        # Check if the user won
        if verify_Win(board, 'X'):
            print_board(board)
            print("You win!")
            break
        if verify_Draw(board):
            print_board(board)
            print("It's a draw!")
            break

        # AI's turn (AI plays as 'O')
        row, col = ai_move(board)
        board[row][col] = 'O'
        print(f"AI played at row {row+1}, col {col+1}")

        # Check if the AI won
        if verify_Win(board, 'O'):
            print_board(board)
            print("AI wins!")
            break
        if verify_Draw(board):
            print_board(board)
            print("It's a draw!")
            break

# Start the game after training
board = [[' ' for _ in range(3)] for _ in range(3)]
user_play(board)