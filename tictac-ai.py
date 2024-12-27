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

def ai_learning():
    ""

def getUser_Input(board,current_Player):
    "Gets user input and plants it into the board"
    while True:
        try:
            coordenada_X = int(input("\n\n" + current_Player + " Diga a linha onde deseja introduzir  ")) -1
            coordenada_Y = int(input(current_Player + " Diga a coluna onde deseja introduzir  ")) -1
            if(verify_input(board,coordenada_X,coordenada_Y)):
                "Insert into the board"
                board[coordenada_X][coordenada_Y] = current_Player
                break
            else:
                print("Invalid move. Try Again")
        except ValueError:
            print("Please enter a valid number.")

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

def main():
    "Game Loop"
    board = [[' ' for _ in range(3)] for _ in range(3)]  # Empty 3x3 board
    current_Player = 'X' # X Starts the game

    # While the game is running ( Set Condition later when Win,Draw or Lose to false or break?)
    while True:
        print_board(board)  #Print the board for the Player to See
        getUser_Input(board,current_Player)  #Get Player input(coordenates)
        if(verify_Win(board,current_Player)): # Verify if any of the players won
            print(current_Player + " has won the game! Congrats.")
            print_board(board)
            print("Press Enter to continue...")
            input()
            break
        if(verify_Draw(board)):
            print("The game ended in a draw")
            break
        current_Player = 'O' if current_Player == 'X' else 'X'  #Alternate Turn

while True:
    main()