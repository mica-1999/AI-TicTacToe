def print_board(board):
    """Print the Tic-Tac-Toe board."""
    for row in board:
        print(" | ".join(row))  # Join the cells with ' | ' for separation
        print("-" * 9)  # Print a separator between rows

# Example usage
board = [[' ', ' ', ' '],
         [' ', ' ', ' '],
         [' ', ' ', ' ']]

print_board(board)