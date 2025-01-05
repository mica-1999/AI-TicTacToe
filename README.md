# 🧠 Tic-Tac-Toe Q-Learning AI

**A Tic-Tac-Toe game powered by Q-Learning AI.** Play against an ever-improving AI or watch two AIs battle it out as they learn optimal strategies through reinforcement learning.

---

## 🚀 Features

- 🎮 **Play Modes**:  
  - **Human vs. AI**: Challenge the AI yourself.  
  - **AI vs. AI**: Watch two AIs learn through self-play.  
  - **Human vs. Human**: Classic two-player mode.

- 🧠 **AI Learning**: Uses Q-Learning to dynamically improve gameplay based on outcomes.

- 🔒 **Thread-Safe Design**: Multi-threaded updates to the AI’s Q-table are synchronized properly.

- 📈 **Progress Tracking**: AI learning is logged in `learning_thread.log`.

---

## ⚙️ How It Works

### 🧠 Q-Learning Overview

Q-Learning is a **model-free reinforcement learning algorithm** that enables the AI to learn optimal strategies by updating a Q-table. The table stores the value of each state-action pair, and these values are updated iteratively based on rewards.  

#### Key Concepts:
- **State**: The current configuration of the Tic-Tac-Toe board.  
- **Action**: The move the AI chooses (e.g., placing an "X" or "O" on the board).  
- **Reward**: Positive values for winning, negative for losing, and neutral for draws.  
- **Learning Rate (α)**: Controls how much new information overrides old knowledge.  
- **Discount Factor (γ)**: Balances immediate rewards with long-term gains.  
