"""
Headless Tetris environment for training.

Reuses all the real game logic from tetris.py (Board, Piece, try_move,
the NES randomizer, scoring) but replaces the keyboard-driven, real-time
pygame loop with a simple step(action) interface an RL loop can call
directly, with no window and no waiting on wall-clock time.

Simplification from the playable version: instead of frame-accurate NES
gravity timing (which would make each training episode enormous), gravity
here advances by a fixed number of steps (GRAVITY_PERIOD), configurable.
This keeps training tractable while still using the real board, real
piece shapes, real rotation rules, and real NES scoring underneath.
"""

from tetris import Board, Piece, NESRandomizer, try_move, BOARD_WIDTH, BOARD_HEIGHT

# How many env steps pass before the piece drops one row on its own.
# Smaller = faster-falling, more pieces placed per episode.
GRAVITY_PERIOD = 1

# Action indices, in the order the readout's output will use.
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_ROTATE = 2
ACTION_SOFT_DROP = 3
ACTION_NONE = 4
NUM_ACTIONS = 5

# Reward shaping.
SURVIVAL_REWARD = 0.01
TOP_OUT_PENALTY = -5.0
HEIGHT_PENALTY_SCALE = 0.02  # penalty per step, scaled by how tall the stack is
# NES-style per-line-clear scores, kept simple (level always effectively 0
# for reward purposes, since episodes are short during early training).
LINE_CLEAR_REWARD = {1: 1.0, 2: 3.0, 3: 6.0, 4: 12.0}


def _max_stack_height(board):
    """Returns how many rows tall the stack is (0 = empty board)."""
    for row in range(BOARD_HEIGHT):
        if any(cell is not None for cell in board.grid[row]):
            return BOARD_HEIGHT - row
    return 0


class TetrisHeadlessEnv:
    def __init__(self, gravity_period=GRAVITY_PERIOD):
        self.gravity_period = gravity_period
        self.reset()

    def reset(self):
        self.board = Board()
        self.randomizer = NESRandomizer()
        self.current_piece = Piece(self.randomizer.next_piece())
        self.next_shape = self.randomizer.next_piece()
        self.step_count = 0
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        """Returns (board, current_piece), the same snapshot format
        eye_adapter.py's board_to_image already expects."""
        return self.board, self.current_piece

    def step(self, action):
        """
        Applies one action, then checks gravity. Returns:
            observation: (board, current_piece)
            reward: float
            done: bool
        """
        if self.done:
            raise RuntimeError("step() called after episode ended, call reset() first.")

        reward = SURVIVAL_REWARD
        if action == ACTION_LEFT:
            try_move(self.board, self.current_piece, dx=-1)
        elif action == ACTION_RIGHT:
            try_move(self.board, self.current_piece, dx=1)
        elif action == ACTION_ROTATE:
            try_move(self.board, self.current_piece, drotation=1)
        elif action == ACTION_SOFT_DROP:
            # Soft drop: try to move down immediately, on top of gravity.
            try_move(self.board, self.current_piece, dy=1)
        # ACTION_NONE: do nothing this step.

        self.step_count += 1

        if self.step_count % self.gravity_period == 0:
            if not try_move(self.board, self.current_piece, dy=1):
                self.board.lock_piece(self.current_piece)
                cleared = self.board.clear_lines()
                if cleared:
                    reward += LINE_CLEAR_REWARD.get(cleared, 0.0)

                self.current_piece = Piece(self.next_shape)
                self.next_shape = self.randomizer.next_piece()

                if not self.board.is_valid(self.current_piece.cells()):
                    self.done = True
                    reward += TOP_OUT_PENALTY

        # Dense shaping signal: taller stacks are progressively
        # discouraged, giving a learning signal even when nothing has
        # locked or cleared yet this step.
        reward -= HEIGHT_PENALTY_SCALE * (_max_stack_height(self.board) / BOARD_HEIGHT)

        return self._get_observation(), reward, self.done
