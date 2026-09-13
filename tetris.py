

import random
import sys

import pygame

# ---- Board and rendering constants ----
BOARD_WIDTH = 10
BOARD_HEIGHT = 20
CELL_SIZE = 30
SIDE_PANEL_WIDTH = 160

SCREEN_WIDTH = BOARD_WIDTH * CELL_SIZE + SIDE_PANEL_WIDTH
SCREEN_HEIGHT = BOARD_HEIGHT * CELL_SIZE

BG_COLOR = (18, 18, 18)
GRID_COLOR = (40, 40, 40)
TEXT_COLOR = (230, 230, 230)

# ---- DAS (delayed auto shift) timing, approximating NES frame counts ----
NES_FRAME_MS = 1000 / 60.0988
DAS_INITIAL_DELAY_MS = 16 * NES_FRAME_MS   # delay before a held key starts repeating
DAS_REPEAT_MS = 6 * NES_FRAME_MS           # repeat rate once it kicks in
SOFT_DROP_INTERVAL_MS = 2 * NES_FRAME_MS   # fall speed while Down is held

# ---- NES gravity curve: frames per row fall, by level ----
# Levels not listed use the last defined value (matches original leveling off).
_LEVEL_FRAMES = {
    0: 48, 1: 43, 2: 38, 3: 33, 4: 28, 5: 23, 6: 18, 7: 13, 8: 8, 9: 6,
    10: 5, 11: 5, 12: 5, 13: 4, 14: 4, 15: 4, 16: 3, 17: 3, 18: 3,
    19: 2, 20: 2, 21: 2, 22: 2, 23: 2, 24: 2, 25: 2, 26: 2, 27: 2, 28: 2,
    29: 1,
}


def gravity_interval_ms(level):
    frames = _LEVEL_FRAMES.get(level, _LEVEL_FRAMES[29] if level > 29 else _LEVEL_FRAMES[0])
    return frames * NES_FRAME_MS


# ---- NES scoring: points depend on lines cleared at once and current level ----
_LINE_SCORES = {1: 40, 2: 100, 3: 300, 4: 1200}


# ---- Tetromino definitions: fixed rotation states, no wall-kick data ----
SHAPES = {
    "I": [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
    ],
    "O": [
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    "T": [
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    "S": [
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
    ],
    "Z": [
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
    ],
    "J": [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    "L": [
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
}

SHAPE_COLORS = {
    "I": (0, 200, 200),
    "O": (200, 200, 0),
    "T": (160, 0, 200),
    "S": (0, 200, 0),
    "Z": (200, 0, 0),
    "J": (0, 0, 200),
    "L": (200, 120, 0),
}

PIECE_KEYS = list(SHAPES.keys())


class NESRandomizer:


    def __init__(self):
        self.previous = None

    def next_piece(self):
        first_roll = random.randint(0, 7)  # 0-6 = pieces, 7 = blank
        if first_roll == 7 or PIECE_KEYS[first_roll] == self.previous:
            choice = random.choice(PIECE_KEYS)
        else:
            choice = PIECE_KEYS[first_roll]
        self.previous = choice
        return choice


class Piece:
    def __init__(self, shape_key):
        self.shape_key = shape_key
        self.rotation = 0
        self.x = BOARD_WIDTH // 2 - 2
        self.y = 0

    def cells(self, rotation=None, x=None, y=None):
        rotation = self.rotation if rotation is None else rotation
        x = self.x if x is None else x
        y = self.y if y is None else y
        rotations = SHAPES[self.shape_key]
        offsets = rotations[rotation % len(rotations)]
        return [(x + dx, y + dy) for dx, dy in offsets]

    def color(self):
        return SHAPE_COLORS[self.shape_key]


class Board:
    def __init__(self):
        self.grid = [[None for _ in range(BOARD_WIDTH)] for _ in range(BOARD_HEIGHT)]
        self.score = 0
        self.lines_cleared = 0
        self.level = 0

    def is_valid(self, cells):
        for col, row in cells:
            if col < 0 or col >= BOARD_WIDTH:
                return False
            if row >= BOARD_HEIGHT:
                return False
            if row >= 0 and self.grid[row][col] is not None:
                return False
        return True

    def lock_piece(self, piece):
        for col, row in piece.cells():
            if row >= 0:
                self.grid[row][col] = piece.shape_key

    def clear_lines(self):
        remaining = [row for row in self.grid if any(cell is None for cell in row)]
        cleared = BOARD_HEIGHT - len(remaining)
        if cleared:
            new_rows = [[None] * BOARD_WIDTH for _ in range(cleared)]
            self.grid = new_rows + remaining
            self.score += _LINE_SCORES[cleared] * (self.level + 1)
            self.lines_cleared += cleared
            self.level = self.lines_cleared // 10
        return cleared

    def is_game_over(self):
        return any(cell is not None for cell in self.grid[0])


def try_move(board, piece, dx=0, dy=0, drotation=0):
    new_rotation = piece.rotation + drotation
    new_cells = piece.cells(rotation=new_rotation, x=piece.x + dx, y=piece.y + dy)
    if board.is_valid(new_cells):
        piece.x += dx
        piece.y += dy
        piece.rotation = new_rotation
        return True
    return False


def draw_board(screen, board, font, next_piece):
    screen.fill(BG_COLOR)

    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            shape_key = board.grid[row][col]
            if shape_key is not None:
                rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(screen, SHAPE_COLORS[shape_key], rect)
                pygame.draw.rect(screen, BG_COLOR, rect, 1)

    for col in range(BOARD_WIDTH + 1):
        x = col * CELL_SIZE
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, SCREEN_HEIGHT))
    for row in range(BOARD_HEIGHT + 1):
        y = row * CELL_SIZE
        pygame.draw.line(screen, GRID_COLOR, (0, y), (BOARD_WIDTH * CELL_SIZE, y))

    panel_x = BOARD_WIDTH * CELL_SIZE + 15
    screen.blit(font.render(f"Score: {board.score}", True, TEXT_COLOR), (panel_x, 20))
    screen.blit(font.render(f"Level: {board.level}", True, TEXT_COLOR), (panel_x, 50))
    screen.blit(font.render(f"Lines: {board.lines_cleared}", True, TEXT_COLOR), (panel_x, 80))
    screen.blit(font.render("Next:", True, TEXT_COLOR), (panel_x, 120))

    preview_origin_x = panel_x
    preview_origin_y = 150
    for dx, dy in SHAPES[next_piece][0]:
        rect = pygame.Rect(
            preview_origin_x + dx * (CELL_SIZE - 5),
            preview_origin_y + dy * (CELL_SIZE - 5),
            CELL_SIZE - 6,
            CELL_SIZE - 6,
        )
        pygame.draw.rect(screen, SHAPE_COLORS[next_piece], rect)


def draw_piece(screen, piece):
    for col, row in piece.cells():
        if row >= 0:
            rect = pygame.Rect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, piece.color(), rect)
            pygame.draw.rect(screen, BG_COLOR, rect, 1)


class HeldKeyRepeater:
    """Implements NES-style DAS: an initial delay before a held direction
    starts repeating, then a faster steady repeat rate."""

    def __init__(self):
        self.direction = 0  # -1, 0, or 1
        self.held_ms = 0
        self.has_repeated_once = False

    def press(self, direction):
        self.direction = direction
        self.held_ms = 0
        self.has_repeated_once = False

    def release(self, direction):
        if self.direction == direction:
            self.direction = 0

    def update(self, dt):
        """Returns True if a move should fire this frame."""
        if self.direction == 0:
            return False
        self.held_ms += dt
        if not self.has_repeated_once:
            if self.held_ms >= DAS_INITIAL_DELAY_MS:
                self.has_repeated_once = True
                self.held_ms = 0
                return True
            return False
        else:
            if self.held_ms >= DAS_REPEAT_MS:
                self.held_ms = 0
                return True
            return False


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("NES-style Tetris")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)

    board = Board()
    randomizer = NESRandomizer()
    current = Piece(randomizer.next_piece())
    next_shape = randomizer.next_piece()

    fall_timer = 0
    repeater = HeldKeyRepeater()
    soft_dropping = False
    running = True

    while running:
        dt = clock.tick(60)
        fall_timer += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    try_move(board, current, dx=-1)
                    repeater.press(-1)
                elif event.key == pygame.K_RIGHT:
                    try_move(board, current, dx=1)
                    repeater.press(1)
                elif event.key == pygame.K_UP:
                    try_move(board, current, drotation=1)
                elif event.key == pygame.K_DOWN:
                    soft_dropping = True
            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    repeater.release(-1)
                elif event.key == pygame.K_RIGHT:
                    repeater.release(1)
                elif event.key == pygame.K_DOWN:
                    soft_dropping = False

        if repeater.update(dt):
            try_move(board, current, dx=repeater.direction)

        interval = SOFT_DROP_INTERVAL_MS if soft_dropping else gravity_interval_ms(board.level)
        if fall_timer >= interval:
            fall_timer = 0
            if not try_move(board, current, dy=1):
                board.lock_piece(current)
                board.clear_lines()
                current = Piece(next_shape)
                next_shape = randomizer.next_piece()
                if not board.is_valid(current.cells()):
                    running = False  # top-out

        draw_board(screen, board, font, next_shape)
        draw_piece(screen, current)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
