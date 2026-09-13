
import copy

from tetris import Board, Piece
from eye_adapter import TetrisEye


def build_falling_sequence(n_frames=8):
    """
    Returns a list of (board, piece) snapshots showing a T-piece falling
    straight down for n_frames steps, one row per frame.
    """
    board = Board()
    piece = Piece("T")

    snapshots = []
    for _ in range(n_frames):
        # Store a copy of the piece's current position for this frame.
        snapshots.append((board, copy.deepcopy(piece)))
        piece.y += 1  # move down one row for the next frame

    return snapshots


def main():
    print("Building a falling-piece sequence...")
    sequence = build_falling_sequence(n_frames=8)

    print("Loading the pretrained fly visual network (first run may take a moment)...")
    eye = TetrisEye()

    print("Running the sequence through the network...")
    responses = eye.process_sequence(sequence)

    # T4c is a real motion-sensitive cell type in the fly visual system.
    # responses.central gives just the central cell of each type, one
    # value per frame, which is the simplest thing to look at first.
    cell_type = "T4c"
    central_activity = responses.central[cell_type].squeeze()

    print(f"\n{cell_type} central cell activity across frames (a real fly motion detector):")
    for frame_index, value in enumerate(central_activity.tolist()):
        print(f"  frame {frame_index}: {value:.4f}")


if __name__ == "__main__":
    main()
