"""
Full pipeline test, v2: Tetris board -> real fly eye (flyvis) -> real
central brain connectivity (neuPrint, two pathways) -> descending
neuron activity.

Every stage here uses real, measured biology. Nothing is trained yet,
that's the next milestone: a small readout turning this descending
neuron activity into an actual Tetris move.
"""

import copy

from tetris import Board, Piece
from eye_adapter import TetrisEye
from central_brain import CentralBrainLayer


def build_falling_sequence(n_frames=8):
    board = Board()
    piece = Piece("T")
    snapshots = []
    for _ in range(n_frames):
        snapshots.append((board, copy.deepcopy(piece)))
        piece.y += 1
    return snapshots


def main():
    print("Building a falling-piece sequence...")
    sequence = build_falling_sequence(n_frames=8)

    print("Loading the pretrained fly visual network...")
    eye = TetrisEye()

    print("Running the sequence through the real visual system...")
    responses = eye.process_sequence(sequence)

    central_layer = CentralBrainLayer()
    needed_types = central_layer.available_input_types()

    print(f"\nReading out activity for {len(needed_types)} flyvis cell types...")
    flyvis_activity = {}
    for cell_type in needed_types:
        try:
            # Last frame's central-cell activity as a snapshot value.
            value = responses.central[cell_type].squeeze()[-1].item()
            flyvis_activity[cell_type] = value
        except (KeyError, ValueError):
            pass  # shouldn't happen now, these types were confirmed available

    print(f"Got real activity for {len(flyvis_activity)} of {len(needed_types)} types.")

    print("\nComputing descending neuron activity through the real central brain wiring...")
    dn_activity = central_layer.compute_dn_activity(flyvis_activity)

    print("\nDescending neuron activity (real weights, real visual input, no training):")
    for dn_type, value in sorted(dn_activity.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {dn_type}: {value:.4f}")


if __name__ == "__main__":
    main()
