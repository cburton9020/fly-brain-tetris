"""
Full pipeline test: Tetris board -> real fly eye (flyvis) -> real central
brain connectivity (neuPrint) -> descending neuron activity.

This is the moment of truth for whether LC4, LC6, and LPLC2 (the visual
types we found real DN connections for) are actually cell types flyvis's
optic lobe model produces activity for. If any aren't, we'll see exactly
which ones and can decide how to bridge that gap next.
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

    print(f"\nChecking which of {needed_types} are actually available from flyvis...")
    cell_type_activity = {}
    unavailable = []

    for cell_type in needed_types:
        try:
            # Use the last frame's central-cell activity as a single
            # snapshot value to feed into the fixed central brain weights.
            value = responses.central[cell_type].squeeze()[-1].item()
            cell_type_activity[cell_type] = value
        except (KeyError, ValueError):
            unavailable.append(cell_type)

    if cell_type_activity:
        print(f"Available: {list(cell_type_activity.keys())}")
    if unavailable:
        print(f"NOT available from flyvis's model: {unavailable}")

    if not cell_type_activity:
        print(
            "\nNone of the visual types with real DN connections are "
            "available from flyvis's model. We'd need to either find a "
            "different, earlier cell type flyvis DOES model that also "
            "connects to these DNs (possibly through an extra hop), or "
            "pull in additional neuPrint connectivity for whichever "
            "flyvis output types we DO have."
        )
        return

    print("\nComputing descending neuron activity from what's available...")
    dn_activity = central_layer.compute_dn_activity(cell_type_activity)

    print("\nDescending neuron activity (real weights, real visual input):")
    for dn_type, value in sorted(dn_activity.items(), key=lambda kv: -abs(kv[1])):
        print(f"  {dn_type}: {value:.4f}")


if __name__ == "__main__":
    main()
