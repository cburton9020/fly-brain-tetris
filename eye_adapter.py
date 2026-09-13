"""
Board-to-hex-eye adapter.

Converts the Tetris board (a plain grid from tetris.Board) into the format
flyvis expects, then runs it through the real, pretrained, connectome-
constrained visual system model.

Before running this, you need flyvis's pretrained weights on disk. From an
activated venv, run once:

    flyvis download-pretrained

This pulls the pretrained ensemble down to flyvis's data directory (a few
hundred MB), so it only needs to be done once.

Pipeline:
    Board.grid + current Piece
      -> grayscale image (numpy)
      -> BoxEye hex-lattice rendering (flyvis, 721 receptor values)
      -> pretrained connectome-constrained network (flyvis)
      -> per-cell-type activity (e.g. motion-sensitive cells)

A single static frame won't show much through the motion-detecting cells,
since those respond to change over time. This module is built to accept a
short sequence of frames (e.g. a few frames as a piece falls one row) so
there's actual motion for the network to respond to.
"""

import numpy as np
import torch

import flyvis
from flyvis.datasets.rendering import BoxEye
from flyvis.utils.activity_utils import LayerActivity

from tetris import BOARD_WIDTH, BOARD_HEIGHT

# Pixels per board cell when we rasterize the grid into an image. This is
# independent of the pygame window, purely for feeding the eye model, so it
# can be tuned without touching the game's rendering.
BLOCK_PX = 20

# flyvis's BoxEye default: 31 columns across the hex array (721 receptors
# total), each sampling a 13x13 pixel region.
HEX_EXTENT = 15
HEX_KERNEL_SIZE = 13

# Integration time step used across flyvis's tutorials for this kind of
# rendered stimulus.
DT = 1 / 100

# How long (in simulated seconds) to fade in from a blank grey stimulus
# before looking at real input. fade_in_state runs int(FADE_IN_DURATION/DT)
# full network timesteps, so this number directly controls a real,
# significant cost, don't set it to a full second without a reason: 1.0s
# means 100 real integration steps, purely to "warm up" the network,
# every single time it's called. A much shorter ramp still moves the
# network well away from flat grey.
FADE_IN_DURATION = 0.1


def board_to_image(board, current_piece=None, block_px=BLOCK_PX):
    """
    Rasterize the board grid (plus the currently falling piece, if given)
    into a single-channel grayscale image as a numpy float32 array, values
    in [0, 1]. Filled cells are white, empty cells are black.
    """
    height_px = BOARD_HEIGHT * block_px
    width_px = BOARD_WIDTH * block_px
    image = np.zeros((height_px, width_px), dtype=np.float32)

    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            if board.grid[row][col] is not None:
                image[
                    row * block_px:(row + 1) * block_px,
                    col * block_px:(col + 1) * block_px,
                ] = 1.0

    if current_piece is not None:
        for col, row in current_piece.cells():
            if 0 <= row < BOARD_HEIGHT and 0 <= col < BOARD_WIDTH:
                image[
                    row * block_px:(row + 1) * block_px,
                    col * block_px:(col + 1) * block_px,
                ] = 1.0

    return image


class TetrisEye:
    """
    Wraps flyvis's BoxEye renderer and pretrained visual network so the
    rest of the project can just hand it board states and get back real
    connectome-constrained neural activity.
    """

    def __init__(self, network_index=0):
        """
        network_index picks one of the 50 pretrained networks in the
        ensemble (0 = lowest task error, i.e. the "best" one, per flyvis's
        own ranking; any index 0-49 is valid).
        """
        self.receptors = BoxEye(extent=HEX_EXTENT, kernel_size=HEX_KERNEL_SIZE)

        model_dir = flyvis.results_dir / "flow" / "0000" / f"{network_index:03d}"
        self.network_view = flyvis.NetworkView(model_dir)
        self.network = self.network_view.init_network()

    def render_frame(self, board, current_piece=None):
        """
        Board + piece -> a single hex-rendered frame, shape (721,).
        Useful for a quick look at what the fly's eye "sees" right now.
        """
        image = board_to_image(board, current_piece)
        frame_tensor = torch.tensor(image, device=flyvis.device).float()[None, None]
        rendered = self.receptors(frame_tensor)  # (1, 1, 1, 721)
        return rendered.squeeze()

    def process_sequence(self, boards_and_pieces):
        """
        Takes a list of (board, current_piece) snapshots, e.g. consecutive
        frames as a piece falls, and runs them through the pretrained
        network.

        Returns a LayerActivity object: index it like responses["T4c"] to
        get that cell type's activity over the sequence, shape
        (1, n_frames, 721), or responses.central["T4c"] for just the
        central cell of each type.
        """
        frames = [board_to_image(board, piece) for board, piece in boards_and_pieces]
        stacked = np.stack(frames, axis=0)  # (n_frames, H, W)
        # BoxEye expects (samples, frames, H, W); we treat this whole
        # sequence as one sample, so we add a single batch dimension.
        frame_tensor = torch.tensor(stacked, device=flyvis.device).float()[None]
        rendered = self.receptors(frame_tensor)  # (1, n_frames, 1, 721) = (samples, frames, 1, hexals)

        # network.simulate wants exactly this 4D shape, don't squeeze it.
        stationary_state = self.network.fade_in_state(FADE_IN_DURATION, DT, rendered[:, 0])
        responses = self.network.simulate(
            rendered, DT, initial_state=stationary_state
        ).cpu()

        return LayerActivity(responses, self.network.connectome, keepref=True)

    def process_sequence_batch(self, list_of_sequences):
        """
        Batched version of process_sequence: takes a list of N sequences
        (each a list of (board, current_piece) snapshots, all the same
        length), and runs all N through the network in a single call.

        This is the key speedup for parallel training: instead of N
        separate, sequential calls to the real neural network (slow),
        we do one call handling all N games at once, which GPUs handle
        far more efficiently.

        Returns a LayerActivity where responses.central[cell_type] has
        shape (N, n_frames) instead of (1, n_frames), index the first
        dimension by which environment you want.
        """
        import time as _time
        t_start = _time.time()

        batch_frames = []
        for sequence in list_of_sequences:
            frames = [board_to_image(board, piece) for board, piece in sequence]
            batch_frames.append(np.stack(frames, axis=0))  # (n_frames, H, W)

        batch = np.stack(batch_frames, axis=0)  # (N, n_frames, H, W)
        frame_tensor = torch.tensor(batch, device=flyvis.device).float()
        t_render_start = _time.time()
        rendered = self.receptors(frame_tensor)  # (N, n_frames, 1, 721)
        t_render_end = _time.time()

        stationary_state = self.network.fade_in_state(FADE_IN_DURATION, DT, rendered[:, 0])
        t_fade_end = _time.time()

        responses = self.network.simulate(
            rendered, DT, initial_state=stationary_state
        ).cpu()
        t_simulate_end = _time.time()

        if getattr(self, "_debug_timing", False):
            print(
                f"    [timing] device={flyvis.device} | render={t_render_end - t_render_start:.2f}s "
                f"| fade_in={t_fade_end - t_render_end:.2f}s | simulate={t_simulate_end - t_fade_end:.2f}s "
                f"| total={t_simulate_end - t_start:.2f}s"
            )

        return LayerActivity(responses, self.network.connectome, keepref=True)


if __name__ == "__main__":
    # Quick smoke test: render one empty board frame and print its shape.
    # This won't show anything interesting (no motion, no filled cells),
    # it just confirms the pipeline runs end to end.
    from tetris import Board, Piece

    eye = TetrisEye()
    board = Board()
    piece = Piece("T")
    frame = eye.render_frame(board, piece)
    print("Rendered hex frame shape:", frame.shape)
