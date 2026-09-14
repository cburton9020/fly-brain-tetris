"""
Watch the trained fly-brain readout actually play Tetris, rendered in a
real pygame window.

Uses TetrisHeadlessEnv for the actual game logic, guaranteeing this
behaves exactly like what the readout was trained against, just with a
window on top instead of running headless. Decisions come from the real
frozen eye + central brain pipeline and the trained readout, using the
same decision cadence (DECISION_INTERVAL, FRAME_WINDOW) as training, so
what you see matches what was actually learned.

Playback runs at a fixed, watchable frame rate (PLAYBACK_FPS below),
independent of the raw gravity speed used during training, purely so a
human can actually follow what's happening. When a game ends (top-out
or hitting the step cap), it automatically starts a new one so you can
just watch continuously.

Run with:
    python watch_agent_play.py
"""

import copy

import pygame
import torch

from tetris import draw_board, draw_piece, SCREEN_WIDTH, SCREEN_HEIGHT, BG_COLOR
from tetris_env import TetrisHeadlessEnv, ACTION_NONE
from eye_adapter import TetrisEye
from central_brain import CentralBrainLayer
from readout import ActionReadout
from train_agent import DECISION_INTERVAL, FRAME_WINDOW, SAVE_PATH

PLAYBACK_FPS = 12  # how fast to actually show it, independent of sim speed
GREEDY = True      # True: always take the highest-probability action (clean,
                    # deterministic, shows what the policy "really thinks" is
                    # best). False: sample stochastically, matching exactly
                    # how it behaved during training (noisier to watch).


def choose_action(readout, dn_activity):
    x = readout.activity_dict_to_tensor(dn_activity)
    logits = readout.forward(x)
    if GREEDY:
        return int(torch.argmax(logits).item())
    else:
        action, _ = readout.act(dn_activity)
        return action


def main():
    print("Loading the frozen pipeline (eye + central brain)...")
    eye = TetrisEye()
    central_layer = CentralBrainLayer()

    print(f"Loading trained readout from {SAVE_PATH}...")
    readout = ActionReadout(central_layer.dn_types)
    readout.load_state_dict(torch.load(SAVE_PATH))
    readout.eval()

    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Fly Brain Plays Tetris")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 26)

    env = TetrisHeadlessEnv()
    env.reset()

    frame_window = []
    current_action = ACTION_NONE
    step = 0
    games_played = 0
    running = True

    print("Watching... close the window or press Esc to stop.")

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        board, piece = env._get_observation()
        frame_window.append((board, copy.deepcopy(piece)))
        if len(frame_window) > FRAME_WINDOW:
            frame_window.pop(0)

        if step % DECISION_INTERVAL == 0 and len(frame_window) == FRAME_WINDOW:
            responses = eye.process_sequence_batch([frame_window])

            flyvis_activity = {}
            for cell_type in central_layer.available_input_types():
                try:
                    flyvis_activity[cell_type] = responses.central[cell_type][0, -1].item()
                except (KeyError, ValueError):
                    pass

            dn_activity = central_layer.compute_dn_activity(flyvis_activity)
            current_action = choose_action(readout, dn_activity)

        _, reward, done = env.step(current_action)
        step += 1

        draw_board(screen, env.board, font, env.next_shape)
        draw_piece(screen, env.current_piece)
        pygame.display.flip()
        clock.tick(PLAYBACK_FPS)

        if done or step >= 800:
            games_played += 1
            print(f"Game {games_played} ended: score={env.board.score}, "
                  f"lines={env.board.lines_cleared}, steps={step}")
            env.reset()
            frame_window = []
            current_action = ACTION_NONE
            step = 0

    pygame.quit()


if __name__ == "__main__":
    main()
