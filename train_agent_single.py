"""
Training loop: teaches the small readout to turn real descending neuron
activity into good Tetris moves, using REINFORCE (a simple policy
gradient method), while the eye, flyvis's visual network, and the
central brain connectivity all stay completely frozen throughout.

Important honesty note on speed: running the real flyvis network is
genuinely more expensive than a normal game loop, since it's simulating
actual neural dynamics. To keep training tractable, the agent only makes
a fresh perception+decision every DECISION_INTERVAL raw game steps
(repeating the same action for the steps in between), rather than every
single frame. Start with a small NUM_EPISODES to see how fast this runs
on your machine before committing to a long training session.
"""

import copy
import time

import torch
import torch.optim as optim

from tetris_env import TetrisHeadlessEnv, ACTION_NONE
from eye_adapter import TetrisEye
from central_brain import CentralBrainLayer
from readout import ActionReadout

# Decisions happen relative to how fast the piece is actually falling,
# not a fixed step count. This is what makes the trained readout's
# behavior transfer correctly to any gravity speed later (e.g. real NES
# timing), since the ratio of "decisions per row fallen" stays constant
# no matter how fast or slow gravity actually is.
DECISIONS_PER_ROW = 2

# How many recent raw frames to feed the eye each decision, so the
# motion-sensitive cells (T4/T5) have something to actually respond to.
FRAME_WINDOW = 3

MAX_STEPS_PER_EPISODE = 200
NUM_EPISODES = 50
GAMMA = 0.99  # discount factor for future rewards
LEARNING_RATE = 0.01
PRINT_EVERY = 1
SAVE_PATH = "readout_weights.pt"


def compute_discounted_returns(rewards, gamma):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    returns = torch.tensor(returns, dtype=torch.float32)
    # Normalize for training stability (standard REINFORCE trick).
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def run_episode(env, eye, central_layer, readout, episode_number, decision_interval):
    """Runs one full episode, returns (log_probs, rewards, total_score)."""
    env.reset()
    frame_window = []
    log_probs = []
    rewards = []

    step = 0
    decision_count = 0
    while step < MAX_STEPS_PER_EPISODE:
        board, piece = env._get_observation()
        frame_window.append((board, copy.deepcopy(piece)))
        if len(frame_window) > FRAME_WINDOW:
            frame_window.pop(0)

        # Only make a fresh decision every decision_interval steps, and
        # only once we have enough frames for real motion signal.
        if step % decision_interval == 0 and len(frame_window) == FRAME_WINDOW:
            responses = eye.process_sequence(frame_window)
            flyvis_activity = {}
            for cell_type in central_layer.available_input_types():
                try:
                    flyvis_activity[cell_type] = (
                        responses.central[cell_type].squeeze()[-1].item()
                    )
                except (KeyError, ValueError):
                    pass
            dn_activity = central_layer.compute_dn_activity(flyvis_activity)
            action, log_prob = readout.act(dn_activity)
            log_probs.append(log_prob)
            current_action = action
            decision_count += 1
            if decision_count % 10 == 0:
                print(
                    f"  [episode {episode_number}] {decision_count} decisions made "
                    f"so far, step {step}...", flush=True
                )
        else:
            current_action = ACTION_NONE

        _, reward, done = env.step(current_action)
        rewards.append(reward)
        step += 1

        if done:
            break

    return log_probs, rewards, env.board.score


def main():
    print("Setting up the frozen pipeline (eye + central brain)...")
    eye = TetrisEye()
    central_layer = CentralBrainLayer()

    print("Setting up the trainable readout...")
    readout = ActionReadout(central_layer.dn_types)
    optimizer = optim.Adam(readout.parameters(), lr=LEARNING_RATE)

    env = TetrisHeadlessEnv()

    # Derive decision frequency from the environment's actual gravity
    # speed, so "decisions per row fallen" stays constant regardless of
    # how fast or slow gravity is set.
    decision_interval = max(1, env.gravity_period // DECISIONS_PER_ROW)
    print(
        f"Gravity period: {env.gravity_period} steps/row -> deciding every "
        f"{decision_interval} step(s) (~{DECISIONS_PER_ROW} decisions per row fallen)\n"
    )

    print(f"Training for up to {NUM_EPISODES} episodes...\n")
    for episode in range(1, NUM_EPISODES + 1):
        start_time = time.time()
        log_probs, rewards, score = run_episode(
            env, eye, central_layer, readout, episode, decision_interval
        )
        elapsed = time.time() - start_time

        if not log_probs:
            continue  # episode ended before any decision was made, skip

        # A reward is collected for every raw step, but decisions only
        # happen every decision_interval steps. Group rewards into chunks
        # matching each decision so returns line up with log_probs.
        grouped_rewards = [
            sum(rewards[i:i + decision_interval])
            for i in range(0, len(rewards), decision_interval)
        ][:len(log_probs)]

        returns = compute_discounted_returns(grouped_rewards, GAMMA)

        loss = 0.0
        for log_prob, ret in zip(log_probs, returns):
            loss -= log_prob * ret

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if episode % PRINT_EVERY == 0:
            total_reward = sum(rewards)
            print(
                f"Episode {episode:4d} | steps: {len(rewards):4d} | "
                f"score: {score:4d} | total reward: {total_reward:7.2f} | "
                f"loss: {loss.item():7.3f} | took {elapsed:5.1f}s"
            )

    torch.save(readout.state_dict(), SAVE_PATH)
    print(f"\nSaved trained readout weights to {SAVE_PATH}")


if __name__ == "__main__":
    main()
