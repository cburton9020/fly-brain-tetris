"""
Training loop, v2: runs several Tetris games in parallel and batches
them through the real flyvis network in a single call per decision
point, instead of one game at a time. This is the main speedup lever,
GPUs are far more efficient processing a batch than doing the same
small amount of work many times in a row.

The eye, flyvis's pretrained visual network, and the central brain
connectivity all stay completely frozen throughout, exactly as before.
Only the small readout learns.

Decision frequency is still tied to gravity speed (DECISIONS_PER_ROW),
not a fixed step count, see tetris_env.py for why that matters for
transferring to different gravity speeds later.
"""

import copy
import time

import torch
import torch.optim as optim

from tetris_env import TetrisHeadlessEnv, ACTION_NONE
from eye_adapter import TetrisEye
from central_brain import CentralBrainLayer
from readout import ActionReadout

N_PARALLEL = 64         # how many games to run simultaneously, batched together
DECISIONS_PER_ROW = 2
FRAME_WINDOW = 3

MAX_STEPS_PER_EPISODE = 200
NUM_ITERATIONS = 15     # one iteration = one batch of N_PARALLEL parallel episodes
GAMMA = 0.99
LEARNING_RATE = 0.01
SAVE_PATH = "readout_weights.pt"


def compute_discounted_returns(rewards, gamma):
    returns = []
    running = 0.0
    for r in reversed(rewards):
        running = r + gamma * running
        returns.insert(0, running)
    returns = torch.tensor(returns, dtype=torch.float32)
    if returns.std() > 1e-6:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


def extract_flyvis_activity(responses, env_index, central_layer):
    """Pulls out one environment's cell-type activity from a batched
    LayerActivity result."""
    activity = {}
    for cell_type in central_layer.available_input_types():
        try:
            activity[cell_type] = responses.central[cell_type][env_index, -1].item()
        except (KeyError, ValueError):
            pass
    return activity


def run_parallel_episodes(envs, eye, central_layer, readout, decision_interval):
    """
    Runs N_PARALLEL environments simultaneously until every one is done
    or the step cap is hit. Returns per-environment (log_probs, rewards,
    score) lists, one entry per environment, plus timing breakdowns to
    help diagnose where time is actually going.
    """
    for env in envs:
        env.reset()

    n = len(envs)
    frame_windows = [[] for _ in range(n)]
    log_probs_per_env = [[] for _ in range(n)]
    rewards_per_env = [[] for _ in range(n)]
    current_actions = [ACTION_NONE] * n
    active = [True] * n
    total_network_time = [0.0]
    total_central_brain_time = [0.0]
    episode_start_time = time.time()
    decision_count = 0

    step = 0
    while step < MAX_STEPS_PER_EPISODE and any(active):
        active_indices = [i for i in range(n) if active[i]]

        for i in active_indices:
            board, piece = envs[i]._get_observation()
            frame_windows[i].append((board, copy.deepcopy(piece)))
            if len(frame_windows[i]) > FRAME_WINDOW:
                frame_windows[i].pop(0)

        ready_indices = [
            i for i in active_indices if len(frame_windows[i]) == FRAME_WINDOW
        ]

        if step % decision_interval == 0 and ready_indices:
            sequences = [frame_windows[i] for i in ready_indices]

            t0 = time.time()
            responses = eye.process_sequence_batch(sequences)
            network_time = time.time() - t0

            t0 = time.time()
            dn_activity_dicts = []
            for batch_position, env_index in enumerate(ready_indices):
                flyvis_activity = extract_flyvis_activity(responses, batch_position, central_layer)
                dn_activity_dicts.append(central_layer.compute_dn_activity(flyvis_activity))
            central_brain_time = time.time() - t0

            actions, log_probs = readout.act_batch(dn_activity_dicts)

            for batch_position, env_index in enumerate(ready_indices):
                current_actions[env_index] = actions[batch_position]
                log_probs_per_env[env_index].append(log_probs[batch_position])

            total_network_time[0] += network_time
            total_central_brain_time[0] += central_brain_time
            decision_count += 1
            if decision_count % 20 == 0:
                print(f"    ...{decision_count} decisions so far, {time.time() - episode_start_time:.1f}s elapsed", flush=True)

        for i in active_indices:
            _, reward, done = envs[i].step(current_actions[i])
            rewards_per_env[i].append(reward)
            if done:
                active[i] = False

        step += 1

    scores = [env.board.score for env in envs]
    total_elapsed = time.time() - episode_start_time
    unaccounted = total_elapsed - total_network_time[0] - total_central_brain_time[0]
    print(
        f"  [breakdown] {decision_count} decisions | wall-clock: {total_elapsed:.1f}s | "
        f"network: {total_network_time[0]:.1f}s | central brain: {total_central_brain_time[0]:.1f}s | "
        f"UNACCOUNTED (game loop / bookkeeping): {unaccounted:.1f}s"
    )
    return log_probs_per_env, rewards_per_env, scores, total_network_time[0], total_central_brain_time[0]


def main():
    print("Setting up the frozen pipeline (eye + central brain)...")
    eye = TetrisEye()
    central_layer = CentralBrainLayer()

    print("Setting up the trainable readout...")
    readout = ActionReadout(central_layer.dn_types)
    optimizer = optim.Adam(readout.parameters(), lr=LEARNING_RATE)

    envs = [TetrisHeadlessEnv() for _ in range(N_PARALLEL)]
    decision_interval = max(1, envs[0].gravity_period // DECISIONS_PER_ROW)
    print(
        f"Running {N_PARALLEL} games in parallel per iteration. "
        f"Gravity period: {envs[0].gravity_period} steps/row -> deciding every "
        f"{decision_interval} step(s) (~{DECISIONS_PER_ROW} decisions per row fallen)\n"
    )

    # Print detailed per-call timing was used earlier to diagnose a real
    # print-overhead bottleneck (verbose per-decision logging was itself
    # slower than the actual GPU computation on this machine). Leave off
    # by default now that the cause is known; the [breakdown] line below
    # still reports accurate real timing per iteration.
    eye._debug_timing = False

    print(f"Training for up to {NUM_ITERATIONS} iterations "
          f"({N_PARALLEL} parallel episodes each)...\n")

    for iteration in range(1, NUM_ITERATIONS + 1):
        start_time = time.time()
        log_probs_per_env, rewards_per_env, scores, network_time, central_brain_time = run_parallel_episodes(
            envs, eye, central_layer, readout, decision_interval
        )
        elapsed = time.time() - start_time

        if iteration == 1:
            eye._debug_timing = False  # only print detailed timing once

        total_loss = 0.0
        episode_total_rewards = []

        for log_probs, rewards in zip(log_probs_per_env, rewards_per_env):
            episode_total_rewards.append(sum(rewards))
            if not log_probs:
                continue

            grouped_rewards = [
                sum(rewards[i:i + decision_interval])
                for i in range(0, len(rewards), decision_interval)
            ][:len(log_probs)]

            returns = compute_discounted_returns(grouped_rewards, GAMMA)
            for log_prob, ret in zip(log_probs, returns):
                total_loss -= log_prob * ret

        optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            optimizer.step()
            loss_value = total_loss.item()
        else:
            # No environment made it to a single decision this iteration
            # (shouldn't normally happen, but safe to skip rather than crash).
            loss_value = 0.0

        avg_score = sum(scores) / len(scores)
        avg_reward = sum(episode_total_rewards) / len(episode_total_rewards)
        print(
            f"Iteration {iteration:4d} | avg score: {avg_score:6.2f} | "
            f"avg reward: {avg_reward:7.2f} | loss: {loss_value:8.3f} | "
            f"took {elapsed:5.1f}s total (network: {network_time:5.1f}s, "
            f"central brain: {central_brain_time:5.1f}s)"
        )

    torch.save(readout.state_dict(), SAVE_PATH)
    print(f"\nSaved trained readout weights to {SAVE_PATH}")


if __name__ == "__main__":
    main()
