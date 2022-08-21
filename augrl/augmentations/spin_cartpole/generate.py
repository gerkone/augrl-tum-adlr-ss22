import sys

import pandas as pd

from .cartpole import CartPoleEnv

LABEL_FREQ = 100


if __name__ == "__main__":
    env = CartPoleEnv(timeout=1000, game=True)
    eps = 10 if len(sys.argv) == 1 else int(sys.argv[1])
    print("Collection {} episodes".format(eps))
    log = []
    prefs = []
    segment = 1
    total_steps = 0
    spins = 0
    for x in range(eps):
        print("Episode {} -- {}/{} with spin".format(x, spins, len(log)))
        env.reset()
        action = 0
        states = []
        actions = []
        steps = last_labeling = 0
        while True:
            action = env.render()
            if action is not None:
                steps += 1
                state, _, terminal, _ = env.step(action)
                states.append(state)
                actions.append(action)
                if terminal:
                    break
            if steps % LABEL_FREQ == 0 and last_labeling < steps:
                while True:
                    try:
                        preference = int(input("Spin (0/1)? \t"))
                        if preference not in [0, 1]:
                            raise ValueError
                    except ValueError:
                        print("What? ", end="")
                    else:
                        break
                if preference == 1:
                    spins += 1
                log.append(
                    {
                        "segment": segment,
                        "state": states,
                        "action": actions,
                        "preference": preference,
                        "size": len(states),
                    }
                )
                last_labeling = steps
                segment += 1
                total_steps += len(states)
                states = []
                actions = []
    results = pd.DataFrame(log)
    results.to_pickle(
        "augrl/augmentations/spin_cartpole/preferences/handmade_results_{}.pickle".format(
            eps
        )
    )
    print(
        "Saved {} segments from {} episodes ({} steps in total)".format(
            len(results), eps, total_steps
        )
    )
