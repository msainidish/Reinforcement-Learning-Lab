# GridWorld - Policy Evaluation and Value Iteration

states = ["S1", "S2", "S3", "S4", "S5"]

actions = ["Left", "Right"]

gamma = 0.9

# Terminal state
goal = "S5"


# -----------------------------
# Transition function
# -----------------------------
def get_next_state(state, action):

    index = states.index(state)

    if state == goal:
        return goal

    if action == "Right":
        next_index = min(index + 1, len(states) - 1)

    else:
        next_index = max(index - 1, 0)

    return states[next_index]


# -----------------------------
# Reward function
# -----------------------------
def get_reward(next_state):

    if next_state == goal:
        return 10

    return -1


# =========================================================
# TASK 2: POLICY EVALUATION
# =========================================================

# Policy: always move Right
policy = {
    "S1": "Right",
    "S2": "Right",
    "S3": "Right",
    "S4": "Right"
}

# Initial values
V = {
    "S1": 0,
    "S2": 0,
    "S3": 0,
    "S4": 0,
    "S5": 10
}

print("POLICY EVALUATION")
print("-" * 40)

for iteration in range(1, 6):

    old_V = V.copy()

    for state in states:

        if state == goal:
            continue

        action = policy[state]

        next_state = get_next_state(state, action)

        reward = get_reward(next_state)

        V[state] = reward + gamma * old_V[next_state]

    delta = max(abs(V[s] - old_V[s]) for s in states)

    status = "Converged" if delta == 0 else "Not converged"

    print(
        "Iteration:", iteration,
        " Maximum Change:", round(delta, 3),
        status
    )

print("\nFinal Values:")

for state in states:
    print(state, "=", round(V[state], 3))


# =========================================================
# TASK 3: VALUE ITERATION
# =========================================================

V = {
    "S1": 0,
    "S2": 0,
    "S3": 0,
    "S4": 0,
    "S5": 10
}

print("\n\nVALUE ITERATION")
print("-" * 40)

for iteration in range(10):

    old_V = V.copy()

    for state in states:

        if state == goal:
            continue

        action_values = []

        for action in actions:

            next_state = get_next_state(state, action)

            reward = get_reward(next_state)

            value = reward + gamma * old_V[next_state]

            action_values.append(value)

        V[state] = max(action_values)

    delta = max(abs(V[s] - old_V[s]) for s in states)

    if delta == 0:
        break


# Find optimal action
optimal_policy = {}

for state in states:

    if state == goal:
        optimal_policy[state] = "Goal"
        continue

    best_action = None
    best_value = float("-inf")

    for action in actions:

        next_state = get_next_state(state, action)

        reward = get_reward(next_state)

        value = reward + gamma * V[next_state]

        if value > best_value:
            best_value = value
            best_action = action

    optimal_policy[state] = best_action


print("\nOptimal Policy:")

for state in states:
    print(
        state,
        "->",
        optimal_policy[state],
        " Value =",
        round(V[state], 3)
    )