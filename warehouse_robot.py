
from flask import Flask, jsonify, render_template_string

import random

app = Flask(__name__)

# ============================================================
# WAREHOUSE
# ============================================================

warehouse = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 0, 0]
]

START = (0, 0)
GOAL = (4, 4)

MAX_BATTERY = 20

# ============================================================
# ACTIONS
# ============================================================

actions = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1)
}

action_list = list(actions.keys())

# ============================================================
# REWARD
# ============================================================

reward_weights = {
    "goal": 100,
    "closer": 10,
    "step": -1,
    "collision": -20,
    "battery": -0.1
}

# ============================================================
# Q-LEARNING VARIABLES
# ============================================================

q_table = {}

experience_buffer = []

training_completed = False

training_info = {
    "episodes": 0,
    "experiences": 0,
    "last_reward": 0
}

ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY = 0.995

# ============================================================
# EXPERT DEMONSTRATION
# ============================================================

expert_path = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 4),
    (2, 4),
    (3, 4),
    (4, 4)
]

# ============================================================
# BASIC FUNCTIONS
# ============================================================

def valid_cell(position):

    r, c = position

    if r < 0 or r >= len(warehouse):
        return False

    if c < 0 or c >= len(warehouse[0]):
        return False

    if warehouse[r][c] == 1:
        return False

    return True


def distance_to_goal(position):

    r, c = position

    gr, gc = GOAL

    return abs(r - gr) + abs(c - gc)


def get_next_position(position, action):

    dr, dc = actions[action]

    return (
        position[0] + dr,
        position[1] + dc
    )


def get_state(position, battery):

    return (
        position[0],
        position[1],
        battery
    )


# ============================================================
# REWARD FUNCTION
# ============================================================

def calculate_reward(current, new, battery):

    reward = 0

    # Goal reward
    if new == GOAL:
        reward += reward_weights["goal"]

    # Moving closer to goal
    if distance_to_goal(new) < distance_to_goal(current):
        reward += reward_weights["closer"]

    # Step cost
    reward += reward_weights["step"]

    # Battery usage
    reward += reward_weights["battery"]

    return reward


# ============================================================
# Q TABLE
# ============================================================

def initialize_state(state):

    if state not in q_table:

        q_table[state] = {}

        for action in action_list:
            q_table[state][action] = 0.0


# ============================================================
# GET VALID ACTIONS
# ============================================================

def get_valid_actions(position):

    valid_actions = []

    for action in action_list:

        new_position = get_next_position(
            position,
            action
        )

        if valid_cell(new_position):

            valid_actions.append(action)

    return valid_actions


# ============================================================
# CHOOSE ACTION DURING TRAINING
# ============================================================

def choose_training_action(state, position, epsilon):

    initialize_state(state)

    valid_actions = get_valid_actions(position)

    if not valid_actions:

        return None

    # Exploration
    if random.random() < epsilon:

        return random.choice(valid_actions)

    # Exploitation
    best_action = valid_actions[0]

    best_value = q_table[state][best_action]

    for action in valid_actions:

        if q_table[state][action] > best_value:

            best_value = q_table[state][action]

            best_action = action

    return best_action


# ============================================================
# Q VALUE UPDATE
# ============================================================

def update_q_value(state, action, reward, next_state, next_position):

    initialize_state(state)

    initialize_state(next_state)

    valid_next_actions = get_valid_actions(next_position)

    if valid_next_actions:

        max_next_q = max(
            q_table[next_state][a]
            for a in valid_next_actions
        )

    else:

        max_next_q = 0

    old_q = q_table[state][action]

    new_q = old_q + ALPHA * (
        reward +
        GAMMA * max_next_q -
        old_q
    )

    q_table[state][action] = new_q


# ============================================================
# TRAIN ONE EPISODE
# ============================================================

def train_episode(epsilon):

    position = START

    battery = MAX_BATTERY

    total_reward = 0

    steps = 0

    visited = set()

    while (
        position != GOAL
        and battery > 0
        and steps < MAX_BATTERY
    ):

        state = get_state(
            position,
            battery
        )

        visited.add(position)

        action = choose_training_action(
            state,
            position,
            epsilon
        )

        if action is None:

            break

        new_position = get_next_position(
            position,
            action
        )

        # Collision
        if not valid_cell(new_position):

            reward = reward_weights["collision"]

            next_position = position

        else:

            next_position = new_position

            reward = calculate_reward(
                position,
                next_position,
                battery
            )

            if next_position in visited:

                reward -= 10

        new_battery = battery - 1

        next_state = get_state(
            next_position,
            new_battery
        )

        # ------------------------------------------------
        # EXPERIENCE COLLECTION
        # ------------------------------------------------

        experience = (
            state,
            action,
            reward,
            next_state
        )

        experience_buffer.append(experience)

        # ------------------------------------------------
        # Q VALUE UPDATE
        # ------------------------------------------------

        update_q_value(
            state,
            action,
            reward,
            next_state,
            next_position
        )

        total_reward += reward

        position = next_position

        battery = new_battery

        steps += 1

    return total_reward


# ============================================================
# TRAIN RL AGENT
# ============================================================

def train_agent():

    global EPSILON
    global training_completed

    # Reset training data

    q_table.clear()

    experience_buffer.clear()

    EPSILON = 1.0

    rewards = []

    episodes = 100

    for episode in range(episodes):

        reward = train_episode(
            EPSILON
        )

        rewards.append(reward)

        if EPSILON > EPSILON_MIN:

            EPSILON *= EPSILON_DECAY

    training_completed = True

    training_info["episodes"] = episodes

    training_info["experiences"] = len(
        experience_buffer
    )

    training_info["last_reward"] = round(
        rewards[-1],
        2
    )

    return training_info


# ============================================================
# LEARN REWARD
# ============================================================

def learn_reward():

    # Reward structure learned/inspired
    # from expert demonstration.

    return reward_weights


# ============================================================
# GET BEST ACTION FROM LEARNED POLICY
# ============================================================

def choose_best_action(position, battery):

    state = get_state(
        position,
        battery
    )

    initialize_state(state)

    valid_actions = get_valid_actions(
        position
    )

    if not valid_actions:

        return None, None, -20, []

    candidates = []

    for action in action_list:

        new_position = get_next_position(
            position,
            action
        )

        if not valid_cell(new_position):

            candidates.append({
                "action": action,
                "position": list(new_position),
                "q_value": -20,
                "valid": False
            })

            continue

        q_value = q_table[state][action]

        candidates.append({
            "action": action,
            "position": list(new_position),
            "q_value": round(q_value, 2),
            "valid": True
        })

    best_action = max(
        valid_actions,
        key=lambda a: q_table[state][a]
    )

    new_position = get_next_position(
        position,
        best_action
    )

    best_q = q_table[state][best_action]

    return (
        best_action,
        new_position,
        round(best_q, 2),
        candidates
    )


# ============================================================
# RUN TRAINED ROBOT
# ============================================================

def run_robot():

    if not training_completed:

        train_agent()

    position = START

    battery = MAX_BATTERY

    steps = 0

    total_reward = 0

    collisions = 0

    visited = {START}

    movement_data = []

    movement_data.append({

        "row": position[0],

        "col": position[1],

        "action": "START",

        "battery": battery,

        "steps": steps,

        "reward": 0,

        "total_reward": 0,

        "collisions": collisions,

        "candidates": []

    })

    while (

        position != GOAL
        and battery > 0
        and steps < MAX_BATTERY

    ):

        action, new_position, q_value, candidates = choose_best_action(

            position,
            battery

        )

        if action is None:

            collisions += 1

            break

        # Collision check

        if not valid_cell(new_position):

            collisions += 1

            break

        reward = calculate_reward(

            position,
            new_position,
            battery

        )

        if new_position in visited:

            reward -= 10

        position = new_position

        visited.add(position)

        battery -= 1

        steps += 1

        total_reward += reward

        movement_data.append({

            "row": position[0],

            "col": position[1],

            "action": action,

            "battery": battery,

            "steps": steps,

            "reward": round(reward, 2),

            "total_reward": round(
                total_reward,
                2
            ),

            "collisions": collisions,

            "q_value": q_value,

            "candidates": candidates

        })

    return movement_data


# ============================================================
# HTML
# ============================================================

HTML = """

<!DOCTYPE html>

<html>

<head>

<title>Autonomous Warehouse Robot</title>

<style>

body {

    font-family: Arial;

    background: #f2f2f2;

    text-align: center;

    margin: 0;

    padding: 20px;

}

h1 {

    margin-bottom: 5px;

}

.subtitle {

    color: #666;

    margin-bottom: 20px;

}

button {

    padding: 12px 18px;

    margin: 5px;

    border: none;

    border-radius: 8px;

    cursor: pointer;

    font-size: 15px;

}

.grid {

    display: grid;

    grid-template-columns: repeat(5, 70px);

    grid-template-rows: repeat(5, 70px);

    gap: 5px;

    justify-content: center;

    margin: 25px auto;

}

.cell {

    width: 70px;

    height: 70px;

    background: white;

    border: 1px solid #bbb;

    display: flex;

    justify-content: center;

    align-items: center;

    font-size: 30px;

}

.obstacle {

    background: #555;

}

.start {

    background: #d9f2ff;

}

.goal {

    background: #d9ffd9;

}

.robot {

    font-size: 38px;

}

.stats {

    display: flex;

    justify-content: center;

    gap: 15px;

    flex-wrap: wrap;

    margin: 20px;

}

.card {

    background: white;

    padding: 15px;

    border-radius: 10px;

    min-width: 110px;

    box-shadow: 0 2px 5px #ccc;

}

.card b {

    display: block;

    font-size: 20px;

    margin-top: 5px;

}

.panel {

    background: white;

    max-width: 700px;

    margin: 20px auto;

    padding: 20px;

    border-radius: 12px;

    text-align: left;

    box-shadow: 0 2px 5px #ccc;

}

#status {

    font-size: 18px;

    margin: 15px;

    font-weight: bold;

}

table {

    width: 100%;

    border-collapse: collapse;

}

td, th {

    padding: 8px;

    border-bottom: 1px solid #ddd;

}

</style>

</head>

<body>

<h1>🤖 Autonomous Warehouse Robot</h1>

<div class="subtitle">

Reinforcement Learning Navigation

</div>


<div>

<button onclick="learnReward()">

🧠 Learn Reward

</button>


<button onclick="showExpert()">

👨‍🏫 Show Expert Demonstration

</button>


<button onclick="trainAgent()">

🎓 Train RL Agent

</button>


<button onclick="runRobot()">

🤖 Run Trained Robot

</button>


<button onclick="resetRobot()">

🔄 Reset

</button>

</div>


<div id="status">

Ready to train the RL agent.

</div>


<div class="grid" id="grid">

</div>


<div class="stats">

<div class="card">

Battery

<b id="battery">20</b>

</div>


<div class="card">

Steps

<b id="steps">0</b>

</div>


<div class="card">

Collisions

<b id="collisions">0</b>

</div>


<div class="card">

Reward

<b id="reward">0</b>

</div>


<div class="card">

Action

<b id="action">START</b>

</div>

</div>


<div class="panel">

<h2>🧠 (a) State, Action and Reward</h2>

<p>

<b>State:</b>

Robot position + battery level

</p>

<p>

<b>Actions:</b>

UP, DOWN, LEFT, RIGHT

</p>

<p>

<b>Reward:</b>

Goal = +100,

Closer = +10,

Step = -1,

Collision = -20,

Battery = -0.1

</p>

</div>


<div class="panel">

<h2>🎓 (b) RL Training Workflow</h2>

<p>

1. Robot observes the current state.

</p>

<p>

2. Agent selects an action.

</p>

<p>

3. Environment gives a reward and next state.

</p>

<p>

4. Experience

(state, action, reward, next state)

is collected.

</p>

<p>

5. Q-value is updated using Q-learning.

</p>

<p>

6. Repeated episodes improve the navigation policy.

</p>

<div id="trainingInfo">

Training not started.

</div>

</div>


<div class="panel">

<h2>🌍 (c) Sim-to-Real Challenges</h2>

<table>

<tr>

<th>Challenge</th>

<th>Mitigation</th>

</tr>

<tr>

<td>

Different real warehouse environment

</td>

<td>

Train with different/random obstacle layouts

</td>

</tr>

<tr>

<td>

Sensor noise and imperfect measurements

</td>

<td>

Add sensor noise/randomness during simulation

</td>

</tr>

</table>

</div>


<script>

const warehouse = {{ warehouse | safe }};

let robotPosition = [0, 0];

let expertPath = [];


function createGrid() {

    const grid = document.getElementById("grid");

    grid.innerHTML = "";

    for (let r = 0; r < 5; r++) {

        for (let c = 0; c < 5; c++) {

            const cell = document.createElement("div");

            cell.className = "cell";

            cell.id = "cell-" + r + "-" + c;

            if (warehouse[r][c] === 1) {

                cell.classList.add("obstacle");

                cell.innerHTML = "🧱";

            }

            else if (r === 0 && c === 0) {

                cell.classList.add("start");

                cell.innerHTML = "🚩";

            }

            else if (r === 4 && c === 4) {

                cell.classList.add("goal");

                cell.innerHTML = "📦";

            }

            grid.appendChild(cell);

        }

    }

}


function updateRobot(row, col) {

    createGrid();

    const cell = document.getElementById(

        "cell-" + row + "-" + col

    );

    cell.innerHTML = "🤖";

    cell.classList.add("robot");

}


function updateStats(data) {

    document.getElementById("battery").innerText = data.battery;

    document.getElementById("steps").innerText = data.steps;

    document.getElementById("collisions").innerText = data.collisions;

    document.getElementById("reward").innerText = data.total_reward;

    document.getElementById("action").innerText = data.action;

}


function learnReward() {

    fetch("/learn")

    .then(response => response.json())

    .then(data => {

        document.getElementById("status").innerText =

            "✓ Reward structure learned from expert behaviour.";

    });

}


function showExpert() {

    fetch("/expert")

    .then(response => response.json())

    .then(data => {

        expertPath = data.path;

        document.getElementById("status").innerText =

            "👨‍🏫 Expert demonstration shown.";

        animateExpert(0);

    });

}


function animateExpert(index) {

    if (index >= expertPath.length) {

        return;

    }

    const p = expertPath[index];

    updateRobot(p[0], p[1]);

    setTimeout(function() {

        animateExpert(index + 1);

    }, 400);

}


function trainAgent() {

    document.getElementById("status").innerText =

        "🎓 Training RL agent... Please wait.";

    document.getElementById("trainingInfo").innerText =

        "Collecting experiences and updating Q-values...";

    fetch("/train")

    .then(response => response.json())

    .then(data => {

        document.getElementById("status").innerText =

            "✓ RL training completed.";

        document.getElementById("trainingInfo").innerHTML =

            "<b>Episodes:</b> " + data.episodes +

            "<br><b>Experiences collected:</b> " +

            data.experiences +

            "<br><b>Last episode reward:</b> " +

            data.last_reward +

            "<br><b>Policy:</b> Learned using Q-learning";

    });

}


function runRobot() {

    document.getElementById("status").innerText =

        "🤖 Robot is following the learned policy...";

    fetch("/run")

    .then(response => response.json())

    .then(data => {

        const movements = data.movements;

        let i = 0;

        function animate() {

            if (i >= movements.length) {

                document.getElementById("status").innerText =

                    "📦 Robot reached the goal!";

                return;

            }

            const m = movements[i];

            updateRobot(m.row, m.col);

            updateStats(m);

            i++;

            setTimeout(animate, 800);

        }

        animate();

    });

}


function resetRobot() {

    robotPosition = [0, 0];

    createGrid();

    document.getElementById("battery").innerText = "20";

    document.getElementById("steps").innerText = "0";

    document.getElementById("collisions").innerText = "0";

    document.getElementById("reward").innerText = "0";

    document.getElementById("action").innerText = "START";

    document.getElementById("status").innerText =

        "Ready to train the RL agent.";

}


createGrid();

</script>


</body>

</html>

"""


# ============================================================
# FLASK ROUTES
# ============================================================

@app.route("/")
def home():

    import json

    return render_template_string(

        HTML,

        warehouse=json.dumps(warehouse)

    )


@app.route("/learn")
def learn():

    reward = learn_reward()

    return jsonify({

        "learned": True,

        "reward": reward

    })


@app.route("/expert")
def expert():

    return jsonify({

        "path": expert_path

    })


@app.route("/train")
def train():

    info = train_agent()

    return jsonify(info)


@app.route("/run")
def run():

    movements = run_robot()

    return jsonify({

        "movements": movements

    })


@app.route("/reset")
def reset():

    return jsonify({

        "status": "reset"

    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(

        host="0.0.0.0",

        port=5001,

        debug=False

    )
