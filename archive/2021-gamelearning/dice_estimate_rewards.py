from collections import Counter
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder

from dice.agents import Smart
from dice.cards import Card, make_a_fill_1000, make_a_bonus_card, make_a_no_dice_card
from dice.q import convert_state_to_q_key
from dice.types import ActionT, StateT


# Load the trained encoder and estimator
with open("dice_trained_encoder.pkl", "rb") as fin:
    encoder: OneHotEncoder = pickle.load(fin)
with open("dice_trained_estimator.pkl", "rb") as fin:
    estimator: RandomForestRegressor = pickle.load(fin)


# Load the trained agent
print("Loading the trained Agent.")
fp = "./dice/trained_agent_1709936115_25000.pkl"
with open(fp, "rb") as fin:
    agent: Smart = pickle.load(fin)


def predict_reward_for_state_and_action(
    card: Card,
    turn_score: int,
    dice: Counter[int],
    holds: Counter[int],
    action: ActionT,
):
    state: StateT = {
        "card": card,
        "turn_score": turn_score,
        "rolled": dice,
        "held": holds,
    }
    numerical = []
    categorical = []
    for item in convert_state_to_q_key(state).split("+"):
        if item.startswith("turn_score_"):
            numerical.append(int(item.rsplit("_")[-1]))
        else:
            categorical.append(item)
    categorical.append(action)
    encoded = encoder.transform([categorical])
    return estimator.predict(
        np.append(numerical, encoded).reshape((1, encoded.shape[1] + 1))
    )


predict_reward_for_state_and_action(
    card=make_a_bonus_card(500),
    turn_score=50,
    dice=Counter([]),
    holds=Counter([]),
    action="roll",
)

print("Done!")
