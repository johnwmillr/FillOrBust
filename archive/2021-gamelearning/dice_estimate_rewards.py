from collections import Counter
import pickle

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack

from dice.agents import Smart
from dice.cards import Card, make_a_fill_1000, make_a_bonus_card, make_a_no_dice_card
from dice.q import convert_state_to_Q_table_key
from dice.types import ActionT, State, StateT


# Load the trained encoder and estimator
with open(
    "./dice/dice_trained_encoder_RandomForestRegressor_1710095106.pkl", "rb"
) as fin:
    encoder: OneHotEncoder = pickle.load(fin)
with open(
    "./dice/dice_trained_estimator_RandomForestRegressor_1710095106.pkl", "rb"
) as fin:
    estimator: RandomForestRegressor = pickle.load(fin)


def predict_reward_for_state_and_action(
    card: Card,
    discard: Counter[Card],
    banked_score: int,
    turn_score: int,
    dice: Counter[int],
    holds: Counter[int],
    action: ActionT,
):
    state: State = State(
        card=card,
        discard=discard,
        banked_score=banked_score,
        turn_score=turn_score,
        dice=dice,
        holds=holds,
    )
    numerical_columns = ["banked_score", "turn_score"]
    categorical_features: list[str] = []
    numerical_features: list[int] = []
    for item in convert_state_to_Q_table_key(state).split("+"):
        if item.rsplit("_", 1)[0] in numerical_columns:
            numerical_features.append(int(item.split("_")[-1]))
        else:
            categorical_features.append(item)
    categorical_features.append(action)
    encoded_categorical_features = encoder.transform([categorical_features])
    return estimator.predict(hstack((numerical_features, encoded_categorical_features)))


predict_reward_for_state_and_action(
    card=make_a_bonus_card(300),
    discard=Counter(),
    banked_score=950,
    turn_score=50,
    dice=Counter([]),
    holds=Counter([]),
    action="stop",
)

print("Done!")
