import numpy as np


def rollDice():
    return np.random.randint(1,6,6)

def calcScore(roll):
    """roll is a Numpy array"""
    # Check for a straight
    print("Roll:   " + str(np.sort(roll)))
    if len(np.unique(roll))==6:
        return 1500
    else:
        score = 0

    # Count of each number's occurence
    counts = np.histogram(roll, bins=6, range=(1,6))[0]        
    print("Counts: " + str(counts))

    # Check for three of a kind
    while (counts>=3).any():
        score = score + np.dot(counts>=3,[1000, 200, 300, 400, 500, 600])
        counts[counts>=3] = counts[counts>=3]-3

    # Check for singles (1's and 5's)
    score = score + np.dot(counts,[100,0,0,0,50,0])

    return score

roll = rollDice()
score = calcScore(roll)
print("Score:  " + str(score))