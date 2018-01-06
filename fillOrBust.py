import numpy as np
import time

class Turn(object):

    def __init__(self):
        self._dice_remaining = 6
        self._score = 0
        self._dice = []
        self._fill = False
        self._bust = False
        self._stopped = False
        
        # Pointer values
        self.__TRIPLE_VALUES = [1000, 200, 300, 400, 500, 600]
        self.__SINGLE_VALUES = [ 100,   0,   0,   0,  50,   0]

    def roll(self):
        assert not self._bust, "Can't roll again, you busted."
        assert not self._fill, "Can't roll again, you filled."
        assert not self._stopped, "Can't roll again, you stopped."
        assert self._dice_remaining > 0 and self._dice_remaining <= 6, "Number of dice must be between 1 and 6."        
        self._dice = np.sort(np.random.randint(1,7,self._dice_remaining))
        self.calcScore()
        return self

    def stop(self):
        assert not self._bust, "You can't stop, you already busted."
        self._stopped = True

    def calcScore(self):      

        # Check for a straight
        if self._dice_remaining == 6:
            if (self._dice==[1,2,3,4,5,6]).all():
                self._score, self._dice_remaining = 1500, 0            
                self._fill, self._bust = True, False
                return                    
        dice_remaining = self._dice_remaining

        # Count of each number's occurence
        counts = np.histogram(self._dice, bins=6, range=(1,6))[0]

        # Check for three of a kind
        score = 0
        while (counts>=3).any():
            score = score + np.dot(counts>=3, self.__TRIPLE_VALUES)
            dice_remaining = dice_remaining - 3*sum(counts>=3)
            counts[counts>=3] = counts[counts>=3]-3

        # Implement some strategy
        # if counts[0] > 0:
            # counts[4] = 0 # Throw back fives
        # if counts[4] > 0:
            # counts[1] = 0 # Throw back ones

        # Check for singles (1's and 5's)
        score = score + np.dot(counts, self.__SINGLE_VALUES)
        dice_remaining = dice_remaining - np.dot(counts,[1,0,0,0,1,0])

        if score == 0:
            self._score, self._dice_remaining = 0, 0
            self._fill, self._bust = False, True
        else:
            self._score = score + self._score
            self._dice_remaining = dice_remaining
            if self._dice_remaining == 0:
                self._fill, self._bust = True, False

    def __restart__(self):
        self.__init__()
        
    @property
    def score(self):
        return self._score

    def __str__(self):
        return "Dice: " + str(self._dice) + ", current score: " + str(self._score) + ", dice remaining: " + str(self._dice_remaining)

    def __repr__(self):
        return "Dice: " + str(self._dice) + ", current score: " + str(self._score) + ", dice remaining: " + str(self._dice_remaining)

class Card(object):

    def __init__(self, name, bonus, requires_fill):
        self._name = name        
        self._bonus = bonus
        self._requires_fill = requires_fill

    @property
    def bonus(self):
        return self._bonus


class Deck(object):

    card_types = ["Bonus300","Bonus400","Bonus500","FillOneThousand","DoubleTrouble","NoDice","MustBust","Vengence"]
    card_counts = [5,5,5,4,2,9,3,3]
    cards = dict(zip(card_types,card_counts))

    def __init__(self):
        self._current_card_num = 1
        self._cards = []        

        self._num_cards = len(self._cards)

        self.shuffle()

    def draw(self):
        if self._current_card_num <= self._num_cards:
            return self._cards[self._current_card_num]
        else:
            self.shuffle()
            self.draw()

    # def shuffle(self):




def main():
    player = Turn()
    bust_count, fill_count, stop_count = 0,0,0
    bonus_values = [300, 400, 500]

    n_turns = 5000
    scores = np.zeros((n_turns,1))
    for n in np.arange(n_turns):
        # print('\n///////////// New Turn /////////////')

        # Draw a card
        card = Card('Garden Variety',np.random.choice(bonus_values),False)

        while player._dice_remaining > 0 and not player._stopped:            

            # Implement some strategy
            # if player._dice_remaining <= 2:
            #     player.stop()
            #     break        

            player.roll()
            # print(player)

        if player._fill:
            player._score = player._score# + card.bonus

        if np.mod(n,np.round(0.2*n_turns))==0: print("{0}/{1}".format(n,n_turns))
        scores[n,0] = player.score

        if player._bust:
            bust_count += 1
        if player._fill:
            fill_count += 1
        if player._stopped:
            stop_count += 1

        player.__restart__()

    print("\nAfter {N} turns, the average score was: {avg_score}.".format(N=n_turns,avg_score=scores.mean()))
    print("You busted {n_busts} times, got {n_fills} fills, and stopped {n_stops} times.".format(n_busts=bust_count,n_fills=fill_count,n_stops=stop_count))
    print("You filled {0}% of the time.".format(100*fill_count/float(bust_count+fill_count)))


if __name__ == '__main__':
    start = time.time()
    main()
    print("Time elapsed: {0}".format(time.time() - start))

