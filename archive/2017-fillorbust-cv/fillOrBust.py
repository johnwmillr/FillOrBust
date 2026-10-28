import numpy as np
import random
import time
from collections import Counter


class Player(object):

    def __init__(self, strategy='normal'):
        self.score = 0
        self.strategy = strategy


class Turn(object):

    def __init__(self):
        self._card = []
        self._score = 0
        self.__set_status__("start")

        # Pointer values
        self.__TRIPLE_VALUES = [1000, 200, 300, 400, 500, 600]
        self.__SINGLE_VALUES = [100, 0, 0, 0, 50, 0]

    def _set_card(self, card):
        assert type(card) == Card, "Card input must be a Card object."
        self._card = card

    def take_turn(self):
        # TODO: I could probably implement some strategy here
        # TODO: I'd like to be able to draw a card from here somehow
        while self._can_continue:
            self.roll()

            # if self.score > 500 or self._dice_remaining <= 2 or self._fill:
                # self.__set_status__("stop")

        self.finishTurn()

    def roll(self):
        assert self._card != [], "Must first draw a card."
        assert self._can_continue, "Can't roll again, turn over."

        # It might make sense to implement the card consequences here
        if self._card == "NoDice":
            self.__set_status__("bust")
        # elif self._card.name in ["Bonus300", "Bonus400", "Bonus500", "Fill1000", "MustBust"]:
        else:
            self._dice = np.sort(np.random.randint(1,7,self._dice_remaining))
            self.calcScore()
        return self

    def stop(self):
        assert self._can_continue, "You can't stop, your turn is over."
        self.__set_status__("stop")

    def __set_status__(self, status):
        if status == "start":
            self._dice_remaining = 6
            self._dice = []
            self._fill = False
            self._bust = False
            self._stopped = False
        elif status == "fill":
            self._fill = True
            self._bust = False
            self._stopped = False
        elif status == "bust":
            self._fill = False
            self._bust = True
            self._stopped = False
        elif status == "stop":
            self._fill = False
            self._bust = False
            self._stopped = True

    @property
    def _can_continue(self):
        if any([self._bust, self._stopped]):
            return False
        else:
            return True

    def calcScore(self):
        # Check for a straigh
        if self._dice_remaining == 6:
            if (self._dice==[1,2,3,4,5,6]).all():
                self._score += 1500
                self.__set_status__("fill")
                return

        # Count of each number's occurence
        counts = np.array([Counter(self._dice)[k] for k in range(1,7)])

        # Check for three of a kind
        score = 0
        while (counts>=3).any():
            score += np.dot(counts>=3, self.__TRIPLE_VALUES)
            self._dice_remaining -= 3*sum(counts>=3)
            counts[counts>=3] = counts[counts>=3]-3

        # TODO: Should strategy go here?
        # Implement some strategy
        # if counts[0] > 0:
            # counts[4] = 0 # Throw back fives
        # if counts[4] > 0:
            # counts[1] = 0 # Throw back ones

        # Check for singles (1's and 5's)
        score += np.dot(counts, self.__SINGLE_VALUES)
        self._dice_remaining -= np.dot(counts,[1,0,0,0,1,0])

        # Determine if this roll is a fill or bust
        if score == 0:
            self.__set_status__("bust")
        else:
            self._score = score + self._score
            if self._dice_remaining == 0:
                if self._card.name != "MustBust":
                    self.__set_status__("fill")
                else:
                    self.__set_status__("start")

    def finishTurn(self):
        # Add card bonus to score if player filled
        if self._fill:
            self._score += self._card.bonus
        elif self._bust and self._card.name != "MustBust":
            self._score = 0

    def __restart__(self):
        self.__init__()

    @property
    def score(self):
        return self._score

    def __str__(self):
        return "Card: " + str(self._card) + ", dice: " + str(self._dice) + ", current score: " + str(self._score) + ", dice remaining: " + str(self._dice_remaining)

    def __repr__(self):
        return "Card: " + str(self._card) + ", dice: " + str(self._dice) + ", current score: " + str(self._score) + ", dice remaining: " + str(self._dice_remaining)


class Card(object):

    def __init__(self, name, bonus, requires_fill):
        self.name = name
        self._bonus = bonus
        self._requires_fill = requires_fill

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __eq__(self, other_card):
        result = False
        if type(self) == type(other_card):
            if self.name == other_card.name:
                result = True
        elif type(other_card) == type(""):
            if self.name == other_card:
                result = True
        return result

    @property
    def bonus(self):
        return self._bonus


class Deck(object):

    NAMES   = ["Bonus300","Bonus400","Bonus500","Fill1000",
               "DoubleTrouble","NoDice","MustBust","Vengence"]
    FREQS = [5,5,5,4,2,9,3,3]
    BONUSES = [300,400,500,1000,0,0,0,0]
    REQUIRES_FILL = [False,False,False,True,True,False,False,True]

    # Populate a list with all of the cards
    ALL_CARDS = []
    for name, bonus, fill in zip(NAMES, BONUSES, REQUIRES_FILL):
        for i in range(FREQS[NAMES.index(name)]):
            ALL_CARDS.append(Card(name, bonus, fill))
    TOTAL_NUM_CARDS_IN_DECK = len(ALL_CARDS)

    class Node(object):

        def __init__(self, value=None, next=None):
            self.value = value
            self.next = next

        def __repr__(self):
            return str(self.value)

    def __init__(self):
        self.top = None
        self.length = 0
        self.shuffle()

    def _push(self, value):
        assert isinstance(value, Card), "New value must be of type Card."
        new_top = self.Node(value, next=self.top)
        self.top = new_top
        self.length += 1

    def _pop(self):
        assert self.top is not None and self.length > 0, "Can't pop from empty Deck."
        old_top = self.top
        self.top = old_top.next
        self.length -= 1
        return old_top.value

    def draw(self):
        if self.top is None or self.length == 0:
            self.shuffle()
        return self._pop()

    def peek(self):
        if self.top is None:
            return None
        else:
            return self.top.value

    def shuffle(self):
        # print('Shuffle!')
        random.shuffle(self.ALL_CARDS)
        for card in self.ALL_CARDS:
             self._push(card)

    def __repr__(self):
        if self.length == 1:
            return "Deck with {} card  remaining.".format(self.length)
        else:
            return "Deck with {} cards remaining.".format(self.length)

def main():
    deck = Deck()
    player = Turn()
    bust_count, fill_count, stop_count = 0,0,0

    n_turns = 5000
    scores = np.zeros((n_turns,1))
    for n in np.arange(n_turns):
        if np.mod(n,np.round(0.2*n_turns))==0: print("Turn: {:>4}/{:>4}".format(n,n_turns))

        while player._can_continue:
            # Draw a card from the deck
            player._set_card(deck.draw())
            player.take_turn()

        scores[n,0] = player.score

        if player._fill:
            fill_count += 1
        elif player._bust:
            bust_count += 1
        elif player._stopped:
            stop_count +=1

        player.__restart__()

    print("\nAfter {N} turns, the average score was: {avg_score}.".format(N=n_turns,avg_score=scores.mean()))
    print("You busted {n_busts} times, got {n_fills} fills, and stopped {n_stops} times.".format(n_busts=bust_count,n_fills=fill_count,n_stops=stop_count))
    print("You busted {0}% of the time.".format(100*bust_count/float(n_turns)))


if __name__ == '__main__':
    start = time.time()
    main()
    print("Time elapsed: {0}".format(time.time() - start))

