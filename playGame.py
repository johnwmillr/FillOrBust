import random


class Card(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class Deck(object):

    CARDS = {}
    CARDS['Bonus 300'] = 12
    CARDS['Bonus 400'] = 10
    CARDS['Bonus 500'] = 8
    CARDS['Fill 1000'] = 6
    CARDS['Double Trouble'] = 2
    CARDS['No Dice'] = 8
    CARDS['Must Bust'] = 4
    CARDS['Vengeance'] = 4

    def __init__(self):
        self.shuffle()

    def shuffle(self):
        print("Shuffle!")
        card_list = []
        for name, number in self.CARDS.items():
            card_list.extend(number * [Card(name)])
        random.shuffle(card_list)
        self.cards = card_list

    @property
    def num_remaining(self):
        return len(self.cards)

    def draw(self):
        if len(self):
            card = self.cards.pop()
            print(card)
            return card
        else:
            self.shuffle()
            self.draw()

    def __repr__(self):
        return f"{self.num_remaining} cards left in the deck."

    def __len__(self):
        return self.num_remaining
