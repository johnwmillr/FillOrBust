class Card(object):

    def __init__(self, **kwargs):
        self.bust = kwargs.get('bust')
        self.fill = kwargs.get('fill')
        self.type = kwargs.get('type')
        self.rolls = kwargs.get('rolls')
        self.score = kwargs.get('score')

    def __repr__(self):
        return f'<{self.type}>'


class Game(object):

    def __init__(self, **kwargs):
        self.game_number = kwargs.get('game_number')
        self.players = kwargs.get('players')
        self.deck_size= kwargs.get('deckSize')
        self.current_player = kwargs.get('currentPlayer')
        self.turn = kwargs.get('turn')
        self.score = kwargs.get('score')
        self.card = kwargs.get('card')
        self.dice = kwargs.get('dice')
        self.invalid_actions = kwargs.get('invalidActions')

    def pprint(self):
        '''Pretty print the current game information.'''
        players = []
        for player in self.players:
            if player['name'] == self.current_player['name']:
                players.append(f'***{player["name"]} ({player["score"]})***')
            else:
                players.append(f'{player["name"]} ({player["score"]})')
        players = ", ".join(players)
        card = f'Card: {self.card["type"]}' if self.card else 'Card:'
        dice = f'Dice: [{", ".join(str(dice["value"]) for dice in self.dice)}]' if self.dice else 'Dice:'
        holds = []
        if self.dice:
            for die in self.dice:
                if not die['live']:
                    holds.append('X')
                    continue
                holds.append('H' if die['held'] else ' ')
        holds = f'       ' + "  ".join(holds)

        m = f'Game #{self.game_number}\n{players}\n  {card}\n  {dice}\n  {holds}'
        print(m)
        return

    def __repr__(self):
        return f'<GAME_{self.game_number}>: {", ".join([player["name"] for player in self.players])}'
