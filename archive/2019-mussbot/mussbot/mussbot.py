from requests import Session
from mussbot.game import Card
from mussbot.game import Game


class Client(object):

    def __init__(self, username, password, host=None, wait=0.3, game_number=None, verbose=True):
        self._wait = wait
        self.verbose = verbose
        self.session = Session()
        if not host:
            host = 'http://muss-buss-env.2qxazt2xnm.us-east-1.elasticbeanstalk.com'
        self.host = host.rstrip('/')
        r = self.__authorize__(username, password)
        self._sid = r.cookies['connect.sid']
        self.headers = {'Connection': 'keep-alive',
                        'Accept': 'application/json, text/plain, */*',
                        'User-Agent': 'gurk',
                        'Accept-Encoding': 'gzip, deflate',
                        'Accept-Language': 'en-US,en;q=0.9,fr;q=0.8',
                        'Cookie': f'connect.sid={self._sid}'}
        data = r.json()
        self.id = data['id']
        self.email = data['email']
        self.name = data['name']
        self.games = data['games']
        if self.games:
            self.game_number = game_number if game_number else self.games[-1]['id']
        else:
            self.game_number = None
        self.update_game()

    def __authorize__(self, username, password):
        headers = {'content type': 'application/json'}
        url = f'{self.host}/auth/login'
        auth = {'email': username, 'password': password}
        return self.session.post(url, headers=headers, data=auth)

    @property
    def game_root(self):
        return f"{self.host}/api/games/{self.game_number}"

    def send_request(self, query, method='get', game_request=True, data=None):
        url = self.game_root if game_request else self.host
        url += f'/{query.lstrip("/")}'

        # Make the request
        r = self.session.request(method=method,
                                 url=url,
                                 headers=self.headers,
                                 data=data)
        self.update_game(r)
        return r

    def pprint(self):
        self.game.pprint()

    def new_game(self, player_names, win_score=10000):
        q = '/api/games'
        config = {'players': [1, 2], 'winScore': win_score}
        r = self.send_request(q,
                              method='POST',
                              game_request=False,
                              data=config)
        msg = f'Error creating game: {r.status_code}'
        assert r.status_code == 201, msg
        game_number = r.json()['id']
        if self.verbose:
            print(f'Success! Created game #{game_number}.')

        # Update the Client's game to the new game
        self.switch_games(game_number, r=r)

    def switch_games(self, game_number, r=None):
        self.game_number = game_number
        self.update_game(r)

    def update_game(self, r=None):
        if not r:
            r = self.session.get(self.game_root)
        self.game = Game(**r.json(), game_number=self.game_number)
        if self.verbose:
            self.pprint()

    def draw(self):
        q = "draw"
        return self.send_request(q)

    def roll(self):
        q = "roll"
        return self.send_request(q)

    def pass_(self):
        q = "pass"
        return self.send_request(q)

    def stop(self):
        q = "stop"
        return self.send_request(q)

    def hold(self, dice_id=6):
        q = f"hold?holdId={dice_id}"
        return self.send_request(q)

    def __repr__(self):
        self.update_game()
        return f'({self.name}) {self.game}'

    def __del__(self):
        self.session.close()


class Player(Client):

    def __init__(self, *args, **kwargs):
        super(Player, self).__init__(*args, **kwargs)

    def take_turn(self):
        player = self.game.current_player
        actions = self.game.invalid_actions
        if self.verbose:
            print(f"Current player: {player['name']}")
            print(f"Starting score: {player['score']}")

        # Pick a valid action for the turn
        if not actions['invalidStop']:
            if self.verbose:
                print('Stopping...')
            self.stop()
            return
        if not actions['invalidDraw']:
            if self.verbose:
                print('Drawing a card...')
            self.draw()
            return
        if not actions['invalidRoll']:
            if self.verbose:
                print('Rolling...')
            self.roll()
            return
        if not actions['invalidHolds']:
            if self.verbose:
                print("Holding...")
            self.hold()
            return
