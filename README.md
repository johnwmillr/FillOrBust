# Fill or Bust

Fill or Bust (also known as Muss Buss) is a push-your-luck game played with six
dice and a deck of cards. Each turn you draw a card, roll, set aside scoring
dice, and decide whether to keep rolling for more points or stop before you
bust.

This is a fresh attempt at modeling the game. Previous attempts live in
[`archive/`](archive/).

## Usage

```sh
uv sync
uv run pytest
```

## History

| Year | Attempt |
|---|---|
| 2017 | [`archive/2017-fillorbust-cv/`](archive/2017-fillorbust-cv/): game simulation and dice detection from photos |
| 2017-2019 | [chrisstephenmiller/muss-buss](https://github.com/chrisstephenmiller/muss-buss): Chris's Muss Buss web app (separate repo) |
| 2018 | [`archive/2018-mussbuss-scratch/`](archive/2018-mussbuss-scratch/): scratch game model and a Selenium bot for Chris's web game |
| 2019 | [`archive/2019-mussbot/`](archive/2019-mussbot/): server client and early Q-learning |
| 2021 | [`archive/2021-gamelearning/`](archive/2021-gamelearning/): reinforcement-learning agents, from the GameLearning repo |
| 2023 | [`archive/2021-simpledice/`](archive/2021-simpledice/): minimal dice game package |
| 2025 | [`archive/2025-dice/`](archive/2025-dice/): typed dice game with Q-learning training |

See [`archive/README.md`](archive/README.md) for details on each attempt.
