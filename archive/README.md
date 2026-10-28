# Archive

Previous attempts at modeling Fill or Bust (also called "dice" or "Muss Buss"),
consolidated here with their git history. They are kept for reference only:
none of this code is maintained, and it is not runnable under the root project's
environment. Each folder keeps its original packaging files, if any.

| Folder | Original location | Original remote | Dates | Source HEAD | Python | Description |
|---|---|---|---|---|---|---|
| [`2017-fillorbust-cv/`](2017-fillorbust-cv/) | `~/Projects/Games/FillOrBustCV` | `github.com/johnwmillr/FillOrBust` (`master`) | 2017-12-24 to 2018-10-12 | `2a37015` | 2.7 / 3.6 | Game simulation plus OpenCV dice detection from photos (`dice.jpg`, `dice02.jpg`). |
| [`2018-mussbuss-scratch/chris_mussbuss/`](2018-mussbuss-scratch/chris_mussbuss/) | `~/Projects/scratch/chris_mussbuss/mussbuss` | none | 2018-07-27 | `44d2214` | 3.6 | Selenium bot that plays Chris's Heroku Muss Buss game. |
| [`2018-mussbuss-scratch/mussbuss_v2/`](2018-mussbuss-scratch/mussbuss_v2/) | `~/Projects/scratch/MussBuss_v2/mussbuss` | none (no git) | 2018-07-01 (mtime) | n/a | 3.6 | Single-file game model. |
| [`2019-mussbot/`](2019-mussbot/) | `~/Projects/Games/MussBot` | none | 2019-12-21 to 2020-02-03 | `d6616e2` | 3.7 | Client for Chris's Muss Buss server and early Q-learning experiments. Also holds `mussBuss.ipynb` from `Games/MussBussChris` (untracked there, 2019-12-21; password redacted). |
| [`2021-gamelearning/`](2021-gamelearning/) | `~/Projects/GameLearning` (dice paths only) | `github.com/johnwmillr/GameLearning` | 2021-12-25 to 2024-12-30 | `6118614` (`rewrite-dice-to-be-simpler`) | 3.8 to 3.10 | Reinforcement-learning agents for dice and Fill or Bust, alongside other games that stayed in GameLearning. |
| [`2021-simpledice/`](2021-simpledice/) | `~/Projects/Games/SimpleDice` | none (no commits) | 2021-08-13 (file mtimes) | n/a | 3.8 | Minimal dice game package and notebook. The folder name came from a later copy date; the files themselves date from 2021. |
| [`2025-dice/`](2025-dice/) | `~/Projects/Games/Dice` | none | 2025-01-02 to 2026-10-26 | `3f0a79d` | 3.13 (Poetry) | Most recent previous attempt: typed dice game, Q-learning training, tests. |

In `2019-mussbot`, `2021-gamelearning` and `2025-dice`, the last commit ("Commit
work in progress before consolidation") captures uncommitted work found when the
archive was assembled.

## Tag

`archive/gamelearning-simplify-fob` preserves GameLearning's unmerged
`simplify-fob` branch (5 commits, 2022-12-30 to 2023-01-06, dice paths only):

```sh
git log archive/gamelearning-simplify-fob
```

## Not archived

Chris's own Muss Buss web app lives at
[chrisstephenmiller/muss-buss](https://github.com/chrisstephenmiller/muss-buss).
