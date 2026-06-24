# Chess Engine

Chess engine with self-trained NNUE in C#.

## PERFT

CPU:  Intel Core i9-14900KF. (1 core)

### Move generation (the legal generator the search uses)

<!-- perftdoc:search:begin -->
| Position | Depth | Nodes | NPS |
|:--|--:|--:|--:|
| Startpos | 6 | 119,060,324 | 543.65 Mnps |
| Kiwipete | 5 | 193,690,690 | 691.75 Mnps |
| Endgame | 6 | 11,030,083 | 424.23 Mnps |
| Tactical | 5 | 15,833,292 | 510.75 Mnps |
| Promotions | 5 | 89,941,194 | 628.96 Mnps |
| Midgame | 5 | 164,075,551 | 672.44 Mnps |
| **Total** | | **593,631,134** | **629.51 Mnps** |
<!-- perftdoc:search:end -->

### Bulk-count perft (popcounts the last ply, perft-only and faster)

<!-- perftdoc:bulk:begin -->
| Position | Depth | Nodes | NPS |
|:--|--:|--:|--:|
| Startpos | 6 | 119,060,324 | 680.34 Mnps |
| Kiwipete | 5 | 193,690,690 | 1345.07 Mnps |
| Endgame | 6 | 11,030,083 | 787.86 Mnps |
| Tactical | 5 | 15,833,292 | 879.63 Mnps |
| Promotions | 5 | 89,941,194 | 1124.26 Mnps |
| Midgame | 5 | 164,075,551 | 1072.39 Mnps |
| **Total** | | **593,631,134** | **1016.49 Mnps** |
<!-- perftdoc:bulk:end -->

## Engine

- **Bitboards** - board state as 64-bit masks
- **Magic + PEXT sliders** - fast rook and bishop attacks
- **Legal movegen** - pins and checks resolved inline
- **Zobrist hashing** - incremental position key
- **Make / unmake** - incremental, hash-free perft path

## Search

- **Negamax alpha-beta** - PVS with iterative deepening
- **Quiescence** - captures and checks resolved at leaves
- **Transposition table** - depth-preferred cache
- **Null-move pruning** - skip when clearly ahead
- **Reverse futility** - static fail-high pruning
- **Delta + SEE** - skip losing captures in quiescence
- **MVV-LVA / history** - move ordering

## NNUE

- **HalfKP** - king-conditioned features (40960)
- **Incremental** accumulator - per-move feature updates
- **CReLU** - clamped activation
- **Residual net** - correction on top of hand-crafted eval
- **Feature factorization** - king-independent virtual weights
- **Adam (Hogwild)** - lock-free parallel training
- **Int16 quantized + AVX2** - fast inference

## Training

- **Self-play generation** - labels from search score and result
- **Replay buffer** - sliding window 1.5m pos default
- **Net vs net gating** - released net gating

## License

MIT, see [LICENSE.txt](LICENSE.txt).
