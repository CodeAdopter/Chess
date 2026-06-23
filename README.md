# Chess Engine

Chess engine with self-trained NNUE in C#.

## PERFT

Throughput on Kiwipete depth 6 (8,031,647,685 nodes), Intel Core i9-14900KF.

### Move generation (legal generator for search)

| Threads | Nodes Per Second | Speedup |
|---:|---:|---:|
| 1  | 606 Million NPS   | 1.0×  |
| 2  | 1.199 Billion NPS | 2.0×  |
| 4  | 2.360 Billion NPS | 3.9×  |
| 8  | 4.415 Billion NPS | 7.3×  |
| 16 | 6.498 Billion NPS | 10.7× |

### Bulk-count perft 

| Threads | Nodes Per Second | Speedup |
|---:|---:|---:|
| 1  | 1.110 Billion NPS  | 1.0×  |
| 2  | 2.210 Billion NPS  | 2.0×  |
| 4  | 4.290 Billion NPS  | 3.9×  |
| 8  | 7.820 Billion NPS  | 7.0×  |
| 16 | 12.688 Billion NPS | 11.4× |

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
