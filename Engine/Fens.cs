namespace Engine;

/// <summary>
/// Standard reference positions for benchmarks and tools, as FEN strings. The trailing halfmove/fullmove counters are omitted where unneeded, since perft and eval ignore them.
/// </summary>
public static class Fens
{
    public const string Startpos   = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";
    public const string Kiwipete   = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";
    public const string Endgame    = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -";
    public const string Tactical   = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -";
    public const string Promotions = "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8";
    public const string Midgame    = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10";
}
