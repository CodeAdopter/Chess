using Core;

namespace Engine;

/// <summary>
/// Stage-0 hand-crafted evaluation: material + piece-square tables. It is intentionally simple, since
/// it exists only to validate that the search actually plays sensible chess. It is the baseline that
/// the NNUE correction is trained on top of, and the yardstick the novel attack-feature net (Stage 3)
/// gets A/B'd against. Returns a score in centipawns from the side-to-move's point of view.
/// </summary>
public static class Eval
{
    // Midgame material values, indexed by PieceType (Pawn..King). King has no material value.
    public static readonly int[] PieceValue = [100, 320, 330, 500, 900, 0];

    public const int MateValue = 30000;
    // Scores at or beyond this magnitude are "mate in N" and get ply-adjusted in the TT.
    public const int MateBound = MateValue - 1000;

    // Piece-square tables in white's frame, index 0=a1 .. 63=h8 (matches Core's Square enum).
    // Values are small nudges (centipawns) layered on top of material. Black mirrors via sq ^ 56.
    private static readonly int[] PawnPst =
    [
          0,   0,   0,   0,   0,   0,   0,   0,
          5,  10,  10, -20, -20,  10,  10,   5,
          5,  -5, -10,   0,   0, -10,  -5,   5,
          0,   0,   0,  20,  20,   0,   0,   0,
          5,   5,  10,  25,  25,  10,   5,   5,
         10,  10,  20,  30,  30,  20,  10,  10,
         50,  50,  50,  50,  50,  50,  50,  50,
          0,   0,   0,   0,   0,   0,   0,   0,
    ];

    private static readonly int[] KnightPst =
    [
        -50, -40, -30, -30, -30, -30, -40, -50,
        -40, -20,   0,   5,   5,   0, -20, -40,
        -30,   5,  10,  15,  15,  10,   5, -30,
        -30,   0,  15,  20,  20,  15,   0, -30,
        -30,   5,  15,  20,  20,  15,   5, -30,
        -30,   0,  10,  15,  15,  10,   0, -30,
        -40, -20,   0,   0,   0,   0, -20, -40,
        -50, -40, -30, -30, -30, -30, -40, -50,
    ];

    private static readonly int[] BishopPst =
    [
        -20, -10, -10, -10, -10, -10, -10, -20,
        -10,   5,   0,   0,   0,   0,   5, -10,
        -10,  10,  10,  10,  10,  10,  10, -10,
        -10,   0,  10,  10,  10,  10,   0, -10,
        -10,   5,   5,  10,  10,   5,   5, -10,
        -10,   0,   5,  10,  10,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10, -10, -10, -10, -10, -20,
    ];

    private static readonly int[] RookPst =
    [
          0,   0,   0,   5,   5,   0,   0,   0,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
         -5,   0,   0,   0,   0,   0,   0,  -5,
          5,  10,  10,  10,  10,  10,  10,   5,
          0,   0,   0,   0,   0,   0,   0,   0,
    ];

    private static readonly int[] QueenPst =
    [
        -20, -10, -10,  -5,  -5, -10, -10, -20,
        -10,   0,   5,   0,   0,   0,   0, -10,
        -10,   5,   5,   5,   5,   5,   0, -10,
          0,   0,   5,   5,   5,   5,   0,  -5,
         -5,   0,   5,   5,   5,   5,   0,  -5,
        -10,   0,   5,   5,   5,   5,   0, -10,
        -10,   0,   0,   0,   0,   0,   0, -10,
        -20, -10, -10,  -5,  -5, -10, -10, -20,
    ];

    // Midgame king PST, which encourages castling and staying tucked away. (Endgame king activity is a
    // refinement for later; Stage 0 keeps a single table.)
    private static readonly int[] KingPst =
    [
         20,  30,  10,   0,   0,  10,  30,  20,
         20,  20,   0,   0,   0,   0,  20,  20,
        -10, -20, -20, -20, -20, -20, -20, -10,
        -20, -30, -30, -40, -40, -30, -30, -20,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
        -30, -40, -40, -50, -50, -40, -40, -30,
    ];

    // Public so the incremental hand-crafted eval (QAccumulatorStack, OPT_INCEVAL) can index the same
    // table; values are unchanged. Indexed by PieceType (Pawn..King), white frame (black mirrors via sq^56).
    public static readonly int[][] Pst = [PawnPst, KnightPst, BishopPst, RookPst, QueenPst, KingPst];

    /// <summary>
    /// Score the position in centipawns from the side-to-move POV (negamax convention).
    /// </summary>
    public static int Evaluate(Position pos)
    {
        int white = ScoreSide(pos, Color.White);
        int black = ScoreSide(pos, Color.Black);
        int whitePov = white - black;
        return pos.Turn == Color.White ? whitePov : -whitePov;
    }

    private static int ScoreSide(Position pos, Color c)
    {
        int score = 0;
        for (int pt = 0; pt < 6; pt++)
        {
            ulong bb = pos.BitboardOf(c, (PieceType)pt);
            int material = PieceValue[pt];
            int[] table = Pst[pt];
            while (bb != 0)
            {
                int sq = (int)Bitboard.PopLsb(ref bb);
                // White reads the table directly; black mirrors vertically (a1<->a8).
                int idx = c == Color.White ? sq : sq ^ 56;
                score += material + table[idx];
            }
        }
        return score;
    }
}
