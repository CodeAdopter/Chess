using Core;

namespace Engine.Nnue;

/// <summary>
/// HalfKP feature indexing. A feature is the triple (our king square, a non-king piece, that piece's
/// square), all viewed from one side's perspective. The king square "buckets" the evaluation on king
/// safety, the network's single most important conditioning signal, which is what makes HalfKP much
/// stronger than a plain piece-square net.
///
/// Two accumulators are kept, one per colour (White-perspective, Black-perspective). At eval time the
/// network is fed [side-to-move accumulator, other accumulator]. Kings are NOT piece features (they
/// only act as the bucket), so a king move forces a full refresh of that colour's accumulator only.
/// </summary>
public static class FeatureSet
{
    /// <summary>
    /// The number of distinct (relation, piece-type) combinations a non-king piece can have: {own, opponent} times {P, N, B, R, Q}.
    /// </summary>
    public const int PieceKinds = 10;                 // {own,opp} x {P,N,B,R,Q}
    /// <summary>
    /// Total feature count: king square (64) times piece kind (10) times piece square (64) = 40960.
    /// </summary>
    public const int NumFeatures = 64 * PieceKinds * 64; // king x kind x square = 40960

    /// <summary>
    /// Orient a square into <paramref name="persp"/>'s view: Black mirrors vertically.
    /// </summary>
    public static int Orient(Color persp, int sq) => persp == Color.White ? sq : sq ^ 56;

    /// <summary>
    /// 0..9 for a non-king piece in this perspective, or -1 for a king (not a feature).
    /// </summary>
    public static int PieceKind(Color persp, Piece piece)
    {
        PieceType pt = Types.TypeOf(piece);
        if (pt == PieceType.King) return -1;
        bool own = Types.ColorOf(piece) == persp;
        return (own ? 0 : 5) + (int)pt;
    }

    /// <summary>
    /// Flat feature index for (perspective king, piece, square).
    /// </summary>
    public static int Index(Color persp, Square kingSq, Piece piece, Square sq)
    {
        int k = Orient(persp, (int)kingSq);
        int kind = PieceKind(persp, piece);
        int s = Orient(persp, (int)sq);
        return (k * PieceKinds + kind) * 64 + s;
    }

    /// <summary>
    /// Append the active feature indices for <paramref name="persp"/> to <paramref name="into"/>.
    /// </summary>
    public static void ActiveFeatures(Position pos, Color persp, List<int> into)
    {
        Square kingSq = Bitboard.Bsf(pos.BitboardOf(persp, PieceType.King));
        for (int sq = 0; sq < 64; sq++)
        {
            Piece p = pos.At((Square)sq);
            if (p == Piece.NoPiece) continue;
            if (Types.TypeOf(p) == PieceType.King) continue;
            into.Add(Index(persp, kingSq, p, (Square)sq));
        }
    }
}
