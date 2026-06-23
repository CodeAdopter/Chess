using Core;

namespace Engine.Nnue;

/// <summary>
/// Incrementally-updated HalfKP accumulators, the speed core of NNUE-in-search. Instead of recomputing
/// both perspective accumulators from scratch at every leaf (full refresh, ~30 feature columns each),
/// a move only adds/removes the handful of feature columns it changes. A per-ply stack lets the search
/// push on make and pop on unmake in O(1).
///
/// HalfKP detail: the king is the bucket, not a feature. So a king move re-buckets that colour's entire
/// accumulator → that side is fully refreshed; the other side just sees the moved pieces' deltas (for
/// castling, the rook). Non-king moves apply piece deltas to both sides with their (unchanged) buckets.
/// </summary>
public sealed class AccumulatorStack
{
    private readonly NnueNetwork net;
    private readonly int H;
    private readonly float[][] white;   // [ply][H]
    private readonly float[][] black;
    public int Top { get; private set; }

    /// <summary>Push an unchanged copy of the top (a null move moves no pieces).</summary>
    public void PushNull()
    {
        Array.Copy(white[Top], white[Top + 1], H);
        Array.Copy(black[Top], black[Top + 1], H);
        Top++;
    }

    /// <summary>
    /// Preallocates the per-ply accumulator buffers for both perspectives (no allocation happens during search).
    /// </summary>
    public AccumulatorStack(NnueNetwork net, int maxPlies = 512)
    {
        this.net = net;
        H = net.H;
        white = new float[maxPlies + 1][];
        black = new float[maxPlies + 1][];
        for (int i = 0; i <= maxPlies; i++) { white[i] = new float[H]; black[i] = new float[H]; }
    }

    /// <summary>
    /// Reset the stack to a full refresh of <paramref name="pos"/> at ply 0.
    /// </summary>
    public void RefreshRoot(Position pos)
    {
        Top = 0;
        RefreshOne(white[0], pos, Color.White);
        RefreshOne(black[0], pos, Color.Black);
    }

    /// <summary>
    /// Centipawn correction at the current top, from <paramref name="stm"/>'s POV.
    /// </summary>
    public int Evaluate(Color stm) =>
        stm == Color.White ? net.Forward(white[Top], black[Top]) : net.Forward(black[Top], white[Top]);

    /// <summary>
    /// Pops back to the parent accumulator (paired with <see cref="Push"/> on unmake).
    /// </summary>
    public void Pop() => Top--;

    /// <summary>
    /// One feature change a move makes: a <see cref="Piece"/> appearing (<c>Sign</c> &gt; 0) or leaving (<c>Sign</c> &lt; 0) square <c>Sq</c>.
    /// </summary>
    public struct Change { public Piece Piece; public int Sq; public int Sign; }

    /// <summary>
    /// Push the child accumulator after a move. <paramref name="posAfter"/> is the position with the move
    /// already played; <paramref name="movingPiece"/>/<paramref name="capturedPiece"/> are captured by the
    /// caller before the move (capturedPiece = NoPiece if none / en passant).
    /// </summary>
    public void Push(Position posAfter, Move m, Piece movingPiece, Piece capturedPiece)
    {
        int parent = Top, child = Top + 1;
        Color moved = Types.ColorOf(movingPiece);
        bool kingMoved = Types.TypeOf(movingPiece) == PieceType.King;  // covers OO/OOO (king is the mover)

        Span<Change> changes = stackalloc Change[5];
        int n = ComputeChanges(m, movingPiece, capturedPiece, changes);

        UpdateSide(Color.White, parent, child, posAfter, kingMoved && moved == Color.White, changes, n);
        UpdateSide(Color.Black, parent, child, posAfter, kingMoved && moved == Color.Black, changes, n);
        Top = child;
    }

    private void UpdateSide(Color persp, int parent, int child, Position posAfter, bool refresh, Span<Change> changes, int n)
    {
        float[] dst = persp == Color.White ? white[child] : black[child];
        if (refresh)
        {
            RefreshOne(dst, posAfter, persp);   // this side's bucket changed → recompute
            return;
        }

        float[] src = persp == Color.White ? white[parent] : black[parent];
        Array.Copy(src, dst, H);

        Square kingSq = Bitboard.Bsf(posAfter.BitboardOf(persp, PieceType.King));  // unchanged for this side
        for (int c = 0; c < n; c++)
        {
            if (FeatureSet.PieceKind(persp, changes[c].Piece) < 0) continue;   // kings are not features
            int f = FeatureSet.Index(persp, kingSq, changes[c].Piece, (Square)changes[c].Sq);
            if (changes[c].Sign > 0) net.AddFeature(dst, f); else net.SubFeature(dst, f);
        }
    }

    private void RefreshOne(float[] acc, Position pos, Color persp)
    {
        Array.Copy(net.L0Bias, acc, H);
        Square kingSq = Bitboard.Bsf(pos.BitboardOf(persp, PieceType.King));
        for (int sq = 0; sq < 64; sq++)
        {
            Piece p = pos.At((Square)sq);
            if (p == Piece.NoPiece || Types.TypeOf(p) == PieceType.King) continue;
            net.AddFeature(acc, FeatureSet.Index(persp, kingSq, p, (Square)sq));
        }
    }

    /// <summary>
    /// Enumerate the (piece, square, add/remove) feature changes a move makes. Returns the count.
    /// </summary>
    public static int ComputeChanges(Move m, Piece movingPiece, Piece capturedPiece, Span<Change> buf)
    {
        const int Add = 1, Rem = -1;
        Color color = Types.ColorOf(movingPiece);
        var flags = m.Flags;
        int from = (int)m.From, to = (int)m.To;
        int n = 0;

        if (flags == MoveFlags.OO || flags == MoveFlags.OOO)
        {
            // castling: move king (skipped as a feature) and rook
            bool kingside = flags == MoveFlags.OO;
            (int rFrom, int rTo) = color == Color.White
                ? (kingside ? (7, 5) : (0, 3))
                : (kingside ? (63, 61) : (56, 59));
            Piece rook = Types.MakePiece(color, PieceType.Rook);
            buf[n++] = new Change { Piece = movingPiece, Sq = from, Sign = Rem };  // king (skipped later)
            buf[n++] = new Change { Piece = movingPiece, Sq = to, Sign = Add };
            buf[n++] = new Change { Piece = rook, Sq = rFrom, Sign = Rem };
            buf[n++] = new Change { Piece = rook, Sq = rTo, Sign = Add };
            return n;
        }

        if ((flags & MoveFlags.Promotions) != 0)
        {
            Piece promoted = Types.MakePiece(color, (PieceType)(1 + ((int)flags & 0b11)));  // Knight..Queen
            buf[n++] = new Change { Piece = movingPiece, Sq = from, Sign = Rem };   // pawn leaves
            buf[n++] = new Change { Piece = promoted, Sq = to, Sign = Add };        // promoted piece arrives
        }
        else
        {
            buf[n++] = new Change { Piece = movingPiece, Sq = from, Sign = Rem };
            buf[n++] = new Change { Piece = movingPiece, Sq = to, Sign = Add };
        }

        if (flags == MoveFlags.EnPassant)
        {
            int capSq = to + (color == Color.White ? -8 : 8);   // captured pawn sits behind the to-square
            buf[n++] = new Change { Piece = Types.MakePiece(color.Flip(), PieceType.Pawn), Sq = capSq, Sign = Rem };
        }
        else if ((flags & MoveFlags.Capture) == MoveFlags.Capture && capturedPiece != Piece.NoPiece)
        {
            buf[n++] = new Change { Piece = capturedPiece, Sq = to, Sign = Rem };
        }

        return n;
    }
}
