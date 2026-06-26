using Core;

namespace Engine.Nnue;

/// <summary>
/// int16 incrementally-updated accumulators for the quantised net, the speed-path twin of
/// <see cref="AccumulatorStack"/>. Integer add/sub is exact, so an incrementally-maintained accumulator
/// equals a full refresh bit-for-bit (no float drift). The move-to-feature-change logic and the HalfKP
/// king-bucket refresh rule are shared with the float path via <see cref="AccumulatorStack.ComputeChanges"/>.
/// It can also carry an incrementally-maintained hand-crafted eval alongside (OPT_INCEVAL) so the search
/// never has to rescan all pieces for the baseline term.
/// </summary>
public sealed class QAccumulatorStack
{
    private readonly QuantizedNetwork net;
    private readonly int H;
    private readonly short[][] white;   // [ply][H]
    private readonly short[][] black;
    private readonly int[] hce;         // [ply] incremental hand-crafted eval (white POV), parallel to accumulators
    public int Top { get; private set; }

    /// <summary>
    /// Preallocates the per-ply int16 accumulator buffers (and the optional hand-crafted-eval stack) for both perspectives.
    /// </summary>
    public QAccumulatorStack(QuantizedNetwork net, int maxPlies = 512)
    {
        this.net = net;
        H = net.H;
        white = new short[maxPlies + 1][];
        black = new short[maxPlies + 1][];
        for (int i = 0; i <= maxPlies; i++) { white[i] = new short[H]; black[i] = new short[H]; }
        hce = new int[maxPlies + 1];
    }

    /// <summary>
    /// Push an unchanged copy of the top (a null move moves no pieces).
    /// </summary>
    public void PushNull()
    {
        Array.Copy(white[Top], white[Top + 1], H);
        Array.Copy(black[Top], black[Top + 1], H);
        hce[Top + 1] = hce[Top];
        Top++;
    }

    /// <summary>
    /// Reset the stack to a full refresh of <paramref name="pos"/> at ply 0.
    /// </summary>
    public void RefreshRoot(Position pos)
    {
        Top = 0;
        RefreshOne(white[0], pos, Color.White);
        RefreshOne(black[0], pos, Color.Black);
        hce[0] = ComputeHceWhite(pos);
    }

    /// <summary>
    /// Centipawn correction at the current top, from <paramref name="stm"/>'s POV (SIMD when available).
    /// </summary>
    public int Evaluate(Color stm) =>
        stm == Color.White ? net.Forward(white[Top], black[Top]) : net.Forward(black[Top], white[Top]);

    /// <summary>
    /// Incrementally-maintained hand-crafted eval at the current top, side-to-move POV.
    /// </summary>
    public int EvaluateHce(Color stm) => stm == Color.White ? hce[Top] : -hce[Top];

    /// <summary>
    /// Scalar-path correction at the current top, kept as a reference for validating the SIMD forward.
    /// </summary>
    public int EvaluateScalar(Color stm) =>
        stm == Color.White ? net.ForwardScalar(white[Top], black[Top]) : net.ForwardScalar(black[Top], white[Top]);

    /// <summary>
    /// Pops back to the parent accumulator (paired with <see cref="Push"/> on unmake).
    /// </summary>
    public void Pop() => Top--;

    /// <summary>
    /// Pushes the child accumulator after a move, applying just the feature deltas it changed (or a full refresh of the side whose king moved). See <see cref="AccumulatorStack.Push"/> for the float-path twin.
    /// </summary>
    public void Push(Position posAfter, Move m, Piece movingPiece, Piece capturedPiece)
    {
        int parent = Top, child = Top + 1;
        Color moved = Types.ColorOf(movingPiece);
        bool kingMoved = Types.TypeOf(movingPiece) == PieceType.King;

        Span<AccumulatorStack.Change> changes = stackalloc AccumulatorStack.Change[5];
        int n = AccumulatorStack.ComputeChanges(m, movingPiece, capturedPiece, changes);

        hce[child] = hce[parent] + HceDelta(changes, n);

        UpdateSide(Color.White, parent, child, posAfter, kingMoved && moved == Color.White, changes, n);
        UpdateSide(Color.Black, parent, child, posAfter, kingMoved && moved == Color.Black, changes, n);
        Top = child;
    }

    private void UpdateSide(Color persp, int parent, int child, Position posAfter, bool refresh, Span<AccumulatorStack.Change> changes, int n)
    {
        short[] dst = persp == Color.White ? white[child] : black[child];
        if (refresh)
        {
            RefreshOne(dst, posAfter, persp);   // this side's bucket changed → recompute
            return;
        }

        short[] src = persp == Color.White ? white[parent] : black[parent];
        Array.Copy(src, dst, H);

        // fused updates https://www.chessprogramming.org/NNUE
        Square kingSq = Bitboard.Bsf(posAfter.BitboardOf(persp, PieceType.King));
        Span<int> adds = stackalloc int[2];
        Span<int> subs = stackalloc int[3];
        int na = 0, ns = 0;
        for (int c = 0; c < n; c++)
        {
            if (FeatureSet.PieceKind(persp, changes[c].Piece) < 0) continue;   // kings are not features
            int f = FeatureSet.Index(persp, kingSq, changes[c].Piece, (Square)changes[c].Sq);
            if (changes[c].Sign > 0) adds[na++] = f; else subs[ns++] = f;
        }

        if (na == 1 && ns == 1) net.AddSub(dst, adds[0], subs[0]);
        else if (na == 1 && ns == 2) net.AddSubSub(dst, adds[0], subs[0], subs[1]);
        else
        {
            for (int a = 0; a < na; a++) net.AddFeature(dst, adds[a]);
            for (int s = 0; s < ns; s++) net.SubFeature(dst, subs[s]);
        }
    }

    private void RefreshOne(short[] acc, Position pos, Color persp)
    {
        net.CopyBias(acc);
        Square kingSq = Bitboard.Bsf(pos.BitboardOf(persp, PieceType.King));
        for (int sq = 0; sq < 64; sq++)
        {
            Piece p = pos.At((Square)sq);
            if (p == Piece.NoPiece || Types.TypeOf(p) == PieceType.King) continue;
            net.AddFeature(acc, FeatureSet.Index(persp, kingSq, p, (Square)sq));
        }
    }

    private static int ComputeHceWhite(Position pos)
    {
        int score = 0;
        for (int sq = 0; sq < 64; sq++)
        {
            Piece p = pos.At((Square)sq);
            if (p == Piece.NoPiece) continue;
            int term = HceTerm(p, sq);
            score += Types.ColorOf(p) == Color.White ? term : -term;
        }
        return score;
    }

    private static int HceDelta(Span<AccumulatorStack.Change> changes, int count)
    {
        int delta = 0;
        for (int i = 0; i < count; i++)
        {
            Piece p = changes[i].Piece;
            int term = HceTerm(p, changes[i].Sq);
            int signed = Types.ColorOf(p) == Color.White ? term : -term;
            delta += changes[i].Sign > 0 ? signed : -signed;
        }
        return delta;
    }

    private static int HceTerm(Piece p, int sq)
    {
        PieceType pt = Types.TypeOf(p);
        int pstSq = Types.ColorOf(p) == Color.White ? sq : sq ^ 56;
        return Eval.PieceValue[(int)pt] + Eval.Pst[(int)pt][pstSq];
    }
}
