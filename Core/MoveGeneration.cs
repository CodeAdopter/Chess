using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Core;

/// <summary>
/// Output target for the legal move generator. The generator computes every legal destination as a bitboard
/// and hands it to a sink; <see cref="EmitSink"/> materialises individual <see cref="Move"/>s into a span (the
/// normal path the search uses), while <see cref="CountSink"/> only popcounts the destinations. Templating the
/// generator on the sink (a struct, so the JIT specialises and inlines it away) lets perft count leaf nodes in
/// bulk, without writing a single Move, while the legality logic lives in exactly one place.
/// </summary>
internal interface IMoveSink
{
    void Quiets(Square from, ulong to);
    void Captures(Square from, ulong to);
    void DoublePushes(Square from, ulong to);
    void PromoQuiets(Square from, ulong to);     // four moves per destination
    void PromoCaptures(Square from, ulong to);   // four moves per destination
    void PawnMoves(ulong to, int dir, MoveFlags flag);   // from = to - dir
    void PawnPromos(ulong to, int dir, bool capture);    // four moves per destination, from = to - dir
    void FromMask(ulong from, Square to, MoveFlags flag); // each set "from" square moves to a fixed target
    void One(Square from, Square to, MoveFlags flag);
}

/// <summary>
/// Writes each generated move into a caller-supplied span. <c>Count</c> is the number written.
/// </summary>
internal ref struct EmitSink : IMoveSink
{
    private ref Move _list;
    private int _idx;
    public EmitSink(Span<Move> list) { _list = ref MemoryMarshal.GetReference(list); _idx = 0; }
    public readonly int Count => _idx;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Quiets(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int prefix = (int)from << 6;
        while (to != 0) Unsafe.Add(ref list, idx++) = new Move((ushort)(prefix | (int)Bitboard.PopLsb(ref to)));
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Captures(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int prefix = ((int)MoveFlags.Capture << 12) | ((int)from << 6);
        while (to != 0) Unsafe.Add(ref list, idx++) = new Move((ushort)(prefix | (int)Bitboard.PopLsb(ref to)));
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DoublePushes(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int prefix = ((int)MoveFlags.DoublePush << 12) | ((int)from << 6);
        while (to != 0) Unsafe.Add(ref list, idx++) = new Move((ushort)(prefix | (int)Bitboard.PopLsb(ref to)));
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PromoQuiets(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int fromBits = (int)from << 6;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrKnight << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrBishop << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrRook << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrQueen << 12) | fromBits | sq));
        }
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PromoCaptures(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int fromBits = (int)from << 6;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcKnight << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcBishop << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcRook << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcQueen << 12) | fromBits | sq));
        }
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PawnMoves(ulong to, int dir, MoveFlags flag)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int flagBits = (int)flag << 12;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            Unsafe.Add(ref list, idx++) = new Move((ushort)(flagBits | ((sq - dir) << 6) | sq));
        }
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PawnPromos(ulong to, int dir, bool capture)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int knightFlag = (int)(capture ? MoveFlags.PcKnight : MoveFlags.PrKnight) << 12;
        int bishopFlag = (int)(capture ? MoveFlags.PcBishop : MoveFlags.PrBishop) << 12;
        int rookFlag = (int)(capture ? MoveFlags.PcRook : MoveFlags.PrRook) << 12;
        int queenFlag = (int)(capture ? MoveFlags.PcQueen : MoveFlags.PrQueen) << 12;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            int move = ((sq - dir) << 6) | sq;
            Unsafe.Add(ref list, idx++) = new Move((ushort)(knightFlag | move));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(bishopFlag | move));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(rookFlag | move));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(queenFlag | move));
        }
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FromMask(ulong from, Square to, MoveFlags flag)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int flagAndTo = ((int)flag << 12) | (int)to;
        while (from != 0) Unsafe.Add(ref list, idx++) = new Move((ushort)(flagAndTo | ((int)Bitboard.PopLsb(ref from) << 6)));
        _idx = idx;
    }
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void One(Square from, Square to, MoveFlags flag)
    {
        Unsafe.Add(ref _list, _idx++) =
            new Move((ushort)(((int)flag << 12) | ((int)from << 6) | (int)to));
    }
}

internal ref struct NoisySink : IMoveSink
{
    private ref Move _list;
    private int _idx;
    public NoisySink(Span<Move> list) { _list = ref MemoryMarshal.GetReference(list); _idx = 0; }
    public readonly int Count => _idx;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Quiets(Square from, ulong to) { }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Captures(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int prefix = ((int)MoveFlags.Capture << 12) | ((int)from << 6);
        while (to != 0) Unsafe.Add(ref list, idx++) = new Move((ushort)(prefix | (int)Bitboard.PopLsb(ref to)));
        _idx = idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DoublePushes(Square from, ulong to) { }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PromoQuiets(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int fromBits = (int)from << 6;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrKnight << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrBishop << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrRook << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PrQueen << 12) | fromBits | sq));
        }
        _idx = idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PromoCaptures(Square from, ulong to)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int fromBits = (int)from << 6;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcKnight << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcBishop << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcRook << 12) | fromBits | sq));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(((int)MoveFlags.PcQueen << 12) | fromBits | sq));
        }
        _idx = idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PawnMoves(ulong to, int dir, MoveFlags flag)
    {
        if (((int)flag & 0b1100) == 0) return;
        ref Move list = ref _list;
        int idx = _idx;
        int flagBits = (int)flag << 12;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            Unsafe.Add(ref list, idx++) = new Move((ushort)(flagBits | ((sq - dir) << 6) | sq));
        }
        _idx = idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PawnPromos(ulong to, int dir, bool capture)
    {
        ref Move list = ref _list;
        int idx = _idx;
        int knightFlag = (int)(capture ? MoveFlags.PcKnight : MoveFlags.PrKnight) << 12;
        int bishopFlag = (int)(capture ? MoveFlags.PcBishop : MoveFlags.PrBishop) << 12;
        int rookFlag = (int)(capture ? MoveFlags.PcRook : MoveFlags.PrRook) << 12;
        int queenFlag = (int)(capture ? MoveFlags.PcQueen : MoveFlags.PrQueen) << 12;
        while (to != 0)
        {
            int sq = (int)Bitboard.PopLsb(ref to);
            int move = ((sq - dir) << 6) | sq;
            Unsafe.Add(ref list, idx++) = new Move((ushort)(knightFlag | move));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(bishopFlag | move));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(rookFlag | move));
            Unsafe.Add(ref list, idx++) = new Move((ushort)(queenFlag | move));
        }
        _idx = idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FromMask(ulong from, Square to, MoveFlags flag)
    {
        if (((int)flag & 0b1100) == 0) return;
        ref Move list = ref _list;
        int idx = _idx;
        int flagAndTo = ((int)flag << 12) | (int)to;
        while (from != 0) Unsafe.Add(ref list, idx++) = new Move((ushort)(flagAndTo | ((int)Bitboard.PopLsb(ref from) << 6)));
        _idx = idx;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void One(Square from, Square to, MoveFlags flag)
    {
        if (((int)flag & 0b1100) == 0) return;
        Unsafe.Add(ref _list, _idx++) =
            new Move((ushort)(((int)flag << 12) | ((int)from << 6) | (int)to));
    }
}

/// <summary>Counts legal moves without materialising any. Used by perft leaf nodes for a bulk count.</summary>
internal struct CountSink : IMoveSink
{
    private int _count;
    public readonly int Count => _count;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Quiets(Square from, ulong to) => _count += BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void Captures(Square from, ulong to) => _count += BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void DoublePushes(Square from, ulong to) => _count += BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PromoQuiets(Square from, ulong to) => _count += 4 * BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PromoCaptures(Square from, ulong to) => _count += 4 * BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PawnMoves(ulong to, int dir, MoveFlags flag) => _count += BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void PawnPromos(ulong to, int dir, bool capture) => _count += 4 * BitOperations.PopCount(to);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void FromMask(ulong from, Square to, MoveFlags flag) => _count += BitOperations.PopCount(from);
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void One(Square from, Square to, MoveFlags flag) => _count++;
}

/// <summary>
/// Fully-legal move generation. Rather than generating pseudo-legal moves and filtering each one with a
/// make/unmake legality test, this computes three things up front for the side to move:
/// <list type="bullet">
///   <item><description>the <b>danger</b> squares the king may not enter (everything the enemy attacks, x-raying through our own king),</description></item>
///   <item><description>the <b>pinned</b> pieces (and the lines they are pinned to), and</description></item>
///   <item><description>the <b>checkers</b> giving check.</description></item>
/// </list>
/// With those in hand, every move it emits is already legal: king moves avoid danger squares, pinned pieces
/// stay on their pin line, and when in check moves must capture the checker or block the check. The generic
/// <typeparamref name="TUs"/> parameter lets the JIT specialise the whole routine per colour. This is the
/// performance-critical core that perft and the search hammer, so it is written for speed over brevity.
/// </summary>
public static class MoveGeneration
{
    private const ulong NotFileA = 0xfefefefefefefefeUL;
    private const ulong NotFileH = 0x7f7f7f7f7f7f7f7fUL;
    private const ulong Rank2 = 0x000000000000ff00UL;
    private const ulong Rank3 = 0x0000000000ff0000UL;
    private const ulong Rank6 = 0x0000ff0000000000UL;
    private const ulong Rank7 = 0x00ff000000000000UL;

    /// <summary>
    /// Writes all legal moves for the side <typeparamref name="TUs"/> into <paramref name="moveList"/> and
    /// returns how many were written. The span must be large enough for the maximum (well under 256) moves.
    /// As a side effect it refreshes <see cref="Position.Checkers"/> and <see cref="Position.Pinned"/>.
    /// </summary>
    public static int GenerateLegalsInto<TUs>(this Position pos, Span<Move> moveList) where TUs : IColor, new()
    {
        var sink = new EmitSink(moveList);
        pos.GenerateLegals<TUs, EmitSink>(ref sink);
        return sink.Count;
    }

    public static int GenerateNoisyLegalsInto<TUs>(this Position pos, Span<Move> moveList) where TUs : IColor, new()
    {
        var sink = new NoisySink(moveList);
        pos.GenerateLegals<TUs, NoisySink>(ref sink);
        return sink.Count;
    }

    /// <summary>
    /// Counts the legal moves for the side <typeparamref name="TUs"/> without materialising any, for perft
    /// leaf nodes. Same legality rules as <see cref="GenerateLegalsInto{TUs}"/> (it shares the body), but the
    /// destination bitboards are popcounted instead of expanded into a move list.
    /// </summary>
    public static int CountLegals<TUs>(this Position pos) where TUs : IColor, new()
    {
        var sink = new CountSink();
        pos.GenerateLegals<TUs, CountSink>(ref sink);
        return sink.Count;
    }

    /// <summary>
    /// The shared generator body. <typeparamref name="TSink"/> decides what happens with each legal-move
    /// bitboard (write them out, or just count them); the JIT specialises this per (colour, sink) pair so the
    /// sink calls inline away to either move stores or popcounts.
    /// </summary>
    private static void GenerateLegals<TUs, TSink>(this Position pos, ref TSink sink)
        where TUs : IColor, new()
        where TSink : IMoveSink, allows ref struct
    {
        // Colour is a compile-time constant via TUs; take it directly and flip it, rather than the
        // IColor.Opposite() path which boxed the colour struct (CastHelpers.Box showed up per node).
        var usColor = new TUs().Value;
        var themColor = usColor.Flip();

        ulong usBb = pos.AllPieces(usColor);
        ulong themBb = pos.AllPieces(themColor);
        ulong all = usBb | themBb;
        ref readonly UndoInfo state = ref pos.History[pos.Ply];
        Square epsq = state.Epsq;
        CastlingRights castling = state.Castling;

        Square ourKing = Bitboard.Bsf(pos.BitboardOf(usColor, PieceType.King));
        Square theirKing = Bitboard.Bsf(pos.BitboardOf(themColor, PieceType.King));

        if (ourKing == Square.NoSquare)
        {
            return;
        }

        if (theirKing == Square.NoSquare)
        {
            return;
        }

        ulong theirDiagSliders = pos.DiagonalSliders(themColor);
        ulong theirOrthSliders = pos.OrthogonalSliders(themColor);

        // Cache our own piece bitboards once per node and thread them into the move-emitting helpers, instead
        // of re-reading Position.BitboardOf / DiagonalSliders / OrthogonalSliders inside each helper.
        ulong usPawns = pos.BitboardOf(usColor, PieceType.Pawn);
        ulong usKnights = pos.BitboardOf(usColor, PieceType.Knight);
        ulong usDiagSliders = pos.DiagonalSliders(usColor);
        ulong usOrthSliders = pos.OrthogonalSliders(usColor);

        ulong b1;

        ulong themPawns = pos.BitboardOf(themColor, PieceType.Pawn);
        ulong themKnights = pos.BitboardOf(themColor, PieceType.Knight);

        // danger squares (everything the enemy attacks). The king may not move into these.
        ulong danger = Tables.PawnAttacks(themColor, themPawns)
                     | Tables.KingAttacks(theirKing);

        b1 = themKnights;
        while (b1 != 0) danger |= Tables.KnightAttacks(Bitboard.PopLsb(ref b1));

        // Sliders x-ray through our king (so squares behind it stay "in danger" and the king can't
        // step back along the check ray). The king-removed occupancy is loop-invariant, so compute it once.
        ulong allNoKing = all ^ (1UL << (int)ourKing);

        ulong kingTargets = Tables.KingAttacks(ourKing) & ~usBb;
        if (kingTargets != 0 || (castling != CastlingRights.None && AnyCastleCorridorClear(usColor, all, castling)))
        {
            b1 = theirDiagSliders;
            while (b1 != 0) danger |= Tables.GetBishopAttacks(Bitboard.PopLsb(ref b1), allNoKing);

            b1 = theirOrthSliders;
            while (b1 != 0) danger |= Tables.GetRookAttacks(Bitboard.PopLsb(ref b1), allNoKing);
        }

        b1 = kingTargets & ~danger;
        sink.Quiets(ourKing, b1 & ~themBb);
        sink.Captures(ourKing, b1 & themBb);

        // captureMask / quietMask constrain where non-king pieces may move (set per check-count below).
        ulong captureask;
        ulong quietMask;
        Square s;

        // --- Phase 3: find checkers and pins. ---
        // Kept in locals through the scan so the repeated reads/XORs stay in registers; Checkers/Pinned are
        // written back to the heap once (the search reads them after generation) just before the dispatch.
        // Leaper checks (knight/pawn) are direct; they can never pin, so they go straight into checkers.
        ulong checkers = Tables.KnightAttacks(ourKing) & themKnights
                       | Tables.PawnAttacks(usColor, ourKing) & themPawns;

        // Candidate sliders are enemy sliders that would hit our king if our own pieces weren't in the way.
        ulong candidates = Tables.GetRookAttacks(ourKing, themBb) & theirOrthSliders
                        | Tables.GetBishopAttacks(ourKing, themBb) & theirDiagSliders;

        ulong pinned = 0;
        while (candidates != 0)
        {
            s = Bitboard.PopLsb(ref candidates);
            b1 = Tables.SQUARES_BETWEEN[((int)ourKing << 6) | (int)s] & usBb;

            if (b1 == 0)
                checkers ^= 1UL << (int)s;                // nothing between: it's a checking slider
            else if ((b1 & (b1 - 1)) == 0)
                pinned ^= b1;                             // exactly one of our pieces between: it's pinned
        }

        ulong notPinned = ~pinned;

        // --- Phase 4: branch on how many pieces give check. ---
        switch (Bitboard.SparsePopCount(checkers))
        {
            case 2:
                // Double check: only the king can move, and those were already generated above.
                return;

            case 1:
                {
                    // Single check: a non-king piece must either capture the checker or block the ray.
                    // captureMask = the checker's square; quietMask = the squares between king and checker.
                    Square checkerSquare = Bitboard.Bsf(checkers);
                    var checkerPiece = pos.At(checkerSquare);

                    if (checkerPiece == Types.MakePiece(themColor, PieceType.Pawn))
                    {
                        // A checking pawn can itself be captured en passant (the rare case the EP pawn is the checker).
                        if (epsq != Square.NoSquare)
                        {
                            var epTarget = 1UL << (int)epsq;
                            var southDir = Types.RelativeDir(usColor, Direction.South);
                            if (checkers == Bitboard.Shift(southDir, epTarget))
                            {
                                b1 = Tables.PawnAttacks(themColor, epsq) & usPawns & notPinned;
                                sink.FromMask(b1, epsq, MoveFlags.EnPassant);
                            }
                        }
                        captureask = checkers;
                        quietMask = 0;
                        break;
                    }
                    else if (checkerPiece == Types.MakePiece(themColor, PieceType.Knight))
                    {
                        // A knight check can't be blocked, only captured (a pinned piece can't make the capture).
                        b1 = pos.AttackersFrom(usColor, checkerSquare, all) & notPinned;
                        sink.FromMask(b1, checkerSquare, MoveFlags.Capture);
                        return;
                    }
                    else
                    {
                        // A sliding check can be answered by capturing the checker or blocking along the ray.
                        captureask = checkers;
                        quietMask = Tables.SQUARES_BETWEEN[((int)ourKing << 6) | (int)checkerSquare];
                        break;
                    }
                }
            case 0:
            default:
                // Not in check: captures may take any enemy piece and quiet moves may go to any empty square.
                captureask = themBb;
                quietMask = ~all;

                // En passant and castling are only available when not in check. Castling and pinned-piece
                // handling do nothing when there are no rights / no pinned pieces (the common case deep in the
                // tree), so guard the calls to skip the argument setup entirely rather than enter and fall out.
                if (epsq != Square.NoSquare)
                {
                    HandleEnPassantInto(usColor, usPawns, notPinned, pinned, ourKing, theirOrthSliders, all, epsq, ref sink);
                }

                if (castling != CastlingRights.None)
                {
                    HandleCastlingInto(usColor, all, danger, castling, ref sink);
                }

                if (pinned != 0)
                {
                    HandlePinnedPiecesInto(pos, usColor, usPawns, usKnights, themBb, notPinned, ourKing, all, captureask, quietMask, ref sink);
                }

                break;
        }

        // Non-pinned pieces (knights, sliders, pawns) move freely within capture/quiet masks. Shared by the
        // not-in-check and single-check paths; pinned pieces were already handled (and skipped) when in check.
        HandleNonPinnedPiecesInto(usColor, usPawns, usKnights, usDiagSliders, usOrthSliders, notPinned, all, captureask, quietMask, ref sink);
    }

    /// <summary>
    /// Generates en-passant captures. The subtlety is the "en-passant pin": capturing removes two pawns from
    /// the same rank at once, which can momentarily expose the king to a rook/queen on that rank, so each
    /// candidate is verified against the king's rank with both pawns removed before it is emitted. A pawn
    /// pinned toward the king can still capture en passant if the capture stays on the pin line.
    /// </summary>
    private static void HandleEnPassantInto<TSink>(Color usColor, ulong usPawns, ulong notPinned,
        ulong pinned, Square ourKing, ulong theirOrthSliders, ulong all, Square epsq, ref TSink sink)
        where TSink : IMoveSink, allows ref struct
    {
        ulong b1, b2;
        Square s;

        b2 = Tables.PawnAttacks(usColor.Flip(), epsq) & usPawns;
        b1 = b2 & notPinned;
        var southDir = Types.RelativeDir(usColor, Direction.South);
        var epCaptureSquare = (Square)((int)epsq + (int)southDir);
        var epCaptureSquareBb = 1UL << (int)epCaptureSquare;
        var rankMask = Bitboard.MASK_RANK[(int)Types.RankOf(ourKing)];

        while (b1 != 0)
        {
            s = Bitboard.PopLsb(ref b1);

            var newOcc = all ^ (1UL << (int)s) ^ epCaptureSquareBb;

            if ((Tables.SlidingAttacks(ourKing, newOcc, rankMask) & theirOrthSliders) == 0)
                sink.One(s, epsq, MoveFlags.EnPassant);
        }

        b1 = b2 & pinned & Tables.LINE_BB[((int)epsq << 6) | (int)ourKing];
        if (b1 != 0)
        {
            sink.One(Bitboard.Bsf(b1), epsq, MoveFlags.EnPassant);
        }
    }

    /// <summary>
    /// Generates the two castling moves for the side to move when still legal: the right must be intact, the
    /// squares between king and rook empty, and the king's start/transit/destination squares free of danger.
    /// The caller only invokes this when not in check.
    /// </summary>
    private static void HandleCastlingInto<TSink>(Color usColor, ulong all, ulong danger,
        CastlingRights rights, ref TSink sink)
        where TSink : IMoveSink, allows ref struct
    {
        if (usColor == Color.White)
        {
            if ((rights & CastlingRights.WhiteOO) != 0 &&
                ((all & 0x60UL) | (danger & 0x70UL)) == 0)
            {
                sink.One(Square.e1, Square.g1, MoveFlags.OO);
            }

            if ((rights & CastlingRights.WhiteOOO) != 0 &&
                ((all & 0xeUL) | (danger & 0x1cUL)) == 0)
            {
                sink.One(Square.e1, Square.c1, MoveFlags.OOO);
            }
        }
        else
        {
            if ((rights & CastlingRights.BlackOO) != 0 &&
                ((all & 0x6000000000000000UL) | (danger & 0x7000000000000000UL)) == 0)
            {
                sink.One(Square.e8, Square.g8, MoveFlags.OO);
            }

            if ((rights & CastlingRights.BlackOOO) != 0 &&
                ((all & 0x0e00000000000000UL) | (danger & 0x1c00000000000000UL)) == 0)
            {
                sink.One(Square.e8, Square.c8, MoveFlags.OOO);
            }
        }
    }

    /// <summary>
    /// True if at least one castling corridor is clear
    /// False if king is boxed in, no move can consume the slider danger map, so the per-node sweep that builds it is skipped.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool AnyCastleCorridorClear(Color us, ulong all, CastlingRights rights)
    {
        return us == Color.White
            ? ((rights & CastlingRights.WhiteOO) != 0 && (all & 0x60UL) == 0)
              || ((rights & CastlingRights.WhiteOOO) != 0 && (all & 0xeUL) == 0)
            : ((rights & CastlingRights.BlackOO) != 0 && (all & 0x6000000000000000UL) == 0)
              || ((rights & CastlingRights.BlackOOO) != 0 && (all & 0x0e00000000000000UL) == 0);
    }

    /// <summary>
    /// Generates moves for pinned pieces. A pinned piece may only move along the line between the king and the
    /// pinning piece, so every destination is intersected with <c>LINE[king][piece]</c>. A pinned knight can
    /// never move at all (it is excluded here), and pinned-pawn pushes/captures/promotions are handled inline.
    /// Only reached when not in check (a pinned piece can never resolve a check).
    /// </summary>
    private static void HandlePinnedPiecesInto<TSink>(Position pos, Color usColor, ulong usPawns, ulong usKnights,
        ulong themBb, ulong notPinned, Square ourKing, ulong all, ulong captureask, ulong quietMask, ref TSink sink)
        where TSink : IMoveSink, allows ref struct
    {
        ulong b1, b2, b3;
        Square s;

        // Pinned non-knights: a pinned knight has no legal move, so knights are masked out of this set.
        b1 = ~(notPinned | usKnights);
        while (b1 != 0)
        {
            s = Bitboard.PopLsb(ref b1);
            var pt = Types.TypeOf(pos.At(s));
            if (pt == PieceType.Pawn) continue;
            ulong line = Tables.LINE_BB[((int)ourKing << 6) | (int)s];
            b2 = Tables.Attacks(pt, s, all) & line;
            sink.Quiets(s, b2 & quietMask);
            sink.Captures(s, b2 & captureask);
        }

        b1 = ~notPinned & usPawns;
        while (b1 != 0)
        {
            s = Bitboard.PopLsb(ref b1);

            ulong line = Tables.LINE_BB[((int)ourKing << 6) | (int)s];
            if (Types.RankOf(s) == Types.RelativeRank(usColor, Rank.Rank7))
            {
                b2 = Tables.PawnAttacks(usColor, s) & captureask & line;
                sink.PromoCaptures(s, b2);

                var northDir = Types.RelativeDir(usColor, Direction.North);
                b2 = Bitboard.Shift((Direction)northDir, 1UL << (int)s) & ~all & line;
                sink.PromoQuiets(s, b2);
            }
            else
            {
                b2 = Tables.PawnAttacks(usColor, s) & themBb & line;

                sink.Captures(s, b2);

                var northDir = Types.RelativeDir(usColor, Direction.North);
                b2 = Bitboard.Shift((Direction)northDir, 1UL << (int)s) & ~all & line;

                b3 = Bitboard.Shift((Direction)northDir, b2 & Bitboard.MASK_RANK[(int)Types.RelativeRank(usColor, Rank.Rank3)])
                   & ~all & line;

                sink.Quiets(s, b2);
                sink.DoublePushes(s, b3);
            }
        }
    }

    /// <summary>
    /// Generates moves for all non-pinned, non-king pieces (knights, bishops, rooks, queens, then pawns),
    /// each restricted to <paramref name="captureask"/> for captures and <paramref name="quietMask"/> for
    /// quiet moves. Those masks already encode the "must block or capture the checker" rule under single check.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HandleNonPinnedPiecesInto<TSink>(Color usColor, ulong usPawns, ulong usKnights,
        ulong usDiagSliders, ulong usOrthSliders, ulong notPinned, ulong all, ulong captureask, ulong quietMask,
        ref TSink sink) where TSink : IMoveSink, allows ref struct
    {
        ulong b1, b2;
        Square s;

        b1 = usKnights & notPinned;
        while (b1 != 0)
        {
            s = Bitboard.PopLsb(ref b1);
            b2 = Tables.KnightAttacks(s);
            sink.Quiets(s, b2 & quietMask);
            sink.Captures(s, b2 & captureask);
        }

        b1 = usDiagSliders & notPinned;
        while (b1 != 0)
        {
            s = Bitboard.PopLsb(ref b1);
            b2 = Tables.GetBishopAttacks(s, all);
            sink.Quiets(s, b2 & quietMask);
            sink.Captures(s, b2 & captureask);
        }

        b1 = usOrthSliders & notPinned;
        while (b1 != 0)
        {
            s = Bitboard.PopLsb(ref b1);
            b2 = Tables.GetRookAttacks(s, all);
            sink.Quiets(s, b2 & quietMask);
            sink.Captures(s, b2 & captureask);
        }

        HandleNonPinnedPawnsInto(usPawns, usColor, notPinned, all, captureask, quietMask, ref sink);
    }

    /// <summary>
    /// Generates all non-pinned pawn moves: single and double pushes, left/right captures, and promotions
    /// (quiet and capturing). Whole groups of pawns are advanced at once by shifting their bitboard in the
    /// forward direction, then each resulting destination bit is turned back into a from/to move. The "north"
    /// directions are taken relative to the side to move, so the same code serves both colours.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HandleNonPinnedPawnsInto<TSink>(ulong usPawns, Color usColor, ulong notPinned,
        ulong all, ulong captureask, ulong quietMask, ref TSink sink) where TSink : IMoveSink, allows ref struct
    {
        ulong b1, b2, b3;

        ulong pawns = usPawns & notPinned;
        if (usColor == Color.White)
        {
            b1 = pawns & ~Rank7;
            b2 = (b1 << 8) & ~all;
            b3 = ((b2 & Rank3) << 8) & quietMask;
            b2 &= quietMask;

            sink.PawnMoves(b2, (int)Direction.North, MoveFlags.Quiet);
            sink.PawnMoves(b3, (int)Direction.NorthNorth, MoveFlags.DoublePush);

            b2 = ((b1 & NotFileA) << 7) & captureask;
            b3 = ((b1 & NotFileH) << 9) & captureask;

            sink.PawnMoves(b2, (int)Direction.NorthWest, MoveFlags.Capture);
            sink.PawnMoves(b3, (int)Direction.NorthEast, MoveFlags.Capture);

            b1 = pawns & Rank7;
            if (b1 != 0)
            {
                b2 = (b1 << 8) & quietMask;
                sink.PawnPromos(b2, (int)Direction.North, capture: false);

                b2 = ((b1 & NotFileA) << 7) & captureask;
                b3 = ((b1 & NotFileH) << 9) & captureask;

                sink.PawnPromos(b2, (int)Direction.NorthWest, capture: true);
                sink.PawnPromos(b3, (int)Direction.NorthEast, capture: true);
            }
        }
        else
        {
            b1 = pawns & ~Rank2;
            b2 = (b1 >> 8) & ~all;
            b3 = ((b2 & Rank6) >> 8) & quietMask;
            b2 &= quietMask;

            sink.PawnMoves(b2, (int)Direction.South, MoveFlags.Quiet);
            sink.PawnMoves(b3, (int)Direction.SouthSouth, MoveFlags.DoublePush);

            b2 = ((b1 & NotFileA) >> 9) & captureask;
            b3 = ((b1 & NotFileH) >> 7) & captureask;

            sink.PawnMoves(b2, (int)Direction.SouthWest, MoveFlags.Capture);
            sink.PawnMoves(b3, (int)Direction.SouthEast, MoveFlags.Capture);

            b1 = pawns & Rank2;
            if (b1 != 0)
            {
                b2 = (b1 >> 8) & quietMask;
                sink.PawnPromos(b2, (int)Direction.South, capture: false);

                b2 = ((b1 & NotFileA) >> 9) & captureask;
                b3 = ((b1 & NotFileH) >> 7) & captureask;

                sink.PawnPromos(b2, (int)Direction.SouthWest, capture: true);
                sink.PawnPromos(b3, (int)Direction.SouthEast, capture: true);
            }
        }
    }
}
