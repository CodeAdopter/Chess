using System.Diagnostics;
using Core;
using Engine.Nnue;

namespace Engine;

public struct SearchLimits
{
    public int MaxDepth;     // hard depth cap
    public long MoveTimeMs;  // 0 => no explicit move-time budget
    public bool Infinite;    // search until told to stop

    public static SearchLimits Depth(int d) => new() { MaxDepth = d, Infinite = false };
}

/// <summary>
/// Alpha-beta searcher: iterative deepening, negamax with PVS, quiescence, a transposition table,
/// null-move pruning, and MVV-LVA / killer / history move ordering. With NNUE enabled, the network is
/// used as a learned correction on top of the hand-crafted eval so training can distil the search/eval
/// gap without relearning every material bucket from scratch.
/// </summary>
public sealed class Searcher(int ttSizeMb = 64)
{
    private const int Inf = 32000;
    private const int MaxPly = 128;

    // ---- self-play generation speed flags ----
    // The "sound stack" (move-choice preserving) defaults ON; disable an individual one with OPT_X=0.
    // SEE is redundant with DELTA, and LMR reduces effective search depth (a data-quality risk), so both default OFF.
    // Per-instance (not static) so an A/B match can run two differently-configured searchers in one process.
    private bool optDelta = Environment.GetEnvironmentVariable("OPT_DELTA") != "0";  // quiescence delta pruning
    private bool optSee   = Environment.GetEnvironmentVariable("OPT_SEE")   == "1";  // skip losing captures (SEE<0) in qsearch
    private bool optRfp   = Environment.GetEnvironmentVariable("OPT_RFP")   != "0";  // reverse futility / static null move
    private bool optQtt   = Environment.GetEnvironmentVariable("OPT_QTT")   != "0";  // quiescence TT probe/store cutoffs
    private bool optLmr   = Environment.GetEnvironmentVariable("OPT_LMR")   == "1";  // late move reductions
    private static readonly bool OptIncEval = Environment.GetEnvironmentVariable("OPT_INCEVAL") != "0";  // incremental HCE (bit-exact)

    /// <summary>Override the per-search optimisation flags (for A/B strength matches). INCEVAL is excluded
    /// because it is bit-exact and never affects move choice, so it stays globally configured.</summary>
    public void SetOpts(bool delta, bool see, bool rfp, bool qtt, bool lmr)
    { optDelta = delta; optSee = see; optRfp = rfp; optQtt = qtt; optLmr = lmr; }

    private const int DeltaMargin  = 200;   // cp slack for quiescence delta pruning
    private const int RfpMaxDepth  = 3;     // only RFP at shallow nodes
    private const int RfpMargin    = 80;    // cp per ply for RFP
    private const int SeeKingValue = 10000; // king never voluntarily recaptures into a defended square

    // LMR reduction table [depth, moveNumber], precomputed once (depends only on depth & move index).
    private static readonly int[,] LmrTable = BuildLmrTable();
    private static int[,] BuildLmrTable()
    {
        var t = new int[MaxPly, 64];
        for (int d = 1; d < MaxPly; d++)
            for (int mn = 1; mn < 64; mn++)
                t[d, mn] = (int)(0.75 + Math.Log(d) * Math.Log(mn) / 2.25);
        return t;
    }

    private readonly TranspositionTable tt = new(ttSizeMb);
    private readonly int[][] history = [new int[64 * 64], new int[64 * 64]];
    private readonly Move[,] killers = new Move[MaxPly, 2];

    private long nodes;
    private long deadline;
    private bool infinite;
    private bool stop;
    private readonly Stopwatch clock = new();

    private Move pvMove;   // best root move found so far (survives an aborted iteration)

    private QAccumulatorStack? acc;  // incremental int16 NNUE accumulator (quantised), synced with make/unmake
    private NnueEvaluator? verify;   // float full-refresh reference, only when NNUE_VERIFY=1 (quantisation-error check)
    public long VerifyMismatches { get; private set; }

    /// <summary>
    /// Suppress per-iteration "info" output (for self-play / benchmarking).
    /// </summary>
    public bool Quiet { get; set; }

    private long interiorNodes;   // Negamax interior nodes this search
    private long qNodes;          // quiescence nodes this search

    public long LastNodes { get; private set; }
    public long LastInteriorNodes { get; private set; }   // benchmark instrumentation: interior vs q split
    public long LastQNodes { get; private set; }
    public long LastMs { get; private set; }
    public int LastScore { get; private set; }   // score (cp, side-to-move POV) of the last completed search

    /// <summary>
    /// Use an NNUE network (incrementally evaluated) for leaf eval; null reverts to hand-crafted.
    /// </summary>
    public void SetNnue(NnueNetwork? net)
    {
        acc = net == null ? null : new QAccumulatorStack(new QuantizedNetwork(net));
        verify = net != null && Environment.GetEnvironmentVariable("NNUE_VERIFY") == "1" ? new NnueEvaluator(net) : null;
    }

    /// <summary>
    /// The leaf evaluation used everywhere in the search. With no network loaded this is just the
    /// hand-crafted <see cref="Eval.Evaluate"/>. With a network, the NNUE output is treated as a learned
    /// correction added on top of the hand-crafted score (clamped to a sane range), which is why training
    /// only has to learn the gap rather than re-derive material values from scratch.
    /// </summary>
    private int EvalPos(Position pos)
    {
        if (acc == null) return Eval.Evaluate(pos);
        int correction = acc.Evaluate(pos.Turn);
        if (verify != null && correction != verify.Evaluate(pos)) VerifyMismatches++;
        // OPT_INCEVAL: use the incrementally-maintained hand-crafted eval (bit-identical to Eval.Evaluate)
        // instead of re-scanning all pieces every q-node. Pure speed; result is unchanged.
        int hc = OptIncEval ? acc.EvaluateHce(pos.Turn) : Eval.Evaluate(pos);
        return Math.Clamp(hc + correction, -10000, 10000);
    }

    // Make/unmake wrappers that keep the incremental NNUE accumulator in lockstep with the position, so the
    // network never has to be re-evaluated from scratch. They fall back to plain Play/Undo when no net is loaded.
    /// <summary>
    /// Plays a move and pushes the matching incremental NNUE update.
    /// </summary>
    private void PlayMove(Position pos, Color us, Move m)
    {
        if (acc != null)
        {
            Piece moving = pos.At(m.From);
            Piece captured = m.IsCapture && m.Flags != MoveFlags.EnPassant ? pos.At(m.To) : Piece.NoPiece;
            pos.Play(us, m);
            acc.Push(pos, m, moving, captured);
        }
        else pos.Play(us, m);
    }

    /// <summary>
    /// Undoes a move and pops the matching incremental NNUE update.
    /// </summary>
    private void UndoMove(Position pos, Color us, Move m)
    {
        pos.Undo(us, m);
        acc?.Pop();
    }

    /// <summary>
    /// Reset per-game search state. <paramref name="clearTt"/> defaults to true (UCI: a fresh game should
    /// not inherit the previous game's table). Self-play passes false: the TT is full-key-verified on probe,
    /// so leftover entries from prior games are auto-rejected. Clearing 16 MB before every one of hundreds
    /// of games is pure memset waste, and warm cross-game entries are a small speedup.
    /// </summary>
    public void NewGame(bool clearTt = true)
    {
        if (clearTt) tt.Clear();
        Array.Clear(history[0]);
        Array.Clear(history[1]);
    }

    /// <summary>
    /// Run iterative deepening and return the best move found within the limits.
    /// </summary>
    public Move Think(Position pos, SearchLimits limits)
    {
        nodes = 0;
        interiorNodes = 0;
        qNodes = 0;
        stop = false;
        infinite = limits.Infinite;
        clock.Restart();
        deadline = limits.MoveTimeMs > 0 ? limits.MoveTimeMs : long.MaxValue;

        Array.Clear(killers, 0, killers.Length);

        Span<Move> rootMoves = stackalloc Move[256];
        int rootCount = GenerateLegal(pos, rootMoves);
        if (rootCount == 0) return default;            // mate/stalemate at root
        pvMove = rootMoves[0];                          // always have a legal fallback

        acc?.RefreshRoot(pos);   // sync the incremental accumulator to the root once per search

        int maxDepth = limits.MaxDepth > 0 ? limits.MaxDepth : MaxPly - 1;
        int lastScore = 0;

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            int score = Negamax(pos, depth, -Inf, Inf, 0);

            if (stop && depth > 1) break;               // discard partial iteration's score

            lastScore = score;
            PrintInfo(pos, depth, score);

            if (Math.Abs(score) >= Eval.MateBound) break; // forced mate found
            if (OutOfTime()) break;
        }

        LastNodes = nodes;
        LastInteriorNodes = interiorNodes;
        LastQNodes = qNodes;
        LastMs = clock.ElapsedMilliseconds;
        LastScore = lastScore;
        return pvMove;
    }

    /// <summary>
    /// The core negamax search with alpha-beta pruning and principal-variation search (PVS). Returns the
    /// score (centipawns, side-to-move POV) of the best line from <paramref name="pos"/> to the given
    /// <paramref name="depth"/>. Along the way it consults and updates the transposition table, applies
    /// reverse-futility and null-move pruning, orders moves, and (optionally) reduces late quiet moves (LMR).
    /// At depth 0 it hands off to <see cref="Quiesce"/>. The window is <paramref name="alpha"/>..<paramref
    /// name="beta"/>; <paramref name="ply"/> is the distance from the root (used for mate scoring).
    /// </summary>
    private int Negamax(Position pos, int depth, int alpha, int beta, int ply)
    {
        if (stop) return 0;
        if ((nodes & 4095) == 0 && OutOfTime()) { stop = true; return 0; }

        bool root = ply == 0;

        // Draw detection (skipped at the root, where the game isn't actually over).
        if (!root && (pos.IsRepetition() || pos.IsFiftyMoveRule()))
            return 0;

        if (depth <= 0)
            return Quiesce(pos, alpha, beta, ply);

        nodes++;
        interiorNodes++;

        ulong key = pos.GetHash();
        Move ttMove = default;
        if (tt.Probe(key, out var te))
        {
            ttMove = new Move(te.Move);
            if (!root && te.Depth >= depth)
            {
                int s = TranspositionTable.ScoreFromTt(te.Score, ply);
                switch (te.Flag)
                {
                    case TtFlag.Exact: return s;
                    case TtFlag.Lower: if (s >= beta) return s; break;
                    case TtFlag.Upper: if (s <= alpha) return s; break;
                }
            }
        }

        bool inCheck = pos.InCheck(pos.Turn);

        // Reverse futility pruning (OPT_RFP / static null move): at shallow non-root, non-check nodes, if
        // the static eval already clears beta by a depth-scaled margin, assume it holds and fail high
        // without searching, skipping the entire quiescence subtree the node would otherwise spawn.
        if (optRfp && !root && !inCheck && depth <= RfpMaxDepth
            && beta < Eval.MateBound && beta > -Eval.MateBound)
        {
            int staticEval = EvalPos(pos);
            if (staticEval - RfpMargin * depth >= beta)
                return staticEval - RfpMargin * depth;
        }

        // Null-move pruning: give the opponent a free move; if we're still winning, prune. Skipped in
        // check and in likely-zugzwang (no non-pawn material), where passing isn't safely equivalent.
        if (!root && !inCheck && depth >= 3 && beta < Eval.MateBound
            && HasNonPawnMaterial(pos, pos.Turn))
        {
            int r = 2 + depth / 6;
            pos.MakeNullMove(); acc?.PushNull();
            int nullScore = -Negamax(pos, depth - 1 - r, -beta, -beta + 1, ply + 1);
            pos.UnmakeNullMove(); acc?.Pop();
            if (stop) return 0;
            if (nullScore >= beta) return beta;
        }

        Span<Move> moves = stackalloc Move[256];
        int count = GenerateLegal(pos, moves);

        if (count == 0)
            return inCheck ? -Eval.MateValue + ply : 0;  // checkmate (ply-adjusted) or stalemate

        Span<int> scores = stackalloc int[count];
        ScoreMoves(pos, moves, count, scores, ttMove, ply);

        int bestScore = -Inf;
        Move bestMove = default;
        int origAlpha = alpha;
        Color us = pos.Turn;

        for (int i = 0; i < count; i++)
        {
            PickNext(moves, scores, count, i);
            Move m = moves[i];

            PlayMove(pos, us, m);
            int score;
            if (i == 0)
            {
                score = -Negamax(pos, depth - 1, -beta, -alpha, ply + 1);
            }
            else
            {
                // PVS + LMR. Reduce late, quiet, non-PV, non-checking moves; the reduced search uses a
                // null window. If it beats alpha we re-search at full depth (still null window) before the
                // standard PVS full-window widen, so a genuinely good reduced move is always verified.
                int reducedDepth = depth - 1;
                if (optLmr && depth >= 3 && i >= 3 && !inCheck
                    && !m.IsCapture && (m.Flags & MoveFlags.Promotions) == 0
                    && m != ttMove && m != killers[ply, 0] && m != killers[ply, 1]
                    && !pos.InCheck(pos.Turn))   // PlayMove already flipped Turn => tests "m gives check"
                {
                    int rr = LmrTable[Math.Min(depth, MaxPly - 1), Math.Min(i, 63)];
                    if (rr < 1) rr = 1;
                    reducedDepth = Math.Max(1, depth - 1 - rr);
                }

                score = -Negamax(pos, reducedDepth, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && reducedDepth < depth - 1)
                    score = -Negamax(pos, depth - 1, -alpha - 1, -alpha, ply + 1);
                if (score > alpha && score < beta)
                    score = -Negamax(pos, depth - 1, -beta, -alpha, ply + 1);
            }
            UndoMove(pos, us, m);

            if (stop) return 0;

            if (score > bestScore)
            {
                bestScore = score;
                bestMove = m;
                if (root) pvMove = m;
            }
            if (score > alpha) alpha = score;

            if (alpha >= beta)
            {
                // Beta cutoff. Reward quiet cutoff moves so they're tried earlier next time.
                if (!m.IsCapture && (m.Flags & MoveFlags.Promotions) == 0)
                {
                    if (killers[ply, 0] != m) { killers[ply, 1] = killers[ply, 0]; killers[ply, 0] = m; }
                    history[(int)us][((int)m.From << 6) | (int)m.To] += depth * depth;
                }
                break;
            }
        }

        TtFlag flag = bestScore <= origAlpha ? TtFlag.Upper
                    : bestScore >= beta ? TtFlag.Lower
                    : TtFlag.Exact;
        tt.Store(key, PackMove(bestMove), TranspositionTable.ScoreToTt(bestScore, ply), depth, flag);

        return bestScore;
    }

    /// <summary>
    /// Quiescence search: the leaf extension that only plays "noisy" moves (captures and promotions, plus all
    /// evasions when in check) until the position is quiet, so the static eval is not taken in the middle of a
    /// capture exchange (the "horizon effect"). Starts from a "stand-pat" eval that the side to move can take
    /// instead of capturing, then tries the noisy moves with delta/SEE pruning. Returns a centipawn score.
    /// </summary>
    private int Quiesce(Position pos, int alpha, int beta, int ply)
    {
        if (stop) return 0;
        if ((nodes & 4095) == 0 && OutOfTime()) { stop = true; return 0; }
        nodes++;
        qNodes++;

        // OPT_QTT: probe the TT for a transposed capture-line result. q-entries are stored at depth 0 so
        // they can never cut a real Negamax node (its gate is te.Depth >= depth, depth >= 1), while a deep
        // Negamax entry is always usable here. Self-play keeps the TT warm across games (NewGame keeps it).
        ulong key = 0;
        Move qttMove = default;
        if (optQtt)
        {
            key = pos.GetHash();
            if (tt.Probe(key, out var te))
            {
                qttMove = new Move(te.Move);
                int s = TranspositionTable.ScoreFromTt(te.Score, ply);
                switch (te.Flag)
                {
                    case TtFlag.Exact: return s;
                    case TtFlag.Lower: if (s >= beta) return s; break;
                    case TtFlag.Upper: if (s <= alpha) return s; break;
                }
            }
        }

        bool inCheck = pos.InCheck(pos.Turn);

        int origAlpha = alpha;
        int bestScore = -Inf;          // fail-soft best, for a correct TT store
        Move bestMove = default;
        int stand = 0;

        if (!inCheck)
        {
            stand = EvalPos(pos);
            bestScore = stand;
            if (stand >= beta)
            {
                if (optQtt) tt.Store(key, 0, TranspositionTable.ScoreToTt(stand, ply), 0, TtFlag.Lower);
                return beta;
            }
            if (stand > alpha) alpha = stand;
        }

        Span<Move> moves = stackalloc Move[256];
        int count = GenerateLegal(pos, moves);

        if (count == 0)
            return inCheck ? -Eval.MateValue + ply : 0;

        Span<int> scores = stackalloc int[count];
        ScoreMoves(pos, moves, count, scores, qttMove, ply);

        Color us = pos.Turn;
        for (int i = 0; i < count; i++)
        {
            PickNext(moves, scores, count, i);
            Move m = moves[i];

            // When not in check, only explore captures and promotions (the "noisy" moves).
            // In check we must consider every evasion to avoid a horizon blunder.
            if (!inCheck && !m.IsCapture && (m.Flags & MoveFlags.Promotions) == 0)
                continue;

            // OPT_DELTA: an optimistic material gain (captured value + promotion gain + margin) that still
            // can't reach alpha is futile, so skip it before paying for a child q-node + NNUE eval.
            if (optDelta && !inCheck && alpha > -Eval.MateBound)
            {
                int gain = 0;
                if (m.IsCapture)
                    gain += m.Flags == MoveFlags.EnPassant
                        ? Eval.PieceValue[(int)PieceType.Pawn]
                        : Eval.PieceValue[(int)Types.TypeOf(pos.At(m.To))];
                if ((m.Flags & MoveFlags.Promotions) != 0)
                    gain += Eval.PieceValue[((int)m.Flags & 0b11) + 1] - Eval.PieceValue[(int)PieceType.Pawn];
                if (stand + gain + DeltaMargin <= alpha)
                    continue;
            }

            // OPT_SEE: skip captures that lose material on the static exchange, since they cannot raise alpha
            // in a quiet node and only spawn wasted child q-nodes (and NNUE evals).
            if (optSee && !inCheck && m.IsCapture && See(pos, m) < 0)
                continue;

            PlayMove(pos, us, m);
            int score = -Quiesce(pos, -beta, -alpha, ply + 1);
            UndoMove(pos, us, m);

            if (stop) return 0;
            if (score > bestScore) { bestScore = score; bestMove = m; }
            if (score >= beta)
            {
                if (optQtt) tt.Store(key, PackMove(m), TranspositionTable.ScoreToTt(score, ply), 0, TtFlag.Lower);
                return beta;
            }
            if (score > alpha) alpha = score;
        }

        // Store a fail-soft bound (qBest) while still returning alpha, so the search tree shape is
        // unchanged except for the cutoffs the probe introduces.
        if (optQtt)
        {
            TtFlag flag = bestScore <= origAlpha ? TtFlag.Upper : TtFlag.Exact;
            tt.Store(key, PackMove(bestMove), TranspositionTable.ScoreToTt(bestScore, ply), 0, flag);
        }

        return alpha;
    }

    // ---- static exchange evaluation (OPT_SEE) -------------------------------------------------

    private static int SeeValue(PieceType pt) =>
        pt == PieceType.King ? SeeKingValue : Eval.PieceValue[(int)pt];

    /// <summary>All pieces of either colour attacking <paramref name="s"/> given occupancy
    /// <paramref name="occ"/>. AttackersFrom omits kings, so add them explicitly.</summary>
    private static ulong AllAttackersTo(Position pos, Square s, ulong occ) =>
        pos.AttackersFrom(Color.White, s, occ)
      | pos.AttackersFrom(Color.Black, s, occ)
      | (Tables.Attacks(PieceType.King, s, occ)
            & (pos.BitboardOf(Color.White, PieceType.King) | pos.BitboardOf(Color.Black, PieceType.King)));

    /// <summary>Static Exchange Evaluation: net material (cp, mover POV) of the capture sequence on m.To,
    /// both sides always recapturing with their least-valuable attacker. >= 0 means "does not lose material".</summary>
    private int See(Position pos, Move m)
    {
        Square to = m.To, from = m.From;
        Color us = pos.Turn;
        ulong occ = (pos.AllPieces(Color.White) | pos.AllPieces(Color.Black)) & ~(1UL << (int)from);

        bool ep = m.Flags == MoveFlags.EnPassant;
        bool promo = (m.Flags & MoveFlags.Promotions) != 0;

        Span<int> gain = stackalloc int[32];
        int d = 0;

        if (ep)
        {
            gain[0] = SeeValue(PieceType.Pawn);
            int capSq = (int)to + (us == Color.White ? -8 : 8);   // captured pawn sits behind the to-square
            occ &= ~(1UL << capSq);
        }
        else
        {
            gain[0] = SeeValue(Types.TypeOf(pos.At(to)));
        }

        PieceType occupant = Types.TypeOf(pos.At(from));   // the piece value now standing on `to`
        if (promo)
        {
            gain[0] += SeeValue(PieceType.Queen) - SeeValue(PieceType.Pawn);   // pawn becomes a queen on arrival
            occupant = PieceType.Queen;
        }

        Color side = us.Flip();
        ulong attackers = AllAttackersTo(pos, to, occ);

        while (true)
        {
            attackers &= occ;
            ulong sideAtt = attackers & pos.AllPieces(side);
            if (sideAtt == 0) break;   // side to move has no recapture → exchange ends

            // Least-valuable attacker of `side`.
            ulong fromBit = 0;
            PieceType lva = PieceType.Pawn;
            for (int pt = (int)PieceType.Pawn; pt <= (int)PieceType.King; pt++)
            {
                ulong subset = sideAtt & pos.BitboardOf(side, (PieceType)pt);
                if (subset != 0) { lva = (PieceType)pt; fromBit = subset & (~subset + 1); break; }
            }

            d++;
            gain[d] = SeeValue(occupant) - gain[d - 1];      // win the current occupant; our piece now exposed
            if (Math.Max(-gain[d - 1], gain[d]) < 0) break;  // this line is already decided

            occupant = lva;                                  // the recapturing piece now sits on `to`
            occ ^= fromBit;
            attackers = AllAttackersTo(pos, to, occ);        // recompute → reveals x-ray sliders behind it
            side = side.Flip();
            if (d >= 30) break;
        }

        while (--d > 0)
            gain[d - 1] = -Math.Max(-gain[d - 1], gain[d]);

        return gain[0];
    }

    // ---- move ordering -----------------------------------------------------------------------

    /// <summary>
    /// Assigns each move an ordering score so the best candidates are tried first (which makes alpha-beta
    /// prune far more). The priority is: the transposition-table move, then captures/promotions ranked by
    /// MVV-LVA (most-valuable victim, least-valuable attacker), then killer moves, then the history heuristic.
    /// </summary>
    private void ScoreMoves(Position pos, Span<Move> moves, int count, Span<int> scores, Move ttMove, int ply)
    {
        Color us = pos.Turn;
        for (int i = 0; i < count; i++)
        {
            Move m = moves[i];
            if (ttMove.ToFrom != 0 && m == ttMove)
            {
                scores[i] = 3_000_000;
            }
            else if (m.IsCapture || (m.Flags & MoveFlags.Promotions) != 0)
            {
                int victim = m.IsCapture ? PieceTypeValue(pos.At(m.To)) : 0;
                int attacker = PieceTypeValue(pos.At(m.From));
                int promo = (m.Flags & MoveFlags.Promotions) != 0 ? Eval.PieceValue[(int)PieceType.Queen] : 0;
                scores[i] = 1_000_000 + victim * 16 - attacker + promo;
            }
            else if (m == killers[ply, 0] || m == killers[ply, 1])
            {
                scores[i] = 900_000;
            }
            else
            {
                scores[i] = history[(int)us][((int)m.From << 6) | (int)m.To];
            }
        }
    }

    /// <summary>
    /// Selection step: swap the highest-scored remaining move into slot <paramref name="i"/>.
    /// </summary>
    private static void PickNext(Span<Move> moves, Span<int> scores, int count, int i)
    {
        int best = i;
        for (int j = i + 1; j < count; j++)
            if (scores[j] > scores[best]) best = j;
        if (best != i)
        {
            (moves[i], moves[best]) = (moves[best], moves[i]);
            (scores[i], scores[best]) = (scores[best], scores[i]);
        }
    }

    private static int PieceTypeValue(Piece p)
    {
        if (p == Piece.NoPiece) return Eval.PieceValue[(int)PieceType.Pawn]; // en-passant victim
        return Eval.PieceValue[(int)Types.TypeOf(p)];
    }

    /// <summary>
    /// True if the colour has any piece other than pawns and the king. Null-move pruning needs this because in a king-and-pawn endgame "passing" can be the only good option (zugzwang), making the null-move assumption unsafe.
    /// </summary>
    private static bool HasNonPawnMaterial(Position pos, Color c) =>
        (pos.BitboardOf(c, PieceType.Knight) | pos.BitboardOf(c, PieceType.Bishop)
       | pos.BitboardOf(c, PieceType.Rook) | pos.BitboardOf(c, PieceType.Queen)) != 0;

    // ---- plumbing ----------------------------------------------------------------------------

    private bool OutOfTime()
    {
        if (infinite) return false;
        return clock.ElapsedMilliseconds >= deadline;
    }

    private static ushort PackMove(Move m) => (ushort)m.ToFrom;

    /// <summary>
    /// Generates all legal moves for the side to move, dispatching to the colour-specialised generator. Returns the move count.
    /// </summary>
    public static int GenerateLegal(Position pos, Span<Move> buffer) =>
        pos.Turn == Color.White
            ? pos.GenerateLegalsInto<White>(buffer)
            : pos.GenerateLegalsInto<Black>(buffer);

    private void PrintInfo(Position pos, int depth, int score)
    {
        if (Quiet) return;
        long ms = clock.ElapsedMilliseconds;
        long nps = ms > 0 ? nodes * 1000 / ms : nodes;
        string scoreStr = Math.Abs(score) >= Eval.MateBound
            ? $"mate {(score > 0 ? (Eval.MateValue - score + 1) / 2 : -(Eval.MateValue + score + 1) / 2)}"
            : $"cp {score}";
        Console.WriteLine($"info depth {depth} score {scoreStr} nodes {nodes} nps {nps} time {ms} pv {PvString(pos, depth)}");
    }

    /// <summary>
    /// Reconstruct a principal variation by walking TT best-moves from the root.
    /// </summary>
    private string PvString(Position pos, int maxLen)
    {
        var sb = new System.Text.StringBuilder();
        int played = 0;
        var line = new List<Move>();
        Span<Move> legal = stackalloc Move[256];
        for (int i = 0; i < maxLen; i++)
        {
            if (!tt.Probe(pos.GetHash(), out var e) || e.Move == 0) break;
            Move m = new(e.Move);
            int n = GenerateLegal(pos, legal);
            bool ok = false;
            for (int k = 0; k < n; k++) if (legal[k] == m) { ok = true; break; }
            if (!ok) break;
            line.Add(m);
            pos.Play(pos.Turn, m);
            played++;
            sb.Append(m).Append(' ');
        }
        for (int i = played - 1; i >= 0; i--)
            pos.Undo(pos.Turn.Flip(), line[i]);
        return sb.ToString().TrimEnd();
    }
}
