using System.Diagnostics;
using Core;
using Engine.Nnue;

namespace Engine;

public struct SearchLimits
{
    public int MaxDepth;     // hard depth cap
    public long MoveTimeMs;  // 0 => no explicit move-time budget
    public long MaxNodes;    // 0 => no explicit node budget
    public bool Infinite;    // search until told to stop

    public static SearchLimits Depth(int d) => new() { MaxDepth = d, Infinite = false };
    public static SearchLimits Time(long ms) => new() { MoveTimeMs = ms, Infinite = false };
    public static SearchLimits Nodes(long n) => new() { MaxNodes = n, Infinite = false };
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
    private bool optDelta = Environment.GetEnvironmentVariable("OPT_DELTA") != "0";                     // quiescence delta pruning
    private bool optSee   = Environment.GetEnvironmentVariable("OPT_SEE")   == "1";                     // skip losing captures (SEE<0) in qsearch
    private bool optRfp   = Environment.GetEnvironmentVariable("OPT_RFP")   != "0";                     // reverse futility / static null move
    private bool optQtt   = Environment.GetEnvironmentVariable("OPT_QTT")   != "0";                     // quiescence TT probe/store cutoffs
    private bool optLmr   = Environment.GetEnvironmentVariable("OPT_LMR")   == "1";                     // late move reductions
    private bool optMainSee = Environment.GetEnvironmentVariable("OPT_MAIN_SEE") == "1";                // shallow losing-capture pruning
    private bool optBadHist = Environment.GetEnvironmentVariable("OPT_BADHIST") == "1";                 // conservative bad-history pruning
    private bool optImproving = Environment.GetEnvironmentVariable("OPT_IMPROVING") != "0";             // improving-aware RFP/LMP/futility/LMR margins
    private bool optContHist  = Environment.GetEnvironmentVariable("OPT_CONTHIST")  != "0";             // continuation history + malus-decay quiet-history update
    private bool optIir = Environment.GetEnvironmentVariable("OPT_IIR") != "0";                         // internal iterative reduction (promoted: depth-per-time win, like LMR; OPT_IIR=0 disables)
    private static readonly bool OptIncEval = Environment.GetEnvironmentVariable("OPT_INCEVAL") != "0"; // bit-exact residual baseline cache

    private bool optModern = Environment.GetEnvironmentVariable("OPT_MODERN") != "0";

    /// <summary>Override the per-search optimisation flags (for A/B strength matches). INCEVAL is excluded
    /// because it is bit-exact and never affects move choice, so it stays globally configured.</summary>
    public void SetOpts(bool delta, bool see, bool rfp, bool qtt, bool lmr)
    { optDelta = delta; optSee = see; optRfp = rfp; optQtt = qtt; optLmr = lmr; }

    public void SetModern(bool on) => optModern = on;

    public void SetHistory(bool improving, bool contHist)
    { optImproving = improving; optContHist = contHist; }

    public void SetIir(bool on) => optIir = on;

    private bool UseLmr => optModern || optLmr;

    private const int DeltaMargin  = 200;               // cp slack for quiescence delta pruning
    private const int RfpMaxDepth  = 3;                 // legacy RFP depth cap
    private const int RfpMaxDepthModern = 6;            // depth cap for reverse futility pruning
    private const int RfpMargin    = 80;                // cp per ply for RFP
    private const int SeeKingValue = 10000;             // king never voluntarily recaptures into a defended square
    private const int ExtBudget    = 16;                // max plies a root-to-leaf path may gain from check extensions
    private const int HistReduceBar = 4000;             // quiet-move history above which LMR reduces one ply less
    private const int HistoryMax = 16384;               // bounded history saturation point
    private const int BadHistoryMax = 4096;             // max score clamp bad move
    private const int BadHistPruneDepth = 4;            // max depth bad history pruning is allowed
    private const int BadHistPruneThreshold = -512;     // bad move score below -512 pruned
    private const int MainSeePruneDepth = 5;            
    private const int MainSeeMargin = 80;               // cp per remaining ply
    private const int MoveKeyCount = 64 * 64;           // table size
    private const int NoStaticEval = int.MinValue / 2;
    // Late-move pruning
    private static readonly int[] LmpCount = [0, 5, 8, 12, 18, 26, 36];
    // At depth 1-2, a quiet move whose optimistic static eval can't reach alpha is skipped. 
    private static readonly int[] FutMargin = [0, 120, 200];

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
    private readonly short[][] badHistory = [new short[MoveKeyCount], new short[MoveKeyCount]];
    private readonly Move[,] killers = new Move[MaxPly, 2];
    private readonly Move[] counterMoves = new Move[64 * 64];
    private readonly int[] evalStack = new int[MaxPly];
    private readonly short[][] continuationHistory = new short[MoveKeyCount][]; // sparse [previous from-to][current from-to]

    /// <summary>
    /// Resize the transposition table
    /// </summary>
    public void SetHashSize(int mb) => tt.Resize(mb);

    private long nodes;
    private long deadline;
    private long maxNodes;   // 0 => no budget
    private int rootDepth;
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
    public int LastDepth { get; private set; }   // deepest fully-completed iteration of the last search

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
        int hc = OptIncEval ? acc.EvaluateHce(pos.Turn) : Eval.Evaluate(pos);
        if (verify != null && (correction != verify.Evaluate(pos) || hc != Eval.Evaluate(pos))) VerifyMismatches++;
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
        Array.Clear(badHistory[0]);
        Array.Clear(badHistory[1]);
        Array.Clear(counterMoves);
        for (int i = 0; i < continuationHistory.Length; i++)
            if (continuationHistory[i] != null) Array.Clear(continuationHistory[i]);
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
        maxNodes = limits.MaxNodes;

        Array.Clear(killers, 0, killers.Length);

        Span<Move> rootMoves = stackalloc Move[256];
        int rootCount = GenerateLegal(pos, rootMoves);
        if (rootCount == 0) return default;            // mate/stalemate at root
        pvMove = rootMoves[0];                          // always have a legal fallback

        acc?.RefreshRoot(pos);   // sync the incremental accumulator to the root once per search

        int maxDepth = limits.MaxDepth > 0 ? limits.MaxDepth : MaxPly - 1;
        int lastScore = 0;
        int completedDepth = 0;

        for (int depth = 1; depth <= maxDepth; depth++)
        {
            int score = AspirationSearch(pos, depth, lastScore);

            if (stop && depth > 1) break;               // discard partial iteration's score

            lastScore = score;
            completedDepth = depth;
            PrintInfo(pos, depth, score);

            if (Math.Abs(score) >= Eval.MateBound) break; // forced mate found
            if (OutOfTime()) break;
        }

        LastNodes = nodes;
        LastInteriorNodes = interiorNodes;
        LastQNodes = qNodes;
        LastMs = clock.ElapsedMilliseconds;
        LastScore = lastScore;
        LastDepth = completedDepth;
        return pvMove;
    }

    private const int AspMinDepth = 4;
    private const int AspInitDelta = 16;

    private int AspirationSearch(Position pos, int depth, int prevScore)
    {
        rootDepth = depth;
        if (!optModern || depth < AspMinDepth || Math.Abs(prevScore) >= Eval.MateBound)
            return Negamax(pos, depth, -Inf, Inf, 0, default);

        int delta = AspInitDelta;
        int alpha = Math.Max(prevScore - delta, -Inf);
        int beta  = Math.Min(prevScore + delta, Inf);
        while (true)
        {
            int score = Negamax(pos, depth, alpha, beta, 0, default);
            if (stop) return score;
            if (score <= alpha)                                 // fail low: drop alpha, nudge beta toward the score
            {
                beta = (alpha + beta) / 2;
                alpha = Math.Max(score - delta, -Inf);
            }
            else if (score >= beta)                             // fail high: raise beta
            {
                beta = Math.Min(score + delta, Inf);
            }
            else
                return score;                                   // score strictly inside the window: accept it
            delta += delta / 2 + 8;                             // widen geometrically each failure
            if (delta > 2000) { alpha = -Inf; beta = Inf; }     // give up narrowing; search the full width
        }
    }

    /// <summary>
    /// The core negamax search with alpha-beta pruning and principal-variation search (PVS). Returns the
    /// score (centipawns, side-to-move POV) of the best line from <paramref name="pos"/> to the given
    /// <paramref name="depth"/>. Along the way it consults and updates the transposition table, applies
    /// reverse-futility and null-move pruning, orders moves, and (optionally) reduces late quiet moves (LMR).
    /// At depth 0 it hands off to <see cref="Quiesce"/>. The window is <paramref name="alpha"/>..<paramref
    /// name="beta"/>; <paramref name="ply"/> is the distance from the root (used for mate scoring)
    /// </summary>
    private int Negamax(Position pos, int depth, int alpha, int beta, int ply, Move prevMove)
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

        if (optIir && optModern && !root && depth >= 4 && ttMove.ToFrom == 0 && !inCheck)
            depth--;

        int rfpCap = optModern ? RfpMaxDepthModern : RfpMaxDepth;
        bool haveStaticEval = !inCheck && depth <= rfpCap && (optRfp || optModern);
        int staticEval = haveStaticEval ? EvalPos(pos) : 0;
        if (ply < MaxPly)
            evalStack[ply] = haveStaticEval ? staticEval : NoStaticEval;
        bool improving = optImproving && optModern && haveStaticEval && ply >= 2
            && evalStack[ply - 2] != NoStaticEval && staticEval > evalStack[ply - 2];

        // Reverse futility pruning (OPT_RFP / static null move): at shallow non-root, non-check nodes, if
        // the static eval already clears beta by a depth-scaled margin, assume it holds and fail high
        // without searching, skipping the entire quiescence subtree the node would otherwise spawn.
        if (optRfp && !root && !inCheck && depth <= rfpCap
            && beta < Eval.MateBound && beta > -Eval.MateBound)
        {
            int margin = (RfpMargin + (improving ? 20 : (optImproving ? -10 : 0))) * depth;
            if (staticEval - margin >= beta)
                return staticEval - margin;
        }

        // Null-move pruning: give the opponent a free move; if we're still winning, prune. Skipped in
        // check and in likely-zugzwang (no non-pawn material), where passing isn't safely equivalent.
        if (!root && !inCheck && depth >= 3 && beta < Eval.MateBound
            && HasNonPawnMaterial(pos, pos.Turn))
        {
            int r = 2 + depth / 6;
            pos.MakeNullMove(); acc?.PushNull();
            int nullScore = -Negamax(pos, depth - 1 - r, -beta, -beta + 1, ply + 1, default);
            pos.UnmakeNullMove(); acc?.Pop();
            if (stop) return 0;
            if (nullScore >= beta) return beta;
        }

        Span<Move> moves = stackalloc Move[256];
        int count = GenerateLegal(pos, moves);

        if (count == 0)
            return inCheck ? -Eval.MateValue + ply : 0;  // checkmate (ply-adjusted) or stalemate

        Span<int> scores = stackalloc int[count];
        ScoreMoves(pos, moves, count, scores, ttMove, ply, prevMove);

        int bestScore = -Inf;
        Move bestMove = default;
        int origAlpha = alpha;
        Color us = pos.Turn;
        bool pvNode = beta - alpha > 1;   // a real PV window (not a null-window scout): reduce/prune less here
        int prevKey = prevMove.ToFrom != 0 ? ((int)prevMove.From << 6) | (int)prevMove.To : -1;
        bool trackQuiets = optModern && (optContHist || optBadHist);
        Span<Move> quietsTried = stackalloc Move[trackQuiets ? 64 : 0];
        int quietsTriedCount = 0;

        for (int i = 0; i < count; i++)
        {
            PickNext(moves, scores, count, i);
            Move m = moves[i];
            bool quiet = !m.IsCapture && (m.Flags & MoveFlags.Promotions) == 0;
            int quietHist = quiet ? QuietHistoryScore(us, m, prevKey) : 0;

            if (optModern && !pvNode && !inCheck && quiet && bestScore > -Eval.MateBound
                && m != killers[ply, 0] && m != killers[ply, 1])
            {
                if (depth <= 6 && i >= LmpCount[depth] + (improving ? 2 : (optImproving ? -1 : 0))) break;
                if (depth <= 2 && staticEval + FutMargin[depth] + (improving ? 40 : (optImproving ? -30 : 0)) <= alpha) break;
                if (!optContHist && optBadHist && depth <= BadHistPruneDepth && i >= 6
                    && badHistory[(int)us][MoveKey(m)] < BadHistPruneThreshold + depth * 64)
                    continue;
            }

            if (optModern && optMainSee && !root && !pvNode && !inCheck && !quiet && bestScore > -Eval.MateBound
                && depth <= MainSeePruneDepth && m != ttMove
                && m.IsCapture && (m.Flags & MoveFlags.Promotions) == 0
                && PieceTypeValue(pos.At(m.From)) > PieceTypeValue(pos.At(m.To))
                && See(pos, m) < -MainSeeMargin * depth)
                continue;

            if (trackQuiets && quiet && quietsTriedCount < quietsTried.Length)
                quietsTried[quietsTriedCount++] = m;

            PlayMove(pos, us, m);
            bool givesCheck = pos.InCheck(pos.Turn);

            int ext = optModern && givesCheck && ply < rootDepth + ExtBudget ? 1 : 0;
            int newDepth = depth - 1 + ext;

            int score;
            if (i == 0)
            {
                score = -Negamax(pos, newDepth, -beta, -alpha, ply + 1, m);
            }
            else
            {
                // PVS + LMR. Reduce late, quiet, non-PV, non-checking moves; the reduced search uses a
                // null window. If it beats alpha we re-search at full depth (still null window) before the
                // standard PVS full-window widen, so a genuinely good reduced move is always verified.
                int searchDepth = newDepth;
                if (UseLmr && depth >= 3 && i >= (pvNode ? 2 : 1) && !inCheck && ext == 0
                    && quiet && m != ttMove && m != killers[ply, 0] && m != killers[ply, 1])
                {
                    int rr = LmrTable[Math.Min(depth, MaxPly - 1), Math.Min(i, 63)];
                    if (pvNode) rr--;   // PV moves are more likely to matter: reduce one ply less
                    if (improving) rr--;
                    else if (optImproving && optModern && haveStaticEval) rr++;
                    if (quietHist > HistReduceBar) rr--;  // historically good
                    if (rr < 1) rr = 1;
                    searchDepth = Math.Max(1, newDepth - rr);
                }

                score = -Negamax(pos, searchDepth, -alpha - 1, -alpha, ply + 1, m);
                if (score > alpha && searchDepth < newDepth)
                    score = -Negamax(pos, newDepth, -alpha - 1, -alpha, ply + 1, m);
                if (score > alpha && score < beta)
                    score = -Negamax(pos, newDepth, -beta, -alpha, ply + 1, m);
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
                // Beta cutoff. Reward quiet cutoff moves so they're tried earlier next time
                if (quiet)
                {
                    if (killers[ply, 0] != m) { killers[ply, 1] = killers[ply, 0]; killers[ply, 0] = m; }
                    if (optContHist && optModern)
                        UpdateQuietHistories(us, m, prevKey, depth, quietsTried[..quietsTriedCount]);
                    else
                    {
                        history[(int)us][MoveKey(m)] += depth * depth;
                        if (optModern && optBadHist)
                            UpdateBadHistory(us, m, depth, quietsTried[..quietsTriedCount]);
                    }
                    if (prevKey >= 0) counterMoves[prevKey] = m;
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
        int count = inCheck ? GenerateLegal(pos, moves) : GenerateNoisyLegal(pos, moves);

        if (count == 0)
            return inCheck ? -Eval.MateValue + ply : alpha;

        Span<int> scores = stackalloc int[count];
        ScoreMoves(pos, moves, count, scores, qttMove, ply, default);

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
            if (optSee && !inCheck && m.IsCapture
                && PieceTypeValue(pos.At(m.From)) > PieceTypeValue(pos.At(m.To))
                && See(pos, m) < 0)
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
    private void ScoreMoves(Position pos, Span<Move> moves, int count, Span<int> scores, Move ttMove, int ply, Move prevMove)
    {
        Color us = pos.Turn;
        int prevKey = optModern && prevMove.ToFrom != 0 ? ((int)prevMove.From << 6) | (int)prevMove.To : -1;
        Move counter = prevKey >= 0 ? counterMoves[prevKey] : default;
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
                int mvv = victim * 16 - attacker + promo;
                bool losing = optModern && m.IsCapture && (m.Flags & MoveFlags.Promotions) == 0
                    && attacker > victim && See(pos, m) < 0;
                scores[i] = losing ? -1_000_000 + mvv : 1_000_000 + mvv;
            }
            else if (m == killers[ply, 0] || m == killers[ply, 1])
            {
                scores[i] = 900_000;
            }
            else if (counter.ToFrom != 0 && m == counter)
            {
                scores[i] = 850_000;
            }
            else
            {
                scores[i] = QuietHistoryScore(us, m, prevKey);
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

    private static int MoveKey(Move m) => ((int)m.From << 6) | (int)m.To;

    private int QuietHistoryScore(Color us, Move m, int prevKey)
    {
        int key = MoveKey(m);
        int score = history[(int)us][key];
        if (optContHist && optModern && prevKey >= 0 && continuationHistory[prevKey] is { } row)
            score += row[key];
        return score;
    }

    private void UpdateQuietHistories(Color us, Move cutoff, int prevKey, int depth, Span<Move> quietsTried)
    {
        int bonus = Math.Min(HistoryMax, depth * depth);
        int malus = -bonus;
        int cutoffKey = MoveKey(cutoff);

        UpdateHistory(history[(int)us], cutoffKey, bonus);
        if (optModern && prevKey >= 0)
            UpdateContinuationHistory(prevKey, cutoffKey, bonus);

        for (int i = 0; i < quietsTried.Length; i++)
        {
            Move m = quietsTried[i];
            if (m == cutoff) continue;

            int key = MoveKey(m);
            UpdateHistory(history[(int)us], key, malus);
            if (optModern && prevKey >= 0)
                UpdateContinuationHistory(prevKey, key, malus);
        }
    }

    private static void UpdateHistory(int[] table, int key, int bonus)
    {
        bonus = Math.Clamp(bonus, -HistoryMax, HistoryMax);
        table[key] += bonus - table[key] * Math.Abs(bonus) / HistoryMax;
    }

    private void UpdateBadHistory(Color us, Move cutoff, int depth, Span<Move> quietsTried)
    {
        short[] table = badHistory[(int)us];
        int bonus = Math.Min(BadHistoryMax, 8 * depth * depth);
        int malus = -bonus / 2;

        UpdateShortHistory(table, MoveKey(cutoff), bonus);
        for (int i = 0; i < quietsTried.Length; i++)
        {
            Move m = quietsTried[i];
            if (m == cutoff) continue;
            UpdateShortHistory(table, MoveKey(m), malus);
        }
    }

    private static void UpdateShortHistory(short[] table, int key, int bonus)
    {
        bonus = Math.Clamp(bonus, -BadHistoryMax, BadHistoryMax);
        int value = table[key];
        value += bonus - value * Math.Abs(bonus) / BadHistoryMax;
        table[key] = (short)Math.Clamp(value, -BadHistoryMax, BadHistoryMax);
    }

    private void UpdateContinuationHistory(int prevKey, int key, int bonus)
    {
        short[] row = continuationHistory[prevKey] ??= new short[MoveKeyCount];
        int value = row[key];
        bonus = Math.Clamp(bonus, -HistoryMax, HistoryMax);
        value += bonus - value * Math.Abs(bonus) / HistoryMax;
        row[key] = (short)Math.Clamp(value, -HistoryMax, HistoryMax);
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
        if (maxNodes > 0 && nodes >= maxNodes) return true;
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

    public static int GenerateNoisyLegal(Position pos, Span<Move> buffer) =>
        pos.Turn == Color.White
            ? pos.GenerateNoisyLegalsInto<White>(buffer)
            : pos.GenerateNoisyLegalsInto<Black>(buffer);

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
