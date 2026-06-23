using System.Diagnostics;
using Core;

namespace Engine.Nnue;

/// <summary>
/// Headless commands for the NNUE path: create a net, verify the plumbing, benchmark the cost, and play it against the hand-crafted eval. These back the engine's <c>nnue*</c> CLI subcommands.
/// </summary>
public static class NnueTools
{
    /// <summary>
    /// Creates a fresh random network of width <paramref name="h"/> and saves it, printing its shape and on-disk size.
    /// </summary>
    public static void Init(string path, int h)
    {
        path = Paths.InModels(path);
        var net = NnueNetwork.CreateRandom(h);
        net.Save(path);
        long bytes = new FileInfo(path).Length;
        Console.WriteLine($"wrote {path}: H={net.H} L1Out={net.L1Out} features={FeatureSet.NumFeatures} " +
                          $"size={bytes / 1024.0 / 1024.0:F1} MB");
    }

    /// <summary>
    /// Loads a net and checks the plumbing: the evaluation is deterministic and finite on a handful of positions, then reports leaf-eval throughput.
    /// </summary>
    public static void Test(string path)
    {
        path = Paths.InModels(path);
        var net = NnueNetwork.Load(path);
        var ev = new NnueEvaluator(net);
        Console.WriteLine($"loaded {path}: H={net.H} L1Out={net.L1Out} scale={net.OutScale}");

        string[] fens =
        [
            Types.DEFAULT_FEN,
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq -",   // startpos, black to move
            Fens.Kiwipete,
            "8/8/8/4k3/8/4K3/4P3/8 w - -",                            // K+P vs K endgame
        ];

        var pos = new Position();
        bool ok = true;
        foreach (var fen in fens)
        {
            Position.Set(fen, pos);
            int correctionA = ev.Evaluate(pos);
            int correctionB = ev.Evaluate(pos);               // determinism: same position → same eval
            int full = Math.Clamp(Eval.Evaluate(pos) + correctionA, -10000, 10000);
            if (correctionA != correctionB) ok = false;
            if (correctionA is < -10000 or > 10000) ok = false; // finite / clamped
            Console.WriteLine($"  eval={full,6} cp   nnue={correctionA,6}   det={(correctionA == correctionB)}   {fen}");
        }

        // Throughput: how many leaf evaluations per second (full-refresh).
        Position.Set(Types.DEFAULT_FEN, pos);
        const int N = 200_000;
        var sw = Stopwatch.StartNew();
        long acc = 0;
        for (int i = 0; i < N; i++) acc += ev.Evaluate(pos);
        sw.Stop();
        double evps = N / sw.Elapsed.TotalSeconds;
        Console.WriteLine($"\nthroughput: {evps / 1e6:F2} M evals/sec (full refresh)   [checksum {acc}]");
        Console.WriteLine(ok ? "PLUMBING OK" : "PLUMBING FAILED");
    }

    /// <summary>
    /// Validate the int8/int16 quantisation: (1) the quantised eval must track the float eval within a few cp
    /// across many positions, and (2) the int16 incremental accumulator must match a full refresh bit-for-bit.
    /// Also reports int8 weight-clipping (a signal that the head weight scale QW is too large).
    /// </summary>
    public static void QuantTest(string path)
    {
        path = Paths.InModels(path);
        var net = NnueNetwork.Load(path);
        var qnet = new QuantizedNetwork(net);
        var fref = new NnueEvaluator(net);

        Console.WriteLine($"loaded {path}: H={net.H} L1Out={net.L1Out} scale={net.OutScale}");
        Console.WriteLine($"int8 range at QW={QuantizedNetwork.QW}: ±{127.0 / QuantizedNetwork.QW:F2}");
        Console.WriteLine($"  L1 weights: maxAbs={qnet.MaxAbsL1:F3}  clipped={qnet.L1Clipped}/{net.L1Weights.Length}");
        Console.WriteLine($"  L2 weights: maxAbs={qnet.MaxAbsL2:F3}  clipped={qnet.L2Clipped}/{net.L2Weights.Length}");

        var rng = new Random(123);
        Span<Move> buf = stackalloc Move[256];
        var qacc = new QAccumulatorStack(qnet);      // incremental
        var qfresh = new QAccumulatorStack(qnet);     // full-refresh reference (reused; RefreshRoot resets it)
        long n = 0, sumAbs = 0; int maxDiff = 0; long incMismatch = 0, simdMismatch = 0;

        for (int g = 0; g < 60; g++)
        {
            var pos = new Position();
            Position.Set(Types.DEFAULT_FEN, pos);
            qacc.RefreshRoot(pos);

            for (int ply = 0; ply < 200; ply++)
            {
                int cnt = Searcher.GenerateLegal(pos, buf);
                if (cnt == 0 || pos.IsRepetition() || pos.IsFiftyMoveRule()) break;

                int qInc = qacc.Evaluate(pos.Turn);            // incremental quant (SIMD)
                if (qInc != qacc.EvaluateScalar(pos.Turn)) simdMismatch++;   // SIMD path must equal scalar exactly
                qfresh.RefreshRoot(pos);
                if (qInc != qfresh.Evaluate(pos.Turn)) incMismatch++;   // exactness of incremental updates

                int d = Math.Abs(qInc - fref.Evaluate(pos));   // vs float reference
                maxDiff = Math.Max(maxDiff, d); sumAbs += d; n++;

                Move m = buf[rng.Next(cnt)];
                Piece moving = pos.At(m.From);
                Piece captured = m.IsCapture && m.Flags != MoveFlags.EnPassant ? pos.At(m.To) : Piece.NoPiece;
                pos.Play(pos.Turn, m);
                qacc.Push(pos, m, moving, captured);
            }
        }

        Console.WriteLine($"\nquant vs float correction over {n:n0} positions:");
        Console.WriteLine($"  mean |quant - float| = {(double)sumAbs / Math.Max(1, n):F2} cp");
        Console.WriteLine($"  max  |quant - float| = {maxDiff} cp");
        Console.WriteLine($"  incremental vs full-refresh mismatches = {incMismatch}  (must be 0, int16 is exact)");
        Console.WriteLine($"  SIMD vs scalar forward mismatches      = {simdMismatch}  (must be 0, same integer math)");
        Console.WriteLine(incMismatch == 0 && simdMismatch == 0 && maxDiff < 50 ? "QUANT OK" : "QUANT NEEDS REVIEW");
    }

    /// <summary>
    /// Searches the start position to a fixed depth twice, once with the hand-crafted eval and once with the net, so the per-node cost of the NNUE evaluation is directly comparable.
    /// </summary>
    public static void Bench(string path, int depth)
    {
        path = Paths.InModels(path);
        var net = NnueNetwork.Load(path);
        var pos = new Position();

        Console.WriteLine($"startpos, depth {depth}:");

        Position.Set(Types.DEFAULT_FEN, pos);
        var hc = new Searcher { Quiet = true };
        hc.NewGame();
        hc.Think(pos, SearchLimits.Depth(depth));
        Report("hand-crafted eval", hc.LastNodes, hc.LastMs);

        Position.Set(Types.DEFAULT_FEN, pos);
        var nn = new Searcher { Quiet = true };
        nn.SetNnue(net);
        nn.NewGame();
        nn.Think(pos, SearchLimits.Depth(depth));
        Report("hand-crafted + NNUE", nn.LastNodes, nn.LastMs);
        if (nn.VerifyMismatches > 0 || Environment.GetEnvironmentVariable("NNUE_VERIFY") == "1")
            Console.WriteLine($"  integration check: {nn.VerifyMismatches} mismatches vs full refresh");
    }

    /// <summary>
    /// Head-to-head: NNUE net vs the hand-crafted eval, alternating colours. Reports from NNUE's POV.
    /// </summary>
    public static void MatchVsHandcrafted(string netPath, int games, int depth, int randomPlies = 8,
        CancellationToken ct = default)
    {
        netPath = Paths.InModels(netPath);
        var net = NnueNetwork.Load(netPath);
        var rng = new Random(2024);
        int w = 0, d = 0, l = 0;

        for (int g = 0; g < games; g++)
        {
            if (ct.IsCancellationRequested) { Console.WriteLine("cancelled."); break; }
            var nnue = new Searcher { Quiet = true };
            nnue.SetNnue(net);
            var hc = new Searcher { Quiet = true };

            bool nnueWhite = (g & 1) == 0;
            var white = nnueWhite ? nnue : hc;
            var black = nnueWhite ? hc : nnue;

            var r = GameRunner.PlayGame(white, black, depth, rng, randomPlies);
            if (r == GameRunner.Result.Draw) d++;
            else
            {
                bool whiteWon = r == GameRunner.Result.WhiteWins;
                bool nnueWon = whiteWon == nnueWhite;
                if (nnueWon) w++; else l++;
            }
            Console.WriteLine($"  game {g + 1,3}: NNUE {(nnueWhite ? "W" : "B")}  →  NNUE +{w} ={d} -{l}");
        }

        int played = w + d + l;
        double score = played > 0 ? (w + 0.5 * d) / played : 0;
        Console.WriteLine($"\nNNUE vs hand-crafted @ depth {depth}: +{w} ={d} -{l}  (score {score:P0})");
    }

    /// <summary>
    /// Play one NNUE(White) vs hand-crafted(Black) game and print the move transcript + both evals.
    /// </summary>
    public static void ShowGame(string netPath, int depth)
    {
        netPath = Paths.InModels(netPath);
        var net = NnueNetwork.Load(netPath);
        var pos = new Position();
        Position.Set(Types.DEFAULT_FEN, pos);
        var nnue = new Searcher { Quiet = true }; nnue.SetNnue(net); nnue.NewGame();
        var hc = new Searcher { Quiet = true }; hc.NewGame();
        Span<Move> buf = stackalloc Move[256];
        var moves = new List<string>();

        for (int ply = 0; ply < 200; ply++)
        {
            int n = Searcher.GenerateLegal(pos, buf);
            if (n == 0) { Console.WriteLine(pos.InCheck(pos.Turn) ? "checkmate" : "stalemate"); break; }
            if (pos.IsRepetition() || pos.IsFiftyMoveRule()) { Console.WriteLine("draw"); break; }

            bool whiteToMove = pos.Turn == Color.White;
            var s = whiteToMove ? nnue : hc;
            Move m = s.Think(pos, SearchLimits.Depth(depth));
            if (m.ToFrom == 0) break;
            moves.Add($"{m}({s.LastScore})");
            pos.Play(pos.Turn, m);
        }

        Console.WriteLine("NNUE=White(score after its move), hand-crafted=Black:");
        for (int i = 0; i < moves.Count; i += 2)
        {
            string w = moves[i];
            string b = i + 1 < moves.Count ? moves[i + 1] : "";
            Console.WriteLine($"  {i / 2 + 1,3}. {w,-18} {b}");
        }
    }

    /// <summary>
    /// Verify the incremental <see cref="AccumulatorStack"/> matches a full <see cref="NnueEvaluator"/>
    /// refresh at every position, across many random games (random play exercises captures, castling,
    /// promotions, and en passant). Uses a random net, since only the equivalence matters, not the values.
    /// </summary>
    public static void AccTest(int games = 30, int plies = 160)
    {
        var net = NnueNetwork.CreateRandom(64, 32, 300f, seed: 5);
        var refEval = new NnueEvaluator(net);
        var rng = new Random(42);
        Span<Move> buf = stackalloc Move[256];
        int maxDiff = 0; long compared = 0;
        int caps = 0, castles = 0, promos = 0, eps = 0;

        for (int g = 0; g < games; g++)
        {
            var acc = new AccumulatorStack(net);
            var pos = new Position();
            Position.Set(Types.DEFAULT_FEN, pos);
            acc.RefreshRoot(pos);

            for (int ply = 0; ply < plies; ply++)
            {
                int n = Searcher.GenerateLegal(pos, buf);
                if (n == 0 || pos.IsRepetition() || pos.IsFiftyMoveRule()) break;

                int inc = acc.Evaluate(pos.Turn);     // incremental, current position
                int re = refEval.Evaluate(pos);       // full refresh, same position
                maxDiff = Math.Max(maxDiff, Math.Abs(inc - re));
                compared++;

                Move m = buf[rng.Next(n)];
                var f = m.Flags;
                if (f == MoveFlags.OO || f == MoveFlags.OOO) castles++;
                if ((f & MoveFlags.Promotions) != 0) promos++;
                if (f == MoveFlags.EnPassant) eps++;
                if (m.IsCapture) caps++;

                Piece moving = pos.At(m.From);
                Piece captured = m.IsCapture && f != MoveFlags.EnPassant ? pos.At(m.To) : Piece.NoPiece;
                pos.Play(pos.Turn, m);
                acc.Push(pos, m, moving, captured);
            }
        }

        Console.WriteLine($"compared {compared:n0} positions across {games} random games");
        Console.WriteLine($"coverage: captures={caps} castles={castles} promotions={promos} en-passant={eps}");
        Console.WriteLine($"max |incremental - refresh| = {maxDiff} cp");

        // Targeted special-move cases (random play hits castling/EP rarely), to exercise every variant.
        (string fen, string move, string label)[] cases =
        [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", "e1g1", "white O-O"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -", "e1c1", "white O-O-O"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq -", "e8g8", "black O-O"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq -", "e8c8", "black O-O-O"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6", "e5d6", "white en passant"),
            ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3", "e4d3", "black en passant"),
            ("k7/4P3/8/8/8/8/8/7K w - -", "e7e8q", "promotion"),
            ("3rk3/4P3/8/8/8/8/8/4K3 w - -", "e7d8q", "promotion-capture"),
        ];
        bool casesOk = true;
        Span<Move> tbuf = stackalloc Move[256];
        foreach (var (fen, uci, label) in cases)
        {
            var acc = new AccumulatorStack(net);
            var pos = new Position();
            Position.Set(fen, pos);
            acc.RefreshRoot(pos);
            int n = Searcher.GenerateLegal(pos, tbuf);
            Move m = default;
            for (int i = 0; i < n; i++) if (tbuf[i].ToString() == uci) { m = tbuf[i]; break; }
            if (m.ToFrom == 0) { Console.WriteLine($"  {label,-20} MOVE NOT LEGAL ({uci})"); casesOk = false; continue; }
            Piece moving = pos.At(m.From);
            Piece captured = m.IsCapture && m.Flags != MoveFlags.EnPassant ? pos.At(m.To) : Piece.NoPiece;
            pos.Play(pos.Turn, m);
            acc.Push(pos, m, moving, captured);
            int diff = Math.Abs(acc.Evaluate(pos.Turn) - refEval.Evaluate(pos));
            maxDiff = Math.Max(maxDiff, diff);
            if (diff > 2) casesOk = false;
            Console.WriteLine($"  {label,-20} diff={diff} cp");
        }

        Console.WriteLine(maxDiff <= 2 && casesOk ? "ACCUMULATOR OK" : "ACCUMULATOR MISMATCH");
    }

    private static void Report(string label, long nodes, long ms)
    {
        Console.WriteLine($"  {label,-22}: {nodes,10} nodes  {ms,6} ms  {Log.Nps(nodes, ms)}");
    }
}
