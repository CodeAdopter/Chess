using System.Globalization;
using Core;
using Engine;
using Engine.Nnue;

namespace Cli;

/// <summary>
/// The workflows the control center exposes. Each method takes already-resolved parameters and runs the
/// matching Engine operation wrapped in <see cref="Run"/>, which mirrors output to a per-run log file and
/// makes the task cancellable with Ctrl-C (the "live progress + cancel" model). One-shot arg dispatch
/// (Program.cs) and the interactive menu (<see cref="Menu"/>) both call these, so behaviour is identical
/// either way. This class is the single seam between user intent and the engine.
/// </summary>
public static class Commands
{
    // Shared defaults so the menu prompts and the one-shot args agree. Centralising them here means a
    // tweak shows up in both entry points at once and the two never drift apart.
    public const int GenGames = 500, GenDepth = 6, GenRandomPlies = 10;
    public const int TrainEpochs = 20, TrainH = 128;
    public const float TrainLr = 0.002f, TrainLambda = 0.9f;
    public const int MatchGames = 14, MatchDepth = 6;
    public const int PerftDepth = 6;

    // ---- workflows ---------------------------------------------------------------------------

    /// <summary>Generate a self-play dataset (.trd) for NNUE training. Plays many games with a few
    /// random opening plies for variety and writes out the labelled positions.</summary>
    public static void Gen(string outName, int games, int depth, int randomPlies, int seed, int threads = 0) =>
        Run("gen", ct => DataGen.Generate(outName, games, depth, randomPlies, seed, threads, ct));

    /// <summary>Self-play throughput benchmark (net teacher by default). Warmup + 3 timed runs, reports
    /// games/s and samples/s (the real training currency) plus the node breakdown. For empirically
    /// comparing search optimisations. net "-" = hand-crafted eval.</summary>
    public static void GenBench(string net, int games, int depth, int cores, int randomPlies, int seed) =>
        Run("genbench", ct =>
        {
            NnueNetwork? n = null;
            string netLabel = "hand-crafted";
            if (!string.IsNullOrEmpty(net) && net != "-" && net != "none")
            {
                n = NnueNetwork.Load(Paths.InModels(net));
                netLabel = net;
            }
            if (cores <= 0) cores = Environment.ProcessorCount;

            Console.WriteLine($"genbench: net={netLabel} depth={depth} cores={cores} games={games} randomPlies={randomPlies} seed={seed}");
            int warm = Math.Max(16, games / 8);
            Console.WriteLine($"  warmup ({warm} games)...");
            DataGen.Benchmark(warm, depth, randomPlies, seed + 99991, n, cores, ct);

            var gps = new List<double>();
            var sps = new List<double>();
            for (int run = 1; run <= 3 && !ct.IsCancellationRequested; run++)
            {
                var r = DataGen.Benchmark(games, depth, randomPlies, seed + run * 100003, n, cores, ct);
                gps.Add(r.GamesPerSec);
                sps.Add(r.SamplesPerSec);
                Console.WriteLine(
                    $"  run {run}: {r.GamesPerSec,5:F1} games/s | {r.SamplesPerSec,7:F0} samples/s | " +
                    $"{r.PliesPerGame,5:F1} plies/game | {r.NodesPerMove,6:F0} nodes/move | q {r.QFraction,4:P0} | {r.Seconds:F1}s");
            }
            gps.Sort();
            sps.Sort();
            if (gps.Count > 0)
                Console.WriteLine($"  MEDIAN: {gps[gps.Count / 2]:F1} games/s | {sps[sps.Count / 2]:F0} samples/s");
        });

    /// <summary>
    /// The first-class flow: fire-and-forget continuous self-improving training, shown on a live
    /// dashboard. Generates self-play data, trains on it, repeats; press Ctrl-C to stop. Unlike the other
    /// workflows this does not go through <see cref="Run"/> because the dashboard owns the display and
    /// streams its own log file, so cancellation is wired up here by hand.
    /// </summary>
    public static void TrainLoop(int cores, int depth = 6, int maxIterations = 0)
    {
        if (cores <= 0) cores = Math.Min(8, Environment.ProcessorCount);
        depth = Math.Clamp(depth, 4, 6);   // shallower = faster generation, slightly noisier labels
        var cfg = new LoopConfig { Cores = cores, Depth = depth, MaxIterations = maxIterations };

        // Ctrl-C cancels cooperatively: e.Cancel = true keeps the process alive so the loop can finish
        // its current step and the dashboard can shut down cleanly.
        using var cts = new CancellationTokenSource();
        void handler(object? _, ConsoleCancelEventArgs e) { e.Cancel = true; cts.Cancel(); }
        Console.CancelKeyPress += handler;

        // Timestamped log so concurrent or repeated runs never overwrite each other.
        string logPath = Path.Combine(Paths.Logs, $"train_{DateTime.Now:yyyyMMdd_HHmmss}.log");
        using var log = new StreamWriter(logPath) { AutoFlush = true };
        try { SpectreDashboard.Run(cfg, cts.Token, log); }
        finally { Console.CancelKeyPress -= handler; }
        Console.WriteLine($"\n[stopped, log: {logPath}]");
    }

    /// <summary>
    /// File-based one-shot training on a fixed dataset (experiments / manual runs).
    /// </summary>
    public static void TrainFile(string data, string netOut, int epochs, int h, float lr, float lambda, int threads = 0) =>
        Run("trainfile", ct => Trainer.Train(data, netOut, epochs, h, lr, lambda, threads, ct));

    /// <summary>Play the net against the hand-crafted evaluation to measure its real playing strength
    /// (the headline "is the net any good?" number).</summary>
    public static void Match(string net, int games, int depth) =>
        Run("match", ct => NnueTools.MatchVsHandcrafted(net, games, depth, 8, ct));

    /// <summary>A/B strength match: same net, two optimisation flag-sets head-to-head. If B scores ~50%
    /// its search is as strong as A, so the optimisations it adds don't degrade move quality (a proxy for
    /// data quality). Specs: comma list of {delta,see,rfp,qtt,lmr}, or "base"/"sound"/"all".</summary>
    public static void OptMatch(string net, int games, int depth, string aSpec, string bSpec) =>
        Run("optmatch", ct =>
        {
            var n = NnueNetwork.Load(Paths.InModels(net));
            var (ad, asee, ar, aq, al) = ParseOpts(aSpec);
            var (bd, bsee, br, bq, bl) = ParseOpts(bSpec);

            var sa = new Searcher(16) { Quiet = true }; sa.SetNnue(n); sa.SetOpts(ad, asee, ar, aq, al);
            var sb = new Searcher(16) { Quiet = true }; sb.SetNnue(n); sb.SetOpts(bd, bsee, br, bq, bl);

            var rng = new Random(12345);
            int w = 0, d = 0, l = 0;
            for (int g = 0; g < games && !ct.IsCancellationRequested; g++)
            {
                bool aWhite = (g & 1) == 0;   // alternate colours so neither side gets a colour bias
                var r = GameRunner.PlayGame(aWhite ? sa : sb, aWhite ? sb : sa, depth, rng, 10);
                if (r == GameRunner.Result.Draw) d++;
                else if ((r == GameRunner.Result.WhiteWins) == aWhite) w++;
                else l++;
                if ((g + 1) % 20 == 0) Console.WriteLine($"  {g + 1}/{games}: A +{w} ={d} -{l}");
            }
            int played = w + d + l;
            double score = played > 0 ? (w + 0.5 * d) / played : 0;
            double sigma = played > 0 ? 50.0 / Math.Sqrt(played) : 0;   // ~1σ on the score in percentage points
            Console.WriteLine($"\nA=[{aSpec}]  vs  B=[{bSpec}]   net={net}  depth={depth}  games={played}");
            Console.WriteLine($"B score (vs A): {1 - score:P1}   A: +{w} ={d} -{l}   1σ ≈ ±{sigma:F1}%");
            // Verdict: "not measurably weaker within 2σ" is the bar we want an optimisation to clear
            // before it earns its place. Anything past 2σ down is treated as a real regression.
            Console.WriteLine((1 - score) >= 0.5 - 2 * sigma / 100
                ? "  => B is NOT measurably weaker than A within 2σ (optimisations preserve move quality)."
                : "  => B looks weaker than A (beyond 2σ); the optimisations may be degrading the search/labels.");
        });

    /// <summary>Parse an optimisation spec into the five searcher flags. Accepts the named presets
    /// "base"/"none"/"off"/"-" (everything off), "sound" (delta+rfp+qtt), "all" (everything on), or a
    /// comma-separated list drawn from {delta,see,rfp,qtt,lmr}.</summary>
    private static (bool d, bool s, bool r, bool q, bool l) ParseOpts(string spec)
    {
        spec = spec.ToLowerInvariant().Trim();
        if (spec is "base" or "none" or "off" or "-") return (false, false, false, false, false);
        if (spec == "sound") return (true, false, true, true, false);
        bool All = spec == "all";
        var parts = spec.Split(',', StringSplitOptions.RemoveEmptyEntries);
        bool Has(string k) => All || Array.IndexOf(parts, k) >= 0;
        return (Has("delta"), Has("see"), Has("rfp"), Has("qtt"), Has("lmr"));
    }

    /// <summary>
    /// Play and print a single net-vs-hand-crafted game so you can eyeball how the net moves.
    /// </summary>
    public static void Show(string net, int depth) =>
        Run("show", _ => NnueTools.ShowGame(net, depth));

    /// <summary>
    /// Play and print one engine-vs-itself game at the given depth.
    /// </summary>
    public static void SelfPlay(int depth) =>
        Run("selfplay", _ => GameRunner.PlayGame(depth));

    /// <summary>Perft on one position: count the leaf nodes to a fixed depth, a correctness and speed
    /// check for move generation.</summary>
    public static void Perft(int depth, string fen) =>
        Run("perft", ct => Engine.Perft.Bench(fen, depth, ct));

    /// <summary>
    /// Run the standard perft suite, the single-core move-generation throughput benchmark (Mnps).
    /// </summary>
    public static void PerftSuite(int runs) =>
        Run("perftsuite", ct => Engine.Perft.Suite(runs, ct));

    /// <summary>
    /// Sweep thread counts to chart how perft scales across cores. <paramref name="bulk"/> times the
    /// bulk-count perft path (perft-only) instead of the emit-path generator the search uses.
    /// </summary>
    public static void PerftScale(int maxThreads, bool bulk = false) =>
        Run("perftscale", ct => Engine.Perft.Scale(maxThreads, bulk, ct));

    /// <summary>Create a fresh randomly-initialised net with accumulator width <paramref name="h"/>,
    /// the starting point for a training run.</summary>
    public static void NnueInit(string outName, int h) =>
        Run("nnueinit", _ => NnueTools.Init(outName, h));

    /// <summary>Sanity-check a net's plumbing (load, infer, and verify the accumulator path) without
    /// playing a full game.</summary>
    public static void NnueTest(string net) =>
        Run("nnuetest", _ => NnueTools.Test(net));

    /// <summary>
    /// Search throughput. With a net, compares hand-crafted vs NNUE; without, just hand-crafted.
    /// </summary>
    public static void Nps(string? net, int depth) => Run("nps", _ =>
    {
        if (!string.IsNullOrEmpty(net)) { NnueTools.Bench(net, depth); return; }
        var pos = new Position();
        Position.Set(Types.DEFAULT_FEN, pos);
        var s = new Searcher { Quiet = true };
        s.NewGame();
        s.Think(pos, SearchLimits.Depth(depth));
        Console.WriteLine($"startpos depth {depth}: {s.LastNodes:n0} nodes  {s.LastMs} ms  {Log.Nps(s.LastNodes, s.LastMs)}");
    });

    // ---- run wrapper: file logging + Ctrl-C cancellation --------------------------------------

    /// <summary>
    /// Shared harness for every workflow above. Opens a per-run log (output is mirrored to a file),
    /// installs a Ctrl-C handler that cancels cooperatively rather than killing the process, runs the
    /// body, and reports where the log landed. Centralising this gives every command the same "live
    /// progress, cancellable, logged" behaviour for free.
    /// </summary>
    private static void Run(string name, Action<CancellationToken> body)
    {
        using var rl = RunLog.Begin(name);
        using var cts = new CancellationTokenSource();
        void handler(object? _, ConsoleCancelEventArgs e)
        {
            e.Cancel = true;                    // don't kill the process, cancel cooperatively
            cts.Cancel();
            Console.WriteLine("\n[cancelling, finishing current step…]");
        }
        Console.CancelKeyPress += handler;
        // The body observes the token and throws OperationCanceledException on a clean stop, which we
        // swallow here so Ctrl-C reads as a normal exit rather than a crash.
        try { body(cts.Token); }
        catch (OperationCanceledException) { Console.WriteLine("[cancelled]"); }
        finally { Console.CancelKeyPress -= handler; }
        Console.WriteLine($"[log saved: {rl.FilePath}]");
    }

    // ---- interactive prompt helpers ----------------------------------------------------------
    // Each reads one line from the console, showing the default in brackets; a blank line (just Enter)
    // accepts that default. Used by the menu to fill in workflow parameters.

    /// <summary>
    /// Prompt for a string, returning <paramref name="def"/> if the user just presses Enter.
    /// </summary>
    public static string AskStr(string label, string def)
    {
        Console.Write($"  {label} [{def}]: ");
        string? s = Console.ReadLine();
        return string.IsNullOrWhiteSpace(s) ? def : s.Trim();
    }

    /// <summary>
    /// Prompt for an integer; falls back to <paramref name="def"/> on blank or unparseable input.
    /// </summary>
    public static int AskInt(string label, int def)
    {
        Console.Write($"  {label} [{def}]: ");
        string? s = Console.ReadLine();
        return int.TryParse(s, NumberStyles.Integer, CultureInfo.InvariantCulture, out var v) ? v : def;
    }

    /// <summary>
    /// Prompt for a float; falls back to <paramref name="def"/> on blank or unparseable input.
    /// </summary>
    public static float AskFloat(string label, float def)
    {
        Console.Write($"  {label} [{def.ToString(CultureInfo.InvariantCulture)}]: ");
        string? s = Console.ReadLine();
        return float.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out var v) ? v : def;
    }
}
