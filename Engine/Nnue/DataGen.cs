using System.Diagnostics;
using Core;

namespace Engine.Nnue;

/// <summary>
/// Generates self-labelled training data: self-play games (random opening plies for variety), each
/// played position recorded with the search's own score as the label, the game result back-filled at
/// the end. Games run in parallel across cores. The evaluator is the hand-crafted eval by default, or a
/// net once the loop has switched to bootstrap mode.
/// </summary>
public static class DataGen
{
    /// <summary>
    /// Generate samples in memory, parallel across <paramref name="threads"/> cores.
    /// </summary>
    public static List<Sample> GenerateSamples(int games, int depth, int randomPlies, int seed,
        NnueNetwork? net, int threads, CancellationToken ct, Action? onGameDone = null)
    {
        threads = threads <= 0 ? Environment.ProcessorCount : threads;
        var all = new List<Sample>(games * 100);
        object mergeLock = new();
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };

        Parallel.For(0, games, options,
            () => new Worker(net),
            (g, _, worker) =>
            {
                if (!ct.IsCancellationRequested)
                {
                    worker.Play(seed + g, depth, randomPlies);
                    onGameDone?.Invoke();
                }
                return worker;
            },
            worker => { lock (mergeLock) all.AddRange(worker.Samples); });

        return all;
    }

    /// <summary>
    /// File-based generation (the <c>gen</c> command): generate, then write a .trd dataset.
    /// </summary>
    public static void Generate(string outPath, int games, int depth, int randomPlies = 8,
        int seed = 1234, int threads = 0, CancellationToken ct = default)
    {
        outPath = Paths.InData(outPath);
        Directory.CreateDirectory(Path.GetDirectoryName(outPath)!);

        var clock = Stopwatch.StartNew();
        int[] done = [0];
        var samples = GenerateSamples(games, depth, randomPlies, seed, null, threads, ct, () =>
        {
            int d = System.Threading.Interlocked.Increment(ref done[0]);
            if (d % 50 == 0 || d == games) Console.WriteLine($"  {d}/{games} games  ({samplesPerSec(d)})");
        });

        using (var writer = new TrainingData.Writer(outPath))
            foreach (var s in samples) writer.Add(s.Fen, s.ScoreCp, s.Result);

        Console.WriteLine($"wrote {outPath}");
        Console.WriteLine($"  {samples.Count} samples in {clock.Elapsed.TotalSeconds:F1}s");

        string samplesPerSec(int gamesDone) =>
            $"{gamesDone / Math.Max(0.001, clock.Elapsed.TotalSeconds):F0} games/s";
    }

    /// <summary>
    /// Aggregate throughput stats from a benchmark run (no samples kept, pure measurement).
    /// </summary>
    public readonly record struct BenchResult(
        int Games, double Seconds, long Samples, long SearchPlies, long Nodes, long Interior, long QNodes)
    {
        public double GamesPerSec => Games / Math.Max(1e-9, Seconds);
        public double SamplesPerSec => Samples / Math.Max(1e-9, Seconds);
        public double PliesPerGame => (double)SearchPlies / Math.Max(1, Games);
        public double NodesPerMove => (double)Nodes / Math.Max(1, SearchPlies);
        public double QFraction => (double)QNodes / Math.Max(1, Nodes);
    }

    /// <summary>
    /// Throughput benchmark: same self-play path as <see cref="GenerateSamples"/> but discards the samples
    /// and aggregates timing + node counts. Used to empirically compare search optimisations.
    /// </summary>
    public static BenchResult Benchmark(int games, int depth, int randomPlies, int seed,
        NnueNetwork? net, int threads, CancellationToken ct)
    {
        threads = threads <= 0 ? Environment.ProcessorCount : threads;
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        long samples = 0, plies = 0, nodes = 0, interior = 0, q = 0;
        object mergeLock = new();

        var clock = Stopwatch.StartNew();
        Parallel.For(0, games, options,
            () => new Worker(net),
            (g, _, worker) =>
            {
                if (!ct.IsCancellationRequested) worker.Play(seed + g, depth, randomPlies);
                return worker;
            },
            worker =>
            {
                lock (mergeLock)
                {
                    samples += worker.Samples.Count;
                    plies += worker.SearchPlies;
                    nodes += worker.Nodes;
                    interior += worker.Interior;
                    q += worker.QNodes;
                }
            });
        clock.Stop();

        return new BenchResult(games, clock.Elapsed.TotalSeconds, samples, plies, nodes, interior, q);
    }

    /// <summary>
    /// One self-play worker: a thread-local searcher and the samples (and benchmark counters) it has accumulated. One per parallel thread so nothing is shared.
    /// </summary>
    private sealed class Worker
    {
        private readonly Searcher searcher;
        public readonly List<Sample> Samples = [];
        public long Nodes;
        public long Interior;     // benchmark counters: accumulated across all games this worker plays
        public long QNodes;
        public long SearchPlies;  // number of searched (non-random) plies

        public Worker(NnueNetwork? net)
        {
            searcher = new Searcher(ttSizeMb: 16) { Quiet = true };  // small TT, since self-play searches are shallow
            if (net != null) searcher.SetNnue(net);
        }

        /// <summary>
        /// Plays one game from the start position (with a random opening prefix) and records the labelled, non-tactical positions it passes through.
        /// </summary>
        public void Play(int seed, int depth, int randomPlies)
        {
            var rng = new Random(seed);
            var pos = new Position();
            Position.Set(Types.DEFAULT_FEN, pos);
            searcher.NewGame(clearTt: false);   // key-verified TT, so no need to memset 16 MB per game
            Span<Move> buf = stackalloc Move[256];

            var pending = new List<(string fen, int score, bool stmWhite)>();
            int whiteResult = 0;

            // Vary the random-opening length per game in [2, 2*randomPlies]. Short prefixes record near-opening
            // positions (so the net learns opening/early-game play), long prefixes inject the material
            // imbalances that teach piece values. A fixed length does one or the other, never both: too short
            // and games stay materially balanced (the net can't learn material); too long and the opening is
            // never in the data (the net plays it out-of-distribution and drifts into lost positions).
            int gameRandomPlies = 2 + rng.Next(Math.Max(1, 2 * randomPlies - 1));

            for (int ply = 0; ply < 400; ply++)
            {
                int n = Searcher.GenerateLegal(pos, buf);
                if (n == 0)
                {
                    if (pos.InCheck(pos.Turn)) whiteResult = pos.Turn == Color.White ? -1 : 1;
                    break;
                }
                if (pos.IsRepetition() || pos.IsFiftyMoveRule()) break;

                Move m;
                if (ply < gameRandomPlies)
                {
                    m = buf[rng.Next(n)];
                }
                else
                {
                    m = searcher.Think(pos, SearchLimits.Depth(depth));
                    Nodes += searcher.LastNodes;
                    Interior += searcher.LastInteriorNodes;
                    QNodes += searcher.LastQNodes;
                    SearchPlies++;
                    if (m.ToFrom == 0) break;
                    // Quiet-position filter: only label positions where the static eval is meaningful.
                    // Skip in-check positions and ones whose best move is a capture/promotion, because there
                    // the value is decided by a tactic, not by the standing assessment the net is learning, so
                    // the label is noise that flattens the eval. Standard NNUE data hygiene.
                    bool tactical = pos.InCheck(pos.Turn) || m.IsCapture || (m.Flags & MoveFlags.Promotions) != 0;
                    if (!tactical)
                        pending.Add((pos.Fen(), searcher.LastScore, pos.Turn == Color.White));
                }
                pos.Play(pos.Turn, m);
            }

            foreach (var (fen, score, stmWhite) in pending)
            {
                int resultStm = stmWhite ? whiteResult : -whiteResult;
                Samples.Add(new Sample(fen, (short)Math.Clamp(score, -10000, 10000), (sbyte)resultStm));
            }
        }
    }
}
