using System.Diagnostics;
using System.Runtime.CompilerServices;
using Core;

namespace Engine;

/// <summary>
/// Move-generation performance test. Perft counts the legal-move tree to a given depth; dividing by
/// elapsed time gives the move generator's raw throughput (the same nodes/sec figure the search and
/// self-play report, via <see cref="Log.Nps"/>).
/// </summary>
public static class Perft
{
    // The per-node move buffer is fully written by the generator and only moves[0..n) are ever read, so
    // the default zero-init of stackalloc is wasted work (a 512-byte memset on every node). Skip it.
    /// <summary>
    /// Straightforward recursive perft: the number of legal positions reachable in exactly <paramref name="depth"/> plies. Uses the hashed make/unmake and a fresh per-node move buffer.
    /// </summary>
    [SkipLocalsInit]
    public static long Count(Position pos, int depth)
    {
        Span<Move> moves = stackalloc Move[256];
        int n = Searcher.GenerateLegal(pos, moves);
        if (depth <= 1) return n;

        long nodes = 0;
        Color us = pos.Turn;
        for (int i = 0; i < n; i++)
        {
            pos.Play(us, moves[i]);
            nodes += Count(pos, depth - 1);
            pos.Undo(us, moves[i]);
        }
        return nodes;
    }

    /// <summary>
    /// Faster perft driver used by the throughput benchmarks. A colour-templated pair of mutually-recursive
    /// helpers (<c>PerftW</c>/<c>PerftB</c>) lets each node call the colour-specialised generator directly,
    /// and one shared buffer is sliced per ply instead of re-stackallocating at every node. Same counts as
    /// <see cref="Count"/>, but uses the hash-free make/unmake since perft never needs the Zobrist hash.
    /// </summary>
    [SkipLocalsInit]
    public static long CountFast(Position pos, int depth)
    {
        Span<Move> buf = stackalloc Move[256 * depth];
        return pos.Turn == Color.White ? PerftW(pos, depth, buf) : PerftB(pos, depth, buf);
    }

    private static long PerftW(Position p, int depth, Span<Move> buf)
    {
        int n = p.GenerateLegalsInto<White>(buf);
        if (depth <= 1) return n;
        long nodes = 0;
        Span<Move> sub = buf[256..];
        for (int i = 0; i < n; i++)
        {
            p.PlayPerft(Color.White, buf[i]);
            nodes += PerftB(p, depth - 1, sub);
            p.UndoPerft(Color.White, buf[i]);
        }
        return nodes;
    }

    private static long PerftB(Position p, int depth, Span<Move> buf)
    {
        int n = p.GenerateLegalsInto<Black>(buf);
        if (depth <= 1) return n;
        long nodes = 0;
        Span<Move> sub = buf[256..];
        for (int i = 0; i < n; i++)
        {
            p.PlayPerft(Color.Black, buf[i]);
            nodes += PerftW(p, depth - 1, sub);
            p.UndoPerft(Color.Black, buf[i]);
        }
        return nodes;
    }

    /// <summary>
    /// Bulk-counting perft driver. Like <see cref="CountFast(Position, int)"/>, but at the last ply it
    /// popcounts the legal-move bitboards instead of materialising and recursing into them, so it is faster
    /// than the emit path the search actually uses. Perft-only (the search needs real moves), and reported
    /// separately in the README. Benchmark it across threads with <c>chess perftbulkscale</c>.
    /// </summary>
    [SkipLocalsInit]
    public static long CountBulk(Position pos, int depth)
    {
        Span<Move> buf = stackalloc Move[256 * depth];
        return pos.Turn == Color.White ? BulkW(pos, depth, buf) : BulkB(pos, depth, buf);
    }

    private static long BulkW(Position p, int depth, Span<Move> buf)
    {
        if (depth == 1) return p.CountLegals<White>();
        int n = p.GenerateLegalsInto<White>(buf);
        long nodes = 0;
        Span<Move> sub = buf[256..];
        for (int i = 0; i < n; i++)
        {
            p.PlayPerft(Color.White, buf[i]);
            nodes += BulkB(p, depth - 1, sub);
            p.UndoPerft(Color.White, buf[i]);
        }
        return nodes;
    }

    private static long BulkB(Position p, int depth, Span<Move> buf)
    {
        if (depth == 1) return p.CountLegals<Black>();
        int n = p.GenerateLegalsInto<Black>(buf);
        long nodes = 0;
        Span<Move> sub = buf[256..];
        for (int i = 0; i < n; i++)
        {
            p.PlayPerft(Color.Black, buf[i]);
            nodes += BulkW(p, depth - 1, sub);
            p.UndoPerft(Color.Black, buf[i]);
        }
        return nodes;
    }

    /// <summary>
    /// Multi-threaded perft. Perft is embarrassingly parallel: the subtrees under each move are
    /// independent, so we enumerate every move sequence of the first couple of plies, clone the board at
    /// each of those frontier nodes (Position is mutable, so each thread needs its own), and farm the
    /// independent subtree counts across <paramref name="threads"/> workers. Splitting two plies yields
    /// hundreds to thousands of jobs, enough for the runtime's dynamic partitioner to balance load even at
    /// high core counts. Same node count as the single-threaded drivers.
    /// </summary>
    public static long CountParallel(Position root, int depth, int threads, bool bulk = false)
    {
        if (threads <= 1 || depth <= 2) return bulk ? CountBulk(root, depth) : CountFast(root, depth);

        int splitPlies = depth >= 4 ? 2 : 1;   // perft(2) frontier jobs for any real depth
        int jobDepth = depth - splitPlies;
        var jobs = new List<Position>();
        BuildJobs(root, splitPlies, jobs);

        long total = 0;
        object gate = new();
        Parallel.ForEach(
            jobs,
            new ParallelOptions { MaxDegreeOfParallelism = threads },
            () => 0L,
            (job, _, local) => local + (bulk ? CountBulk(job, jobDepth) : CountFast(job, jobDepth)),
            local => { lock (gate) total += local; });
        return total;
    }

    // Enumerate all move sequences `splitPlies` deep and snapshot the board at each frontier node. Uses the
    // hashed Play/Undo (only a couple of plies, negligible) and clones via the copy constructor.
    private static void BuildJobs(Position p, int splitPlies, List<Position> jobs)
    {
        if (splitPlies == 0) { jobs.Add(new Position(p)); return; }
        Span<Move> moves = stackalloc Move[256];
        int n = Searcher.GenerateLegal(p, moves);
        Color us = p.Turn;
        for (int i = 0; i < n; i++)
        {
            p.Play(us, moves[i]);
            BuildJobs(p, splitPlies - 1, jobs);
            p.Undo(us, moves[i]);
        }
    }



    /// <summary> Thread-scaling sweep: runs one large perft at increasing thread counts and reports throughput
    /// and speedup vs single core, so multi-core movegen scaling is visible at a glance. With
    /// <paramref name="bulk"/> set, it times the bulk-count perft driver (leaves popcounted, perft-only)
    /// instead of the emit-path generator the search uses.
    /// </summary>
    public static void Scale(int maxThreads = 0, bool bulk = false, CancellationToken ct = default)
    {
        // Kiwipete depth 6 is ~8.0G nodes, big enough that even 32 threads run long enough to time cleanly.
        const string fen = Fens.Kiwipete;
        const int depth = 6;
        const long expected = 8_031_647_685;

        int cap = maxThreads > 0 ? maxThreads : Environment.ProcessorCount;
        var root = new Position();
        Position.Set(fen, root);

        // Thread-count ladder (filtered to the cap), with intermediate points so the physical-core →
        // hyperthread → efficiency-core transitions on hybrid CPUs are visible.
        var ladder = new List<int>();
        foreach (int t in new[] { 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64 })
            if (t <= cap) ladder.Add(t);
        if (ladder.Count == 0 || ladder[^1] != cap) ladder.Add(cap);

        CountParallel(root, 4, cap, bulk);   // warmup

        string label = bulk ? "bulk-count (perft-only)" : "move-gen (emit path, search uses this)";
        Console.WriteLine($"perft scaling [{label}]  {fen}  depth {depth}  ({expected:n0} nodes)  [{Environment.ProcessorCount} logical cores]");
        double baseMnps = 0;
        foreach (int t in ladder)
        {
            if (ct.IsCancellationRequested) { Console.WriteLine("cancelled."); return; }
            long best = long.MaxValue, nodes = 0;
            int reps = 2;                  // best-of-2 so the single-core baseline is a fair floor too
            for (int r = 0; r < reps; r++)
            {
                var sw = Stopwatch.StartNew();
                nodes = CountParallel(root, depth, t, bulk);
                sw.Stop();
                if (sw.ElapsedMilliseconds < best) best = sw.ElapsedMilliseconds;
            }
            double mnps = nodes * 1000.0 / best / 1e6;
            if (t == 1) baseMnps = mnps;
            string ok = nodes == expected ? "ok  " : "FAIL";
            Console.WriteLine($"  {ok} {t,3} thread(s): {best,7} ms  {mnps,8:F1} Mnps  {mnps / baseMnps,5:F1}x");
        }
    }

    public readonly record struct SuiteResult(string Name, string Fen, int Depth, long Nodes, long Ms, bool Ok);

    private static readonly (string name, string fen, int depth, long expected)[] SuitePositions =
    [
        ("Startpos",   Fens.Startpos,   6, 119_060_324),
        ("Kiwipete",   Fens.Kiwipete,   5, 193_690_690),
        ("Endgame",    Fens.Endgame,    6, 11_030_083),
        ("Tactical",   Fens.Tactical,   5, 15_833_292),
        ("Promotions", Fens.Promotions, 5, 89_941_194),
        ("Midgame",    Fens.Midgame,    5, 164_075_551),
    ];


    public static List<SuiteResult> RunSuite(int runs, bool bulk, CancellationToken ct = default)
    {
        var warm = new Position();
        Position.Set(Fens.Startpos, warm);
        for (int i = 0; i < 3; i++) { if (bulk) CountBulk(warm, 5); else CountFast(warm, 5); }

        var results = new List<SuiteResult>(SuitePositions.Length);
        foreach (var (name, fen, depth, expected) in SuitePositions)
        {
            ct.ThrowIfCancellationRequested();
            var pos = new Position();
            Position.Set(fen, pos);
            long best = long.MaxValue, nodes = 0;
            for (int r = 0; r < runs; r++)
            {
                ct.ThrowIfCancellationRequested();
                var sw = Stopwatch.StartNew();
                nodes = bulk ? CountBulk(pos, depth) : CountFast(pos, depth);
                sw.Stop();
                if (sw.ElapsedMilliseconds < best) best = sw.ElapsedMilliseconds;
            }
            results.Add(new SuiteResult(name, fen, depth, nodes, best, nodes == expected));
        }
        return results;
    }

    public static void Suite(int runs = 3, bool bulk = false, CancellationToken ct = default)
    {
        var results = RunSuite(runs, bulk, ct);

        Console.WriteLine($"perft suite [{(bulk ? "bulk-count, perft-only" : "emit path, search uses this")}]  (best of {runs})");
        long totalNodes = 0, totalMs = 0;
        bool allOk = true;
        foreach (var r in results)
        {
            totalNodes += r.Nodes;
            totalMs += r.Ms;
            allOk &= r.Ok;
            Console.WriteLine($"  {(r.Ok ? "ok  " : "FAIL")} d{r.Depth} {r.Nodes,14:n0}  {r.Ms,6} ms  {Log.Nps(r.Nodes, r.Ms),12}   {r.Fen}");
        }
        Console.WriteLine($"  ---- total {totalNodes,14:n0} nodes  {totalMs,6} ms  {Log.Nps(totalNodes, totalMs),12}   [{(allOk ? "all correct" : "MISMATCH!")}]");
    }

    /// <summary>
    /// Run perft at depths 1..maxDepth from a FEN, logging nodes, time, and throughput.
    /// </summary>
    public static void Bench(string fen, int maxDepth, CancellationToken ct = default)
    {
        var pos = new Position();
        Position.Set(fen, pos);
        Console.WriteLine($"perft  {fen}");
        for (int d = 1; d <= maxDepth; d++)
        {
            if (ct.IsCancellationRequested) { Console.WriteLine("cancelled."); return; }
            var sw = Stopwatch.StartNew();
            long nodes = Count(pos, d);
            sw.Stop();
            Console.WriteLine($"  depth {d,2}: {nodes,14:n0} nodes  {sw.ElapsedMilliseconds,7} ms  {Log.Nps(nodes, sw.ElapsedMilliseconds)}");
        }
    }
}
