using System.Runtime.CompilerServices;
using Core;
using Xunit;

namespace Tests;

/// <summary>
/// Builds the global attack and Zobrist tables once, before any test runs (a module initializer runs automatically at assembly load).
/// </summary>
internal static class TestInit
{
    [ModuleInitializer]
    public static void Init()
    {
        Tables.Init();
        Zobrist.Init();
    }
}

/// <summary>
/// Perft (performance test) correctness suite, the gold-standard check for a move generator. For each
/// position it counts the exact size of the legal-move tree at each depth and compares against known-good
/// values; any bug in move generation, make/unmake, or special-move handling shows up as a wrong count.
/// The positions are chosen to isolate each piece type and special move (castling, en passant, promotion).
/// </summary>
public class PerftTests
{
    /// <summary>
    /// Deepest perft depth exercised in the test run (kept modest so the suite stays fast).
    /// </summary>
    const int MaxTestDepth = 5;

    // Each row: a FEN, a human label, and the known-correct node counts at depths 1, 2, 3, ... A 0 count
    // marks a depth that is too expensive to verify here and is skipped.
    static readonly (string Fen, string Title, ulong[] Counts)[] Suite =
    [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1","startpos",        [20, 400, 8902, 197281, 4865609, 119060324, 3195901860]),
        ("4k3/pppppppp/8/8/8/8/PPPPPPPP/3K4 w - - 0 1",             "pawns",           [18, 324, 5658, 98766, 1683599, 28677559, 479771205]),
        ("3k1q2/8/8/8/8/8/8/2Q1K3 w - - 0 1",                       "queens",          [20, 322, 6371, 123074, 2456875, 48349901, 961477665]),
        ("2bk1b2/8/8/8/8/8/8/2B1KB2 w - - 0 1",                     "bishops",         [18, 305, 5587, 100301, 1889516, 35099794, 673156899]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                    "rooks castle",    [26, 568, 13744, 314346, 7594526, 179862938, 4408318687]),
        ("r3k2r/8/8/8/8/8/8/R3K2R w - - 0 1",                       "rooks no castle", [24, 482, 11522, 261282, 6326061, 149788933, 3680687823]),
        ("3k4/8/8/8/8/8/8/4K3 w - - 0 1",                           "kings",           [5, 25, 170, 1156, 7922, 53932, 375660]),
        ("1n1k2n1/8/8/8/8/8/8/1N2K1N1 w - - 0 1",                   "knights",         [11, 121, 1551, 19764, 273291, 3736172, 54351347]),
        ("8/4pppp/2k5/pppp4/4PPPP/5K2/PPPP4/8 w - - 0 1",           "en passant",      [18, 308, 5353, 91461, 1561105, 26214838, 435307144]),
        ("3kr3/7p/8/b2p2P1/1p2P2B/8/P7/3RK3 w - - 0 1",             "bishops+rooks",   [17, 263, 4760, 87145, 1737314, 34741198, 729088284]),
        ("3kr3/q6p/8/b2pn1P1/1p1NP2B/8/P6Q/3RK3 w - - 0 1",         "mixed pieces",    [35, 1134, 36189, 1161095, 37473046, 1214202944, 39761457145]),
        ("r2k4/1P6/P1PP4/8/8/4pp1p/6p1/4K2R w - - 0 1",             "promotion",       [16, 156, 1451, 14421, 157920, 1862695, 24837114]),
        ("r3k2r/1pq5/2b5/6Np/Pn6/5B2/5QP1/R3K2R w KQkq - 0 1",      "everything",      [46, 1911, 78942, 3177521, 128733003, 5120701935, 0]),
        ("RNBKQBNR/PPPPPPPP/8/8/8/8/pppppppp/rnbqkbnr w - - 0 1",   "flipped",         [4, 16, 176, 1936, 22428, 255135, 3830756]),
        ("rnbqkbnr/8/8/8/8/8/8/RNBQKBNR w KQkq - 0 1",              "major pieces",    [50, 2125, 96062, 4200525, 191462298, 8509434052, 0]),
    ];

    /// <summary>
    /// Expands <see cref="Suite"/> into one xUnit test case per (position, depth), skipping the depths marked unverified (count 0).
    /// </summary>
    public static TheoryData<string, string, int, ulong> Cases()
    {
        var data = new TheoryData<string, string, int, ulong>();
        foreach (var (fen, title, counts) in Suite)
            for (int d = 1; d <= MaxTestDepth && d <= counts.Length; d++)
                if (counts[d - 1] != 0)
                    data.Add(title, fen, d, counts[d - 1]);
        return data;
    }

    /// <summary>
    /// Runs perft on one position at one depth and asserts the node count matches the known-good value.
    /// </summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void Matches_known_count(string title, string fen, int depth, ulong expected)
    {
        var pos = new Position();
        Position.Set(fen, pos);
        ulong actual = Perft(pos, (uint)depth);
        Assert.True(actual == expected, $"{title} depth {depth}: expected {expected:N0}, got {actual:N0}");
    }

    // Self-contained perft driver (a colour-templated pair, like Engine.Perft) so the tests don't depend on
    // the Engine project. One shared buffer is sliced per ply to avoid re-allocating at every node.
    static ulong Perft(Position p, uint depth)
    {
        Span<Move> buf = stackalloc Move[256 * (int)depth];
        return p.Turn == Color.White ? PerftW(p, depth, buf) : PerftB(p, depth, buf);
    }

    static ulong PerftW(Position p, uint depth, Span<Move> buf)
    {
        int n = p.GenerateLegalsInto<White>(buf);
        if (depth == 1) return (ulong)n;
        ulong nodes = 0;
        for (int i = 0; i < n; i++)
        {
            p.Play(Color.White, buf[i]);
            nodes += PerftB(p, depth - 1, buf[256..]);
            p.Undo(Color.White, buf[i]);
        }
        return nodes;
    }

    static ulong PerftB(Position p, uint depth, Span<Move> buf)
    {
        int n = p.GenerateLegalsInto<Black>(buf);
        if (depth == 1) return (ulong)n;
        ulong nodes = 0;
        for (int i = 0; i < n; i++)
        {
            p.Play(Color.Black, buf[i]);
            nodes += PerftW(p, depth - 1, buf[256..]);
            p.Undo(Color.Black, buf[i]);
        }
        return nodes;
    }
}
