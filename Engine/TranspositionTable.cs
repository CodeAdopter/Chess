using Core;

namespace Engine;

/// <summary>
/// What a stored score means relative to the search window. <see cref="Exact"/> is a true value (the node
/// was searched fully); <see cref="Lower"/> is a lower bound (a beta cutoff, the real score is at least
/// this); <see cref="Upper"/> is an upper bound (a fail-low, the real score is at most this).
/// <see cref="None"/> marks an empty slot.
/// </summary>
public enum TtFlag : byte { None = 0, Exact = 1, Lower = 2, Upper = 3 }

/// <summary>
/// Fixed-size, power-of-two transposition table keyed on the position's incrementally-maintained
/// Zobrist hash (<see cref="Position.GetHash"/>). Depth-preferred replacement with an always-replace
/// fallback so deep, important entries survive but the table never stalls.
/// </summary>
public sealed class TranspositionTable
{
    /// <summary>
    /// One table slot: the full key (to detect index collisions), the best move found, the score and its bound type, and the depth it was searched to.
    /// </summary>
    public struct Entry
    {
        public ulong Key;
        public ushort Move;   // packed Move (0 = none)
        public short Score;
        public short Eval;     // reserved for static-eval caching (Stage 1)
        public byte Depth;
        public TtFlag Flag;
    }

    private Entry[] table = [];
    private ulong mask;

    /// <summary>
    /// Creates a table of approximately <paramref name="sizeMb"/> megabytes.
    /// </summary>
    public TranspositionTable(int sizeMb = 64) => Resize(sizeMb);

    /// <summary>
    /// Reallocates the table to about <paramref name="sizeMb"/> MB, rounding the entry count down to a power of two so lookups can mask the key instead of taking a modulo. Discards all existing entries.
    /// </summary>
    public void Resize(int sizeMb)
    {
        int entrySize = System.Runtime.InteropServices.Marshal.SizeOf<Entry>();
        long bytes = (long)sizeMb * 1024 * 1024;
        long want = bytes / entrySize;
        // Round down to a power of two so we can mask instead of modulo.
        int pow = 1;
        while ((long)pow * 2 <= want) pow *= 2;
        table = new Entry[pow];
        mask = (ulong)(pow - 1);
    }

    /// <summary>
    /// Empties the table (e.g. at the start of a fresh game).
    /// </summary>
    public void Clear() => Array.Clear(table);

    /// <summary>
    /// Looks up a position by hash key. Returns true and fills <paramref name="entry"/> only on a genuine hit (the stored key matches, ruling out an index collision with a different position).
    /// </summary>
    public bool Probe(ulong key, out Entry entry)
    {
        ref Entry e = ref table[key & mask];
        entry = e;
        return e.Key == key && e.Flag != TtFlag.None;
    }

    /// <summary>
    /// Store a result. <paramref name="score"/> is passed already adjusted for mate-distance
    /// (see <see cref="ScoreToTt"/>); the caller converts on the way in and out.
    /// </summary>
    public void Store(ulong key, ushort move, int score, int depth, TtFlag flag)
    {
        ref Entry e = ref table[key & mask];
        // Keep the deeper entry unless this slot is empty or holds a different position.
        if (e.Flag != TtFlag.None && e.Key == key && e.Depth > depth && flag != TtFlag.Exact)
            return;
        e.Key = key;
        if (move != 0 || e.Key != key) e.Move = move;
        e.Score = (short)score;
        e.Depth = (byte)depth;
        e.Flag = flag;
    }

    // Mate scores are stored relative to the node (distance-to-mate from here) and converted back to
    // root-relative on retrieval, so a mate found at depth N is still recognised after a TT hit.
    /// <summary>
    /// Adjusts a score for storage: rebases a mate score to be measured from this node rather than the root.
    /// </summary>
    public static int ScoreToTt(int score, int ply)
    {
        if (score >= Eval.MateBound) return score + ply;
        if (score <= -Eval.MateBound) return score - ply;
        return score;
    }

    /// <summary>
    /// The inverse of <see cref="ScoreToTt"/>: rebases a retrieved mate score back to root-relative distance.
    /// </summary>
    public static int ScoreFromTt(int score, int ply)
    {
        if (score >= Eval.MateBound) return score - ply;
        if (score <= -Eval.MateBound) return score + ply;
        return score;
    }
}
