using System.Runtime.CompilerServices;

namespace Core;

/// <summary>
/// A small, fast, deterministic pseudo-random generator (xorshift64*). Seeding it with a fixed value
/// makes the Zobrist keys identical on every run, which is what lets transposition-table and
/// repetition logic stay reproducible. It is not cryptographically secure and is not meant to be.
/// </summary>
public class PRNG(ulong seed)
{
    private ulong s = seed;

    /// <summary>
    /// Advances the state one step and returns the next 64-bit value (xorshift triple, then a multiply to mix the bits).
    /// </summary>
    private ulong Rand64()
    {
        s ^= s >> 12;
        s ^= s << 25;
        s ^= s >> 27;
        return s * 2685821657736338717UL;
    }

    /// <summary>
    /// Returns the next random value. Only <c>ulong</c> is supported; any other type throws.
    /// </summary>
    public T Rand<T>() where T : struct
    {
        if (typeof(T) == typeof(ulong))
            return (T)(object)Rand64();
        throw new NotSupportedException($"Type {typeof(T)} is not supported");
    }

    /// <summary>
    /// Returns a "sparse" random value, the bitwise AND of three draws, so each bit is set only ~1/8 of the
    /// time. Used where a value with few set bits is wanted (e.g. magic-number searching).
    /// </summary>
    public T SparseRand<T>() where T : struct
    {
        if (typeof(T) == typeof(ulong))
            return (T)(object)(Rand64() & Rand64() & Rand64());
        throw new NotSupportedException($"Type {typeof(T)} is not supported");
    }
}

/// <summary>
/// Zobrist hashing: every position gets a 64-bit key built by XOR-ing together one random number per
/// (piece, square), plus keys for side-to-move, castling rights, and the en-passant file. Because XOR is
/// its own inverse, a move updates the key incrementally (XOR the moving piece out of its old square and
/// into its new one) instead of rehashing the whole board, which is what makes the transposition table and
/// threefold-repetition detection cheap. Call <see cref="Init"/> once at startup to fill the tables.
/// </summary>
public static class Zobrist
{
    // Flat [piece * 64 + square] layout (NSQUARES == 64, so the index is (piece << 6) | square). A 1D
    // array is indexed with a single bounds check the JIT can elide, where the old ulong[,] paid two
    // bounds checks and a multiply per access on the make/unmake hot path. Values/order are unchanged,
    // so every hash (and thus the TT and repetition detection) is bit-for-bit identical.
    public static readonly ulong[] ZobristTable = new ulong[Types.NPIECES * Types.NSQUARES];
    /// <summary>
    /// One key per castling-rights combination (16 = 2^4 possible right sets), indexed by the <see cref="CastlingRights"/> bits.
    /// </summary>
    public static readonly ulong[] Castling = new ulong[16];
    /// <summary>
    /// One key per file on which an en-passant capture is currently available (only the file matters for hashing).
    /// </summary>
    public static readonly ulong[] EnPassantFile = new ulong[8];
    // Drawn once in Init() (it must be the PRNG's first value, before the table), so it can't be a
    // field-initialised 'readonly'. Set during single-threaded startup, read-only thereafter.
    #pragma warning disable CA2211 // populated once at startup, read-only thereafter
    public static ulong SideToMove;
    #pragma warning restore CA2211

    /// <summary>
    /// The hash key for a given piece sitting on a given square. XOR it into (or out of) a position hash to add (or remove) that piece.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Piece(Piece pc, Square s) => ZobristTable[((int)pc << 6) | (int)s];

    /// <summary>
    /// Fills every key table with values from the fixed-seed <see cref="PRNG"/>. Must run once during
    /// single-threaded startup, before any position is hashed. The draw order is fixed (side-to-move first,
    /// then the piece/square table, then castling, then en-passant) so the keys are stable across runs.
    /// </summary>
    public static void Init()
    {
        PRNG rng = new(70026072);

        SideToMove = rng.Rand<ulong>();

        for (int i = 0; i < Types.NPIECES; i++)
        {
            for (int j = 0; j < Types.NSQUARES; j++)
            {
                ZobristTable[(i << 6) | j] = rng.Rand<ulong>();
            }
        }

        for (int i = 0; i < Castling.Length; i++)
            Castling[i] = rng.Rand<ulong>();

        for (int i = 0; i < EnPassantFile.Length; i++)
            EnPassantFile[i] = rng.Rand<ulong>();
    }
}
