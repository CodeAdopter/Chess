using System.Numerics;
using System.Runtime.CompilerServices;

namespace Core;

/// <summary>
/// Bitboard primitives: a board is a 64-bit integer with bit <c>n</c> set when square <c>n</c>
/// (a1=0 .. h8=63, see <see cref="Square"/>) is occupied. This class holds the constant masks the
/// rest of the engine intersects against (files, ranks, diagonals, castling corridors) plus the few
/// low-level bit operations (population count, least-significant-bit extraction, and directional
/// shifts) that move generation and evaluation run millions of times per search.
/// </summary>
public static class Bitboard
{
    /// <summary>
    /// The eight files as bitboards: <c>MASK_FILE[FileA]</c> has every square on the a-file set, and so on.
    /// </summary>
    public static ReadOnlySpan<ulong> MASK_FILE =>
    [
        0x0101010101010101UL, 0x0202020202020202UL, 0x0404040404040404UL, 0x0808080808080808UL,
        0x1010101010101010UL, 0x2020202020202020UL, 0x4040404040404040UL, 0x8080808080808080UL
    ];

    /// <summary>
    /// The eight ranks as bitboards: <c>MASK_RANK[Rank1]</c> is the bottom row (a1..h1), etc.
    /// </summary>
    public static ReadOnlySpan<ulong> MASK_RANK =>
    [
        0xffUL, 0xff00UL, 0xff0000UL, 0xff000000UL,
        0xff00000000UL, 0xff0000000000UL, 0xff000000000000UL, 0xff00000000000000UL
    ];

    /// <summary>
    /// The 15 "/" diagonals as bitboards, indexed by <see cref="Types.DiagonalOf"/>. Used to build sliding-piece (bishop/queen) attacks.
    /// </summary>
    public static ReadOnlySpan<ulong> MASK_DIAGONAL =>
    [
        0x80UL, 0x8040UL, 0x804020UL,
        0x80402010UL, 0x8040201008UL, 0x804020100804UL,
        0x80402010080402UL, 0x8040201008040201UL, 0x4020100804020100UL,
        0x2010080402010000UL, 0x1008040201000000UL, 0x804020100000000UL,
        0x402010000000000UL, 0x201000000000000UL, 0x100000000000000UL
    ];

    /// <summary>
    /// The 15 "\" anti-diagonals as bitboards, indexed by <see cref="Types.AntiDiagonalOf"/>. The other half of sliding-piece attack construction.
    /// </summary>
    public static ReadOnlySpan<ulong> MASK_ANTI_DIAGONAL =>
    [
        0x1UL, 0x102UL, 0x10204UL,
        0x1020408UL, 0x102040810UL, 0x10204081020UL,
        0x1020408102040UL, 0x102040810204080UL, 0x204081020408000UL,
        0x408102040800000UL, 0x810204080000000UL, 0x1020408000000000UL,
        0x2040800000000000UL, 0x4080000000000000UL, 0x8000000000000000UL
    ];

    /// <summary>
    /// A single-bit mask per square: <c>SQUARE_BB[s] == 1UL &lt;&lt; s</c>. Index 64 (<see cref="Square.NoSquare"/>)
    /// is 0, which lets code mask in a square unconditionally even when it might be "no square".
    /// </summary>
    public static ReadOnlySpan<ulong> SQUARE_BB =>
    [
        0x1UL, 0x2UL, 0x4UL, 0x8UL,
        0x10UL, 0x20UL, 0x40UL, 0x80UL,
        0x100UL, 0x200UL, 0x400UL, 0x800UL,
        0x1000UL, 0x2000UL, 0x4000UL, 0x8000UL,
        0x10000UL, 0x20000UL, 0x40000UL, 0x80000UL,
        0x100000UL, 0x200000UL, 0x400000UL, 0x800000UL,
        0x1000000UL, 0x2000000UL, 0x4000000UL, 0x8000000UL,
        0x10000000UL, 0x20000000UL, 0x40000000UL, 0x80000000UL,
        0x100000000UL, 0x200000000UL, 0x400000000UL, 0x800000000UL,
        0x1000000000UL, 0x2000000000UL, 0x4000000000UL, 0x8000000000UL,
        0x10000000000UL, 0x20000000000UL, 0x40000000000UL, 0x80000000000UL,
        0x100000000000UL, 0x200000000000UL, 0x400000000000UL, 0x800000000000UL,
        0x1000000000000UL, 0x2000000000000UL, 0x4000000000000UL, 0x8000000000000UL,
        0x10000000000000UL, 0x20000000000000UL, 0x40000000000000UL, 0x80000000000000UL,
        0x100000000000000UL, 0x200000000000000UL, 0x400000000000000UL, 0x800000000000000UL,
        0x1000000000000000UL, 0x2000000000000000UL, 0x4000000000000000UL, 0x8000000000000000UL,
        0x0UL
    ];

    // Castling masks, two per side per wing. The legality test for a castle is:
    //   * the king and the relevant rook are still on their home squares ("...OO_MASK"), AND
    //   * the squares the king passes over are empty and unattacked ("...BLOCKERS_AND_ATTACKERS_MASK").
    // Encoding them as bitboards turns those checks into a couple of AND operations in the generator.
    public const ulong WHITE_OO_MASK = 0x90UL;     // white king (e1) + kingside rook (h1)
    public const ulong WHITE_OOO_MASK = 0x11UL;    // white king (e1) + queenside rook (a1)
    public const ulong WHITE_OO_BLOCKERS_AND_ATTACKERS_MASK = 0x60UL;   // f1, g1
    public const ulong WHITE_OOO_BLOCKERS_AND_ATTACKERS_MASK = 0xeUL;   // b1, c1, d1
    public const ulong BLACK_OO_MASK = 0x9000000000000000UL;
    public const ulong BLACK_OOO_MASK = 0x1100000000000000UL;
    public const ulong BLACK_OO_BLOCKERS_AND_ATTACKERS_MASK = 0x6000000000000000UL;
    public const ulong BLACK_OOO_BLOCKERS_AND_ATTACKERS_MASK = 0xE00000000000000UL;
    /// <summary>
    /// All four corner rook squares plus both king squares, i.e. the squares whose occupancy can change castling rights.
    /// </summary>
    public const ulong ALL_CASTLING_MASK = 0x9100000000000091UL;

    // De Bruijn sequence + lookup for a bit-scan-forward fallback. Kept for reference / non-x64 paths;
    // the live Bsf below uses the hardware-friendly BitOperations.TrailingZeroCount instead.
    private static ReadOnlySpan<int> DEBRUIJN64 =>
    [
        0, 47,  1, 56, 48, 27,  2, 60,
        57, 49, 41, 37, 28, 16,  3, 61,
        54, 58, 35, 52, 50, 42, 21, 44,
        38, 32, 29, 23, 17, 11,  4, 62,
        46, 55, 26, 59, 40, 36, 15, 53,
        34, 51, 20, 43, 31, 22, 10, 45,
        25, 39, 14, 33, 19, 30,  9, 24,
        13, 18,  8, 12,  7,  6,  5, 63
    ];

    private const ulong MAGIC = 0x03f79d71b4cb0a89UL;

    /// <summary>
    /// Prints a bitboard as an 8×8 grid of 0/1 (rank 8 at the top) for debugging.
    /// </summary>
    public static void PrintBitboard(ulong b)
    {
        for (int i = 56; i >= 0; i -= 8)
        {
            for (int j = 0; j < 8; j++)
                Console.Write((char)(((b >> (i + j)) & 1) + '0') + " ");
            Console.WriteLine();
        }
        Console.WriteLine();
    }

    /// <summary>
    /// Counts the set bits (occupied squares) in a bitboard using the classic SWAR "Hamming weight"
    /// algorithm, a fixed sequence of mask/add/shift steps with no loop, so its cost is independent of
    /// how many bits are set. Preferred when the board may be densely populated.
    /// </summary>
    public static int PopCount(ulong x)
    {
        const ulong k1 = 0x5555555555555555UL;
        const ulong k2 = 0x3333333333333333UL;
        const ulong k4 = 0x0f0f0f0f0f0f0f0fUL;
        const ulong kf = 0x0101010101010101UL;

        x -= (x >> 1) & k1;                  // pairwise sums into 2-bit fields
        x = (x & k2) + ((x >> 2) & k2);      // 2-bit -> 4-bit field sums
        x = (x + (x >> 4)) & k4;             // 4-bit -> 8-bit field sums
        x = (x * kf) >> 56;                  // horizontal sum of the eight byte counts into the top byte
        return (int)x;
    }

    /// <summary>
    /// Counts set bits by clearing the lowest set bit one at a time (<c>x &amp;= x - 1</c>). The loop runs
    /// once per set bit, so this wins over <see cref="PopCount"/> only when very few bits are set.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int SparsePopCount(ulong x)
    {
        int count = 0;
        while (x != 0)
        {
            count++;
            x &= x - 1;
        }
        return count;
    }

    /// <summary>
    /// Returns the least-significant set bit's square and clears it from <paramref name="b"/>. This is the
    /// standard way to iterate the squares of a bitboard: <c>while (bb != 0) { var s = PopLsb(ref bb); ... }</c>.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Square PopLsb(ref ulong b)
    {
        Square lsb = Bsf(b);
        b &= b - 1;
        return lsb;
    }

    /// <summary>
    /// Bit-scan-forward: the index of the least-significant set bit, i.e. the lowest occupied square.
    /// </summary>
    /// <remarks>
    /// Portable across x64 (JITs to a single <c>tzcnt</c>) and WASM. <c>TrailingZeroCount(0) == 64 ==
    /// Square.NoSquare</c>, matching the previous Bmi1.X64 behaviour the move generator relies on, so an
    /// empty board scans to the "no square" sentinel rather than producing garbage.
    /// </remarks>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Square Bsf(ulong b)
    {
        return (Square)BitOperations.TrailingZeroCount(b);
    }

    /// <summary>
    /// Slides every set square one step in direction <paramref name="d"/>, dropping bits that fall off the
    /// edge. East/west shifts pre-mask the wrap-around file (e.g. a piece on h-file moving East would
    /// otherwise reappear on the a-file) so the result stays geometrically correct.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong Shift(Direction d, ulong b) => d switch
    {
        Direction.North => b << 8,
        Direction.South => b >> 8,
        Direction.NorthNorth => b << 16,
        Direction.SouthSouth => b >> 16,
        Direction.East => (b & ~MASK_FILE[(int)File.FileH]) << 1,
        Direction.West => (b & ~MASK_FILE[(int)File.FileA]) >> 1,
        Direction.NorthEast => (b & ~MASK_FILE[(int)File.FileH]) << 9,
        Direction.NorthWest => (b & ~MASK_FILE[(int)File.FileA]) << 7,
        Direction.SouthEast => (b & ~MASK_FILE[(int)File.FileH]) >> 7,
        Direction.SouthWest => (b & ~MASK_FILE[(int)File.FileA]) >> 9,
        _ => 0,
    };

    // Colour-selecting wrappers over the castling masks above, so the generator can stay colour-agnostic.
    /// <summary>
    /// King + kingside-rook home-square mask for the given colour.
    /// </summary>
    public static ulong OoMask(Color c) => c == Color.White ? WHITE_OO_MASK : BLACK_OO_MASK;
    /// <summary>
    /// King + queenside-rook home-square mask for the given colour.
    /// </summary>
    public static ulong OooMask(Color c) => c == Color.White ? WHITE_OOO_MASK : BLACK_OOO_MASK;
    /// <summary>
    /// Squares that must be empty and unattacked to castle kingside, for the given colour.
    /// </summary>
    public static ulong OoBlockersMask(Color c) =>
        c == Color.White ? WHITE_OO_BLOCKERS_AND_ATTACKERS_MASK : BLACK_OO_BLOCKERS_AND_ATTACKERS_MASK;
    /// <summary>
    /// Squares that must be empty and unattacked to castle queenside, for the given colour.
    /// </summary>
    public static ulong OooBlockersMask(Color c) =>
        c == Color.White ? WHITE_OOO_BLOCKERS_AND_ATTACKERS_MASK : BLACK_OOO_BLOCKERS_AND_ATTACKERS_MASK;

    /// <summary>
    /// The b1/b8 square, which the king does <em>not</em> pass over when castling queenside and so may be
    /// attacked without preventing the castle. Excluded from the "attackers" test (only blockers matter there).
    /// </summary>
    public static ulong IgnoreOooDanger(Color c) => c == Color.White ? 0x2UL : 0x200000000000000UL;
}
