using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;

namespace Core;

/// <summary>
/// Precomputed attack tables for every piece type, the heart of fast move generation. "Leapers" (king,
/// knight, pawn) are plain per-square lookups. "Sliders" (rook, bishop, queen) depend on which squares are
/// blocked, so they use the classic magic-bitboard scheme: a per-square mask selects the relevant blocker
/// squares, a hash turns that blocker pattern into a small index, and a precomputed table maps the index to
/// the resulting attack set. Build the tables once with <see cref="Init"/> before generating any moves.
/// </summary>
public static class Tables
{
    // On CPUs with fast BMI2 (Intel Haswell+, AMD Zen3+) the slider index is a single PEXT instruction
    // (occ -> packed relevant-bits index) instead of the magic multiply+shift, and the per-square table is
    // sized to exactly 2^relevantBits, both smaller (better cache) and cheaper to index. Lookups branch
    // on Bmi2.X64.IsSupported, which the JIT treats as a constant, so the unused path is compiled away;
    // the attack tables are built in whichever layout matches. Where BMI2 is absent (e.g. WASM) the
    // classic magic multiply+shift over fixed-size tables is used instead.

    /// <summary>
    /// King attack set per square (the up-to-8 adjacent squares), indexed by square.
    /// </summary>
    public static ReadOnlySpan<ulong> KING_ATTACKS =>
    [
        0x302UL, 0x705UL, 0xe0aUL, 0x1c14UL,
        0x3828UL, 0x7050UL, 0xe0a0UL, 0xc040UL,
        0x30203UL, 0x70507UL, 0xe0a0eUL, 0x1c141cUL,
        0x382838UL, 0x705070UL, 0xe0a0e0UL, 0xc040c0UL,
        0x3020300UL, 0x7050700UL, 0xe0a0e00UL, 0x1c141c00UL,
        0x38283800UL, 0x70507000UL, 0xe0a0e000UL, 0xc040c000UL,
        0x302030000UL, 0x705070000UL, 0xe0a0e0000UL, 0x1c141c0000UL,
        0x3828380000UL, 0x7050700000UL, 0xe0a0e00000UL, 0xc040c00000UL,
        0x30203000000UL, 0x70507000000UL, 0xe0a0e000000UL, 0x1c141c000000UL,
        0x382838000000UL, 0x705070000000UL, 0xe0a0e0000000UL, 0xc040c0000000UL,
        0x3020300000000UL, 0x7050700000000UL, 0xe0a0e00000000UL, 0x1c141c00000000UL,
        0x38283800000000UL, 0x70507000000000UL, 0xe0a0e000000000UL, 0xc040c000000000UL,
        0x302030000000000UL, 0x705070000000000UL, 0xe0a0e0000000000UL, 0x1c141c0000000000UL,
        0x3828380000000000UL, 0x7050700000000000UL, 0xe0a0e00000000000UL, 0xc040c00000000000UL,
        0x203000000000000UL, 0x507000000000000UL, 0xa0e000000000000UL, 0x141c000000000000UL,
        0x2838000000000000UL, 0x5070000000000000UL, 0xa0e0000000000000UL, 0x40c0000000000000UL,
    ];

    /// <summary>
    /// Knight attack set per square (the up-to-8 L-shaped jumps), indexed by square.
    /// </summary>
    public static ReadOnlySpan<ulong> KNIGHT_ATTACKS =>
    [
        0x20400UL, 0x50800UL, 0xa1100UL, 0x142200UL,
        0x284400UL, 0x508800UL, 0xa01000UL, 0x402000UL,
        0x2040004UL, 0x5080008UL, 0xa110011UL, 0x14220022UL,
        0x28440044UL, 0x50880088UL, 0xa0100010UL, 0x40200020UL,
        0x204000402UL, 0x508000805UL, 0xa1100110aUL, 0x1422002214UL,
        0x2844004428UL, 0x5088008850UL, 0xa0100010a0UL, 0x4020002040UL,
        0x20400040200UL, 0x50800080500UL, 0xa1100110a00UL, 0x142200221400UL,
        0x284400442800UL, 0x508800885000UL, 0xa0100010a000UL, 0x402000204000UL,
        0x2040004020000UL, 0x5080008050000UL, 0xa1100110a0000UL, 0x14220022140000UL,
        0x28440044280000UL, 0x50880088500000UL, 0xa0100010a00000UL, 0x40200020400000UL,
        0x204000402000000UL, 0x508000805000000UL, 0xa1100110a000000UL, 0x1422002214000000UL,
        0x2844004428000000UL, 0x5088008850000000UL, 0xa0100010a0000000UL, 0x4020002040000000UL,
        0x400040200000000UL, 0x800080500000000UL, 0x1100110a00000000UL, 0x2200221400000000UL,
        0x4400442800000000UL, 0x8800885000000000UL, 0x100010a000000000UL, 0x2000204000000000UL,
        0x4020000000000UL, 0x8050000000000UL, 0x110a0000000000UL, 0x22140000000000UL,
        0x44280000000000UL, 0x0088500000000000UL, 0x0010a00000000000UL, 0x20400000000000UL
    ];

    /// <summary>
    /// Squares a white pawn on each square attacks (the two forward diagonals). The top two ranks are 0 since pawns never stand there.
    /// </summary>
    public static ReadOnlySpan<ulong> WHITE_PAWN_ATTACKS =>
    [
        0x200UL, 0x500UL, 0xa00UL, 0x1400UL,
        0x2800UL, 0x5000UL, 0xa000UL, 0x4000UL,
        0x20000UL, 0x50000UL, 0xa0000UL, 0x140000UL,
        0x280000UL, 0x500000UL, 0xa00000UL, 0x400000UL,
        0x2000000UL, 0x5000000UL, 0xa000000UL, 0x14000000UL,
        0x28000000UL, 0x50000000UL, 0xa0000000UL, 0x40000000UL,
        0x200000000UL, 0x500000000UL, 0xa00000000UL, 0x1400000000UL,
        0x2800000000UL, 0x5000000000UL, 0xa000000000UL, 0x4000000000UL,
        0x20000000000UL, 0x50000000000UL, 0xa0000000000UL, 0x140000000000UL,
        0x280000000000UL, 0x500000000000UL, 0xa00000000000UL, 0x400000000000UL,
        0x2000000000000UL, 0x5000000000000UL, 0xa000000000000UL, 0x14000000000000UL,
        0x28000000000000UL, 0x50000000000000UL, 0xa0000000000000UL, 0x40000000000000UL,
        0x200000000000000UL, 0x500000000000000UL, 0xa00000000000000UL, 0x1400000000000000UL,
        0x2800000000000000UL, 0x5000000000000000UL, 0xa000000000000000UL, 0x4000000000000000UL,
        0x0UL, 0x0UL, 0x0UL, 0x0UL,
        0x0UL, 0x0UL, 0x0UL, 0x0UL,
    ];

    /// <summary>
    /// Squares a black pawn on each square attacks (the two downward diagonals). The bottom two ranks are 0 since pawns never stand there.
    /// </summary>
    public static ReadOnlySpan<ulong> BLACK_PAWN_ATTACKS =>
    [
        0x0UL, 0x0UL, 0x0UL, 0x0UL,
        0x0UL, 0x0UL, 0x0UL, 0x0UL,
        0x2UL, 0x5UL, 0xaUL, 0x14UL,
        0x28UL, 0x50UL, 0xa0UL, 0x40UL,
        0x200UL, 0x500UL, 0xa00UL, 0x1400UL,
        0x2800UL, 0x5000UL, 0xa000UL, 0x4000UL,
        0x20000UL, 0x50000UL, 0xa0000UL, 0x140000UL,
        0x280000UL, 0x500000UL, 0xa00000UL, 0x400000UL,
        0x2000000UL, 0x5000000UL, 0xa000000UL, 0x14000000UL,
        0x28000000UL, 0x50000000UL, 0xa0000000UL, 0x40000000UL,
        0x200000000UL, 0x500000000UL, 0xa00000000UL, 0x1400000000UL,
        0x2800000000UL, 0x5000000000UL, 0xa000000000UL, 0x4000000000UL,
        0x20000000000UL, 0x50000000000UL, 0xa0000000000UL, 0x140000000000UL,
        0x280000000000UL, 0x500000000000UL, 0xa00000000000UL, 0x400000000000UL,
        0x2000000000000UL, 0x5000000000000UL, 0xa000000000000UL, 0x14000000000000UL,
        0x28000000000000UL, 0x50000000000000UL, 0xa0000000000000UL, 0x40000000000000UL,
    ];

    /// <summary>
    /// Reverses the order of all 64 bits (square 0 swaps with 63, etc.) via a butterfly network. Used by <see cref="SlidingAttacks"/> to compute attacks in the "negative" direction along a line.
    /// </summary>
    public static ulong Reverse(ulong b)
    {
        b = (b & 0x5555555555555555UL) << 1 | (b >> 1) & 0x5555555555555555UL;
        b = (b & 0x3333333333333333UL) << 2 | (b >> 2) & 0x3333333333333333UL;
        b = (b & 0x0f0f0f0f0f0f0f0fUL) << 4 | (b >> 4) & 0x0f0f0f0f0f0f0f0fUL;
        b = (b & 0x00ff00ff00ff00ffUL) << 8 | (b >> 8) & 0x00ff00ff00ff00ffUL;

        return (b << 48) | ((b & 0xffff0000UL) << 16) |
            ((b >> 16) & 0xffff0000UL) | (b >> 48);
    }

    /// <summary>
    /// Attacks along a single line (one rank, file, or diagonal) for a slider on <paramref name="square"/>
    /// given occupancy <paramref name="occ"/>, using the "o ^ (o - 2r)" subtraction trick plus its
    /// bit-reversed twin to cover both directions. Used only when building the tables, not at search time.
    /// </summary>
    public static ulong SlidingAttacks(Square square, ulong occ, ulong mask)
    {
        return (((mask & occ) - Bitboard.SQUARE_BB[(int)square] * 2) ^
            Reverse(Reverse(mask & occ) - Reverse(Bitboard.SQUARE_BB[(int)square]) * 2)) & mask;
    }

    /// <summary>
    /// Full rook attack set (rank + file lines) for a square and occupancy. Used to populate the lookup tables.
    /// </summary>
    public static ulong GetRookAttacksForInit(Square square, ulong occ)
    {
        return SlidingAttacks(square, occ, Bitboard.MASK_FILE[(int)Types.FileOf(square)]) |
            SlidingAttacks(square, occ, Bitboard.MASK_RANK[(int)Types.RankOf(square)]);
    }

    /// <summary>
    /// Per-square mask of the "relevant" blocker squares for a rook (its rank+file, excluding the board edges).
    /// </summary>
    public static readonly ulong[] ROOK_ATTACK_MASKS = new ulong[64];
    /// <summary>
    /// Per-square right-shift (64 - popcount(mask)) used by the magic-index path to drop the unused high bits.
    /// </summary>
    public static readonly int[] ROOK_ATTACK_SHIFTS = new int[64];
    // One contiguous attack table (sum of per-square 2^relevantBits) plus a per-square base offset, rather
    // than 64 separately-allocated rows scattered across the GC heap, for better cache/TLB locality on the
    // slider-lookup hot path. Index = ROOK_BASE[sq] + (pext|magic index).
    public static readonly int[] ROOK_BASE = new int[64];
    // Sized at runtime (sum of per-square 2^relevantBits) inside InitialiseRookAttacks, which runs once
    // during single-threaded startup; reassigned there, so it can't be 'readonly'.
    #pragma warning disable CA2211 // populated once at startup, read-only thereafter
    /// <summary>
    /// The flat rook attack table; entry <c>ROOK_BASE[sq] + index</c> is the attack set for that square/blocker pattern.
    /// </summary>
    public static ulong[] ROOK_ATTACKS = null!;
    #pragma warning restore CA2211

    /// <summary>
    /// Per-square "magic" multipliers that hash a masked blocker pattern into a dense index, for CPUs without BMI2/PEXT.
    /// </summary>
    public static ReadOnlySpan<ulong> ROOK_MAGICS =>
    [
        0x0080001020400080UL, 0x0040001000200040UL, 0x0080081000200080UL, 0x0080040800100080UL,
        0x0080020400080080UL, 0x0080010200040080UL, 0x0080008001000200UL, 0x0080002040800100UL,
        0x0000800020400080UL, 0x0000400020005000UL, 0x0000801000200080UL, 0x0000800800100080UL,
        0x0000800400080080UL, 0x0000800200040080UL, 0x0000800100020080UL, 0x0000800040800100UL,
        0x0000208000400080UL, 0x0000404000201000UL, 0x0000808010002000UL, 0x0000808008001000UL,
        0x0000808004000800UL, 0x0000808002000400UL, 0x0000010100020004UL, 0x0000020000408104UL,
        0x0000208080004000UL, 0x0000200040005000UL, 0x0000100080200080UL, 0x0000080080100080UL,
        0x0000040080080080UL, 0x0000020080040080UL, 0x0000010080800200UL, 0x0000800080004100UL,
        0x0000204000800080UL, 0x0000200040401000UL, 0x0000100080802000UL, 0x0000080080801000UL,
        0x0000040080800800UL, 0x0000020080800400UL, 0x0000020001010004UL, 0x0000800040800100UL,
        0x0000204000808000UL, 0x0000200040008080UL, 0x0000100020008080UL, 0x0000080010008080UL,
        0x0000040008008080UL, 0x0000020004008080UL, 0x0000010002008080UL, 0x0000004081020004UL,
        0x0000204000800080UL, 0x0000200040008080UL, 0x0000100020008080UL, 0x0000080010008080UL,
        0x0000040008008080UL, 0x0000020004008080UL, 0x0000800100020080UL, 0x0000800041000080UL,
        0x00FFFCDDFCED714AUL, 0x007FFCDDFCED714AUL, 0x003FFFCDFFD88096UL, 0x0000040810002101UL,
        0x0001000204080011UL, 0x0001000204000801UL, 0x0001000082000401UL, 0x0001FFFAABFAD1A2UL
    ];

    /// <summary>
    /// Builds the rook lookup tables. For each square it computes the relevant-blocker mask, then enumerates
    /// every blocker subset of that mask (via the carry-rippler trick, <c>subset = (subset - mask) &amp; mask</c>)
    /// and stores the resulting attack set at the hashed index. Runs once at startup.
    /// </summary>
    public static void InitialiseRookAttacks()
    {
        ulong edges, subset;

        // First pass: per-square mask/shift/base and the running total table size.
        int total = 0;
        for (Square sq = Square.a1; sq <= Square.h8; sq++)
        {
            edges = ((Bitboard.MASK_RANK[(int)File.FileA] | Bitboard.MASK_RANK[(int)File.FileH]) & ~Bitboard.MASK_RANK[(int)Types.RankOf(sq)]) |
                ((Bitboard.MASK_FILE[(int)File.FileA] | Bitboard.MASK_FILE[(int)File.FileH]) & ~Bitboard.MASK_FILE[(int)Types.FileOf(sq)]);
            ulong mask = (Bitboard.MASK_RANK[(int)Types.RankOf(sq)]
                ^ Bitboard.MASK_FILE[(int)Types.FileOf(sq)]) & ~edges;
            ROOK_ATTACK_MASKS[(int)sq] = mask;
            int bits = Bitboard.PopCount(mask);
            ROOK_ATTACK_SHIFTS[(int)sq] = 64 - bits;
            ROOK_BASE[(int)sq] = total;
            // PEXT packs into 2^bits dense slots; the magic shift also yields a bits-wide index, so 2^bits
            // is the right per-square size for both paths.
            total += 1 << bits;
        }

        ROOK_ATTACKS = new ulong[total];
        for (Square sq = Square.a1; sq <= Square.h8; sq++)
        {
            ulong mask = ROOK_ATTACK_MASKS[(int)sq];
            int baseIdx = ROOK_BASE[(int)sq];
            subset = 0;
            do
            {
                ulong index = Bmi2.X64.IsSupported
                    ? Bmi2.X64.ParallelBitExtract(subset, mask)
                    : (subset * ROOK_MAGICS[(int)sq]) >> ROOK_ATTACK_SHIFTS[(int)sq];
                ROOK_ATTACKS[baseIdx + (int)index] = GetRookAttacksForInit(sq, subset);
                subset = (subset - mask) & mask;
            } while (subset != 0);
        }
    }

    /// <summary>
    /// The rook (and rook-half of queen) attack set for a square, given the full board occupancy. The hot-path lookup used throughout move generation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong GetRookAttacks(Square square, ulong occ)
    {
        int sq = (int)square;
        ulong mask = ROOK_ATTACK_MASKS[sq];
        if (Bmi2.X64.IsSupported)
            return ROOK_ATTACKS[ROOK_BASE[sq] + (int)Bmi2.X64.ParallelBitExtract(occ, mask)];
        return ROOK_ATTACKS[ROOK_BASE[sq] + (int)(((occ & mask) * ROOK_MAGICS[sq]) >> ROOK_ATTACK_SHIFTS[sq])];
    }

    /// <summary>
    /// Rook attacks that "see through" the first set of <paramref name="blockers"/> to the squares behind
    /// them. Removing the first blockers and re-querying exposes the square beyond, which is how pins and
    /// discovered-check candidates are found.
    /// </summary>
    public static ulong GetXrayRookAttacks(Square square, ulong occ, ulong blockers)
    {
        ulong attacks = GetRookAttacks(square, occ);
        blockers &= attacks;
        return attacks ^ GetRookAttacks(square, occ ^ blockers);
    }

    /// <summary>
    /// Full bishop attack set (both diagonals) for a square and occupancy. Used to populate the lookup tables.
    /// </summary>
    public static ulong GetBishopAttacksForInit(Square square, ulong occ)
    {
        return SlidingAttacks(square, occ, Bitboard.MASK_DIAGONAL[Types.DiagonalOf(square)]) |
            SlidingAttacks(square, occ, Bitboard.MASK_ANTI_DIAGONAL[Types.AntiDiagonalOf(square)]);
    }

    /// <summary>
    /// Per-square mask of the relevant blocker squares for a bishop (its two diagonals, excluding the board edges).
    /// </summary>
    public static readonly ulong[] BISHOP_ATTACK_MASKS = new ulong[64];
    /// <summary>
    /// Per-square right-shift for the bishop magic-index path (64 - popcount(mask)).
    /// </summary>
    public static readonly int[] BISHOP_ATTACK_SHIFTS = new int[64];
    /// <summary>
    /// Per-square base offset into the flat <see cref="BISHOP_ATTACKS"/> table.
    /// </summary>
    public static readonly int[] BISHOP_BASE = new int[64];
    // Sized at runtime inside InitialiseBishopAttacks (see ROOK_ATTACKS note); can't be 'readonly'.
    #pragma warning disable CA2211 // populated once at startup, read-only thereafter
    /// <summary>
    /// The flat bishop attack table; entry <c>BISHOP_BASE[sq] + index</c> is the attack set for that square/blocker pattern.
    /// </summary>
    public static ulong[] BISHOP_ATTACKS = null!;
    #pragma warning restore CA2211

    /// <summary>
    /// Per-square magic multipliers for the non-PEXT bishop path (see <see cref="ROOK_MAGICS"/>).
    /// </summary>
    public static ReadOnlySpan<ulong> BISHOP_MAGICS =>
    [
        0x0002020202020200UL, 0x0002020202020000UL, 0x0004010202000000UL, 0x0004040080000000UL,
        0x0001104000000000UL, 0x0000821040000000UL, 0x0000410410400000UL, 0x0000104104104000UL,
        0x0000040404040400UL, 0x0000020202020200UL, 0x0000040102020000UL, 0x0000040400800000UL,
        0x0000011040000000UL, 0x0000008210400000UL, 0x0000004104104000UL, 0x0000002082082000UL,
        0x0004000808080800UL, 0x0002000404040400UL, 0x0001000202020200UL, 0x0000800802004000UL,
        0x0000800400A00000UL, 0x0000200100884000UL, 0x0000400082082000UL, 0x0000200041041000UL,
        0x0002080010101000UL, 0x0001040008080800UL, 0x0000208004010400UL, 0x0000404004010200UL,
        0x0000840000802000UL, 0x0000404002011000UL, 0x0000808001041000UL, 0x0000404000820800UL,
        0x0001041000202000UL, 0x0000820800101000UL, 0x0000104400080800UL, 0x0000020080080080UL,
        0x0000404040040100UL, 0x0000808100020100UL, 0x0001010100020800UL, 0x0000808080010400UL,
        0x0000820820004000UL, 0x0000410410002000UL, 0x0000082088001000UL, 0x0000002011000800UL,
        0x0000080100400400UL, 0x0001010101000200UL, 0x0002020202000400UL, 0x0001010101000200UL,
        0x0000410410400000UL, 0x0000208208200000UL, 0x0000002084100000UL, 0x0000000020880000UL,
        0x0000001002020000UL, 0x0000040408020000UL, 0x0004040404040000UL, 0x0002020202020000UL,
        0x0000104104104000UL, 0x0000002082082000UL, 0x0000000020841000UL, 0x0000000000208800UL,
        0x0000000010020200UL, 0x0000000404080200UL, 0x0000040404040400UL, 0x0002020202020200UL
    ];

    /// <summary>
    /// Builds the bishop lookup tables, exactly as <see cref="InitialiseRookAttacks"/> does but over the diagonal masks. Runs once at startup.
    /// </summary>
    public static void InitialiseBishopAttacks()
    {
        ulong edges, subset;

        int total = 0;
        for (Square sq = Square.a1; sq <= Square.h8; sq++)
        {
            edges = ((Bitboard.MASK_RANK[(int)File.FileA] | Bitboard.MASK_RANK[(int)File.FileH]) & ~Bitboard.MASK_RANK[(int)Types.RankOf(sq)]) |
                ((Bitboard.MASK_FILE[(int)File.FileA] | Bitboard.MASK_FILE[(int)File.FileH]) & ~Bitboard.MASK_FILE[(int)Types.FileOf(sq)]);
            ulong mask = (Bitboard.MASK_DIAGONAL[Types.DiagonalOf(sq)]
                ^ Bitboard.MASK_ANTI_DIAGONAL[Types.AntiDiagonalOf(sq)]) & ~edges;
            BISHOP_ATTACK_MASKS[(int)sq] = mask;
            int bits = Bitboard.PopCount(mask);
            BISHOP_ATTACK_SHIFTS[(int)sq] = 64 - bits;
            BISHOP_BASE[(int)sq] = total;
            total += 1 << bits;
        }

        BISHOP_ATTACKS = new ulong[total];
        for (Square sq = Square.a1; sq <= Square.h8; sq++)
        {
            ulong mask = BISHOP_ATTACK_MASKS[(int)sq];
            int baseIdx = BISHOP_BASE[(int)sq];
            subset = 0;
            do
            {
                ulong index = Bmi2.X64.IsSupported
                    ? Bmi2.X64.ParallelBitExtract(subset, mask)
                    : (subset * BISHOP_MAGICS[(int)sq]) >> BISHOP_ATTACK_SHIFTS[(int)sq];
                BISHOP_ATTACKS[baseIdx + (int)index] = GetBishopAttacksForInit(sq, subset);
                subset = (subset - mask) & mask;
            } while (subset != 0);
        }
    }


    /// <summary>
    /// The bishop (and bishop-half of queen) attack set for a square, given the full board occupancy. The hot-path lookup used throughout move generation.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong GetBishopAttacks(Square square, ulong occ)
    {
        int sq = (int)square;
        ulong mask = BISHOP_ATTACK_MASKS[sq];
        if (Bmi2.X64.IsSupported)
            return BISHOP_ATTACKS[BISHOP_BASE[sq] + (int)Bmi2.X64.ParallelBitExtract(occ, mask)];
        return BISHOP_ATTACKS[BISHOP_BASE[sq] + (int)(((occ & mask) * BISHOP_MAGICS[sq]) >> BISHOP_ATTACK_SHIFTS[sq])];
    }

    /// <summary>Direct knight/king pseudo-attack lookups that skip the <see cref="Attacks"/> guard+switch
    /// dispatch, for the move-generation hot path where the square is always valid.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong KnightAttacks(Square s) => PSEUDO_LEGAL_ATTACKS[(int)PieceType.Knight][(int)s];

    /// <inheritdoc cref="KnightAttacks"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static ulong KingAttacks(Square s) => PSEUDO_LEGAL_ATTACKS[(int)PieceType.King][(int)s];

    /// <summary>
    /// X-ray bishop attacks: see <see cref="GetXrayRookAttacks"/>, but along diagonals.
    /// </summary>
    public static ulong GetXrayBishopAttacks(Square square, ulong occ, ulong blockers)
    {
        ulong attacks = GetBishopAttacks(square, occ);
        blockers &= attacks;
        return attacks ^ GetBishopAttacks(square, occ ^ blockers);
    }

    /// <summary>
    /// <c>SQUARES_BETWEEN_BB[a][b]</c> is the set of squares strictly between a and b when they share a
    /// rank, file, or diagonal (empty otherwise). Used to test whether a check can be blocked and which
    /// square a pinned piece may still move along.
    /// </summary>
    public static readonly ulong[][] SQUARES_BETWEEN_BB = new ulong[64][];

    /// <summary>
    /// Fills <see cref="SQUARES_BETWEEN_BB"/> by intersecting the two squares' slider attacks over a board occupied only by themselves.
    /// </summary>
    public static void InitialiseSquaresBetween()
    {
        ulong sqs;
        for (Square sq1 = Square.a1; sq1 <= Square.h8; sq1++)
        {
            SQUARES_BETWEEN_BB[(int)sq1] = new ulong[64];
            for (Square sq2 = Square.a1; sq2 <= Square.h8; sq2++)
            {
                sqs = Bitboard.SQUARE_BB[(int)sq1] | Bitboard.SQUARE_BB[(int)sq2];
                if (Types.FileOf(sq1) == Types.FileOf(sq2) || Types.RankOf(sq1) == Types.RankOf(sq2))
                    SQUARES_BETWEEN_BB[(int)sq1][(int)sq2] =
                    GetRookAttacksForInit(sq1, sqs) & GetRookAttacksForInit(sq2, sqs);
                else if (Types.DiagonalOf(sq1) == Types.DiagonalOf(sq2) || Types.AntiDiagonalOf(sq1) == Types.AntiDiagonalOf(sq2))
                    SQUARES_BETWEEN_BB[(int)sq1][(int)sq2] =
                    GetBishopAttacksForInit(sq1, sqs) & GetBishopAttacksForInit(sq2, sqs);
            }
        }
    }

    /// <summary>
    /// <c>LINE[a][b]</c> is the full line (rank, file, or diagonal) through a and b, including both
    /// endpoints, or empty if they are not collinear. Used to keep a pinned piece moving along the pin ray.
    /// </summary>
    public static readonly ulong[][] LINE = new ulong[64][];

    /// <summary>
    /// Fills <see cref="LINE"/> from the slider attacks of each square on an empty board, OR-ed with the two endpoints.
    /// </summary>
    public static void InitialiseLine()
    {
        for (Square sq1 = Square.a1; sq1 <= Square.h8; sq1++)
        {
            LINE[(int)sq1] = new ulong[64];
            for (Square sq2 = Square.a1; sq2 <= Square.h8; sq2++)
            {
                if (Types.FileOf(sq1) == Types.FileOf(sq2) || Types.RankOf(sq1) == Types.RankOf(sq2))
                    LINE[(int)sq1][(int)sq2] =
                    GetRookAttacksForInit(sq1, 0) & GetRookAttacksForInit(sq2, 0)
                    | Bitboard.SQUARE_BB[(int)sq1] | Bitboard.SQUARE_BB[(int)sq2];
                else if (Types.DiagonalOf(sq1) == Types.DiagonalOf(sq2) || Types.AntiDiagonalOf(sq1) == Types.AntiDiagonalOf(sq2))
                    LINE[(int)sq1][(int)sq2] =
                    GetBishopAttacksForInit(sq1, 0) & GetBishopAttacksForInit(sq2, 0)
                    | Bitboard.SQUARE_BB[(int)sq1] | Bitboard.SQUARE_BB[(int)sq2];
            }
        }
    }

    /// <summary>
    /// Pawn attack sets indexed by [colour][square], copied from the white/black pawn-attack spans at startup.
    /// </summary>
    public static readonly ulong[][] PAWN_ATTACKS = new ulong[Types.NCOLORS][];
    /// <summary>
    /// Attack sets on an otherwise-empty board, indexed by [pieceType][square]. For sliders these are the unobstructed rays; the occupancy-aware versions live in the magic tables.
    /// </summary>
    public static readonly ulong[][] PSEUDO_LEGAL_ATTACKS = new ulong[Types.NPIECE_TYPES][];

    /// <summary>
    /// Populates <see cref="PAWN_ATTACKS"/> and the leaper/empty-board slider rows of <see cref="PSEUDO_LEGAL_ATTACKS"/>.
    /// </summary>
    public static void InitialisePseudoLegal()
    {
        PAWN_ATTACKS[(int)Color.White] = new ulong[Types.NSQUARES];
        PAWN_ATTACKS[(int)Color.Black] = new ulong[Types.NSQUARES];
        WHITE_PAWN_ATTACKS.CopyTo(PAWN_ATTACKS[(int)Color.White]);
        BLACK_PAWN_ATTACKS.CopyTo(PAWN_ATTACKS[(int)Color.Black]);

        for (int i = 0; i < Types.NPIECE_TYPES; i++)
        {
            PSEUDO_LEGAL_ATTACKS[i] = new ulong[Types.NSQUARES];
        }

        KNIGHT_ATTACKS.CopyTo(PSEUDO_LEGAL_ATTACKS[(int)PieceType.Knight]);
        KING_ATTACKS.CopyTo(PSEUDO_LEGAL_ATTACKS[(int)PieceType.King]);

        for (Square s = Square.a1; s <= Square.h8; s++)
        {
            PSEUDO_LEGAL_ATTACKS[(int)PieceType.Rook][(int)s] = GetRookAttacksForInit(s, 0);
            PSEUDO_LEGAL_ATTACKS[(int)PieceType.Bishop][(int)s] = GetBishopAttacksForInit(s, 0);
            PSEUDO_LEGAL_ATTACKS[(int)PieceType.Queen][(int)s] = PSEUDO_LEGAL_ATTACKS[(int)PieceType.Rook][(int)s] |
                PSEUDO_LEGAL_ATTACKS[(int)PieceType.Bishop][(int)s];
        }
    }
    /// <summary>
    /// Builds every lookup table. Call once at program startup before any move generation or search.
    /// </summary>
    public static void Init()
    {
        InitialiseRookAttacks();
        InitialiseBishopAttacks();
        InitialiseSquaresBetween();
        InitialiseLine();
        InitialisePseudoLegal();
    }

    /// <summary>
    /// General attack lookup for any non-pawn piece on <paramref name="s"/> given board occupancy
    /// <paramref name="occ"/>. Returns empty for an invalid square. Pawns are asymmetric (they capture
    /// differently from how they move), so they are rejected here; use <see cref="PawnAttacks(Color, Square)"/>.
    /// </summary>
    public static ulong Attacks(PieceType pt, Square s, ulong occ)
    {
        if (s == Square.NoSquare || (int)s >= 64)
            return 0UL;

        return pt switch
        {
            PieceType.Pawn => throw new ArgumentException("The piece type may not be a pawn; use PawnAttacks instead"),
            PieceType.Rook => GetRookAttacks(s, occ),
            PieceType.Bishop => GetBishopAttacks(s, occ),
            PieceType.Queen => GetRookAttacks(s, occ) | GetBishopAttacks(s, occ),
            _ => PSEUDO_LEGAL_ATTACKS[(int)pt][(int)s],
        };
    }

    /// <summary>
    /// The squares attacked by a whole set of pawns at once (used e.g. to compute the opponent's pawn-controlled squares), by shifting the pawn bitboard along both capture diagonals.
    /// </summary>
    public static ulong PawnAttacks(Color c, ulong p)
    {
        return c == Color.White ?
            Bitboard.Shift(Direction.NorthWest, p) | Bitboard.Shift(Direction.NorthEast, p) :
            Bitboard.Shift(Direction.SouthWest, p) | Bitboard.Shift(Direction.SouthEast, p);
    }

    /// <summary>
    /// The squares a single pawn of colour <paramref name="c"/> on square <paramref name="s"/> attacks. Returns empty for <see cref="Square.NoSquare"/>.
    /// </summary>
    public static ulong PawnAttacks(Color c, Square s)
    {
        if (s == Square.NoSquare)
            return 0;
        return PAWN_ATTACKS[(int)c][(int)s];
    }
}
