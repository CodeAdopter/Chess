namespace Core;

// ---------------------------------------------------------------------------
// Core domain vocabulary: colours, pieces, squares, directions, and moves.
//
// Almost every other file leans on the numeric encodings defined here, so they
// are deliberately chosen to make the common operations cheap bit-twiddles
// rather than table lookups or branches. The key conventions are:
//
//   * Square index 0..63 runs a1, b1, ... h1, a2, ... h8  (file-major, rank-minor).
//     So  square = rank * 8 + file,  and the low 3 bits are the file, the next
//     3 bits are the rank.
//   * A Piece packs colour and type into one int: colour is bit 3, type is bits
//     0..2  (piece = colour << 3 | type).
//   * A Move packs into 16 bits: flags (4) | from (6) | to (6).
//
// Read the comments on each type for the "why" behind the exact bit layout.
// ---------------------------------------------------------------------------

/// <summary>
/// A side to move expressed as a <em>type</em> rather than a value. The generic
/// move generator is specialised on <see cref="White"/> / <see cref="Black"/> so the
/// JIT can fold the colour-dependent constants (pawn push direction, promotion rank,
/// castling masks) at compile time and emit branch-free code for each side.
/// </summary>
public interface IColor { Color Value { get; } IColor Opposite(); }

/// <summary>
/// The white side, as a zero-size type usable as a generic argument. See <see cref="IColor"/>.
/// </summary>
public struct White : IColor { public readonly Color Value => Color.White; public readonly IColor Opposite() => new Black(); }

/// <summary>
/// The black side, as a zero-size type usable as a generic argument. See <see cref="IColor"/>.
/// </summary>
public struct Black : IColor { public readonly Color Value => Color.Black; public readonly IColor Opposite() => new White(); }

/// <summary>
/// The side to move. Values are fixed at 0/1 so they can index two-element arrays directly.
/// </summary>
public enum Color : int
{
    White = 0,
    Black = 1
}

public static class ColorExtensions
{
    /// <summary>
    /// Returns the opposing colour. Implemented as a single XOR (White^1 == Black, Black^1 == White).
    /// </summary>
    public static Color Flip(this Color c) => (Color)((int)c ^ (int)Color.Black);
}

/// <summary>
/// A step on the board expressed as a signed offset to a square index (see the file header for the
/// 0..63 layout). Because squares are file-major, North/South move by a whole rank (±8) and East/West
/// by one file (±1). These are used to shift entire bitboards at once via <see cref="Bitboard.Shift"/>.
/// </summary>
public enum Direction : int
{
    North = 8,
    NorthEast = 9,
    East = 1,
    SouthEast = -7,
    South = -8,
    SouthWest = -9,
    West = -1,
    NorthWest = 7,
    NorthNorth = 16,  // a pawn's initial two-square advance
    SouthSouth = -16
}

/// <summary>
/// The six kinds of piece, ignoring colour. Values 0..5 index the piece-square and material tables.
/// </summary>
public enum PieceType : int
{
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5
}

/// <summary>
/// A coloured piece, encoded as <c>colour &lt;&lt; 3 | type</c>. Bit 3 is the colour and the low three
/// bits are the <see cref="PieceType"/>, which is why white pieces are 0..5 and black pieces are 8..13.
/// <see cref="NoPiece"/> (14) marks an empty square. See <see cref="Types.MakePiece"/>,
/// <see cref="Types.TypeOf"/> and <see cref="Types.ColorOf"/> for the matching pack/unpack helpers.
/// </summary>
public enum Piece : int
{
    WhitePawn = 0,
    WhiteKnight = 1,
    WhiteBishop = 2,
    WhiteRook = 3,
    WhiteQueen = 4,
    WhiteKing = 5,
    BlackPawn = 8,
    BlackKnight = 9,
    BlackBishop = 10,
    BlackRook = 11,
    BlackQueen = 12,
    BlackKing = 13,
    NoPiece = 14
}

/// <summary>
/// A board square, indexed 0..63 as a1, b1, ... h1, a2, ... h8 (file-major). This is the canonical
/// index used by every bitboard and lookup table: bit <c>n</c> of a <c>ulong</c> bitboard is this square.
/// <see cref="NoSquare"/> (64) is the "no such square" sentinel, e.g. an absent en-passant target.
/// </summary>
public enum Square : int
{
    a1, b1, c1, d1, e1, f1, g1, h1,
    a2, b2, c2, d2, e2, f2, g2, h2,
    a3, b3, c3, d3, e3, f3, g3, h3,
    a4, b4, c4, d4, e4, f4, g4, h4,
    a5, b5, c5, d5, e5, f5, g5, h5,
    a6, b6, c6, d6, e6, f6, g6, h6,
    a7, b7, c7, d7, e7, f7, g7, h7,
    a8, b8, c8, d8, e8, f8, g8, h8,
    NoSquare = 64
}

/// <summary>
/// A board file (column) a..h, numbered 0..7. Equals the low three bits of a <see cref="Square"/>.
/// </summary>
public enum File : int
{
    FileA = 0, FileB = 1, FileC = 2, FileD = 3,
    FileE = 4, FileF = 5, FileG = 6, FileH = 7
}

/// <summary>
/// A board rank (row) 1..8, numbered 0..7. Equals bits 3..5 of a <see cref="Square"/>.
/// </summary>
public enum Rank : int
{
    Rank1 = 0, Rank2 = 1, Rank3 = 2, Rank4 = 3,
    Rank5 = 4, Rank6 = 5, Rank7 = 6, Rank8 = 7
}

/// <summary>
/// The 4-bit "kind of move" stored in the top nibble of a <see cref="Move"/>. The bit layout is meaningful,
/// not arbitrary: bit 3 (0b1000) marks a capture and bit 2 (0b0100) marks a promotion, so a move's broad
/// category can be tested with a single mask. The eight promotion variants reuse those mask values as their
/// own names (see the note below) to keep the encoding free of redundant constants.
/// </summary>
public enum MoveFlags : int
{
    Quiet = 0b0000,
    DoublePush = 0b0001,
    OO = 0b0010,    // kingside castle  (O-O)
    OOO = 0b0011,   // queenside castle (O-O-O)
    Capture = 0b1000,
    Captures = 0b1111,   // mask matching any capturing move (all capture/promotion-capture variants)
    EnPassant = 0b1010,
    // Mask bit shared by all promotion variants (Pr* = 0b01XX, Pc* = 0b11XX).
    // Use as `(flags & Promotions) != 0` to test "is this any kind of promotion?".
    Promotions = 0b0100,
    PromotionCaptures = 0b1100,
    // Promotion variants deliberately share the mask values above (PrKnight == Promotions,
    // PcKnight == PromotionCaptures, PcQueen == Captures): the named aliases double as the masks.
    #pragma warning disable CA1069 // enum members intentionally share constant values
    PrKnight = 0b0100,   // promote to knight (quiet)
    PrBishop = 0b0101,
    PrRook = 0b0110,
    PrQueen = 0b0111,
    PcKnight = 0b1100,   // promote to knight while capturing
    PcBishop = 0b1101,
    PcRook = 0b1110,
    PcQueen = 0b1111
    #pragma warning restore CA1069
}

/// <summary>
/// The castling rights still available, as a 4-bit flag set (one bit per side per wing). Stored in
/// <see cref="UndoInfo"/> and folded into the Zobrist hash so two otherwise-identical positions with
/// different rights are treated as distinct.
/// </summary>
[Flags]
public enum CastlingRights : byte
{
    None = 0,
    WhiteOO = 1,    // white kingside
    WhiteOOO = 2,   // white queenside
    BlackOO = 4,
    BlackOOO = 8,
    White = WhiteOO | WhiteOOO,
    Black = BlackOO | BlackOOO,
    All = White | Black
}

/// <summary>
/// A move packed into 16 bits: <c>flags (bits 12..15) | from (bits 6..11) | to (bits 0..5)</c>.
/// The compact form keeps move lists cache-friendly and makes equality a single integer compare.
/// Note that two moves are considered equal on their from/to squares alone (<see cref="ToFrom"/>);
/// the flags are not part of equality, which matters when matching a TT/PV move against a freshly
/// generated move list.
/// </summary>
public readonly struct Move
{
    private readonly ushort move;

    /// <summary>
    /// Wraps a pre-packed 16-bit move word.
    /// </summary>
    public Move(ushort m) => move = m;

    /// <summary>
    /// Builds a quiet move (<see cref="MoveFlags.Quiet"/>) from one square to another.
    /// </summary>
    public Move(Square from, Square to) => move = (ushort)(((int)from << 6) | (int)to);

    /// <summary>
    /// Builds a move with explicit <paramref name="flags"/> (capture, promotion, castle, etc.).
    /// </summary>
    public Move(Square from, Square to, MoveFlags flags) =>
        move = (ushort)(((int)flags << 12) | ((int)from << 6) | (int)to);

    /// <summary>
    /// Parses the first four characters of a coordinate move string such as "e2e4" or "e7e8q".
    /// Only the from/to squares are decoded here; any promotion suffix is ignored, so callers that
    /// need promotions must set the flag themselves (see the move generator / UCI layer).
    /// </summary>
    public Move(string moveStr)
    {
        var from = Types.CreateSquare((File)(moveStr[0] - 'a'), (Rank)(moveStr[1] - '1'));
        var to = Types.CreateSquare((File)(moveStr[2] - 'a'), (Rank)(moveStr[3] - '1'));
        move = (ushort)(((int)from << 6) | (int)to);
    }

    /// <summary>
    /// The destination square (low 6 bits).
    /// </summary>
    public readonly Square To => (Square)(move & 0x3f);

    /// <summary>
    /// The origin square (bits 6..11).
    /// </summary>
    public readonly Square From => (Square)((move >> 6) & 0x3f);

    /// <summary>
    /// The from+to bits as a single 12-bit integer, the identity used for equality and history indexing.
    /// </summary>
    public readonly int ToFrom => move & 0xffff;

    /// <summary>
    /// The move kind stored in the top nibble.
    /// </summary>
    public readonly MoveFlags Flags => (MoveFlags)((move >> 12) & 0xf);

    /// <summary>
    /// True if this move captures (tests the capture bit, 0b1000, of the flags).
    /// </summary>
    public readonly bool IsCapture => ((move >> 12) & 0b1000) != 0;

    /// <summary>
    /// Renders the move in long algebraic / UCI coordinate notation, e.g. "e2e4" or "e7e8q".
    /// </summary>
    public override string ToString()
    {
        return Types.SQSTR[(int)From] + Types.SQSTR[(int)To] + GetPromotionChar();
    }

    /// <summary>
    /// The promotion piece letter for UCI output, or "" when the move is not a promotion.
    /// </summary>
    private readonly string GetPromotionChar() => Flags switch
    {
        MoveFlags.PrQueen or MoveFlags.PcQueen => "q",
        MoveFlags.PrRook or MoveFlags.PcRook => "r",
        MoveFlags.PrBishop or MoveFlags.PcBishop => "b",
        MoveFlags.PrKnight or MoveFlags.PcKnight => "n",
        _ => "",
    };

    // Equality and hashing key off the from/to squares only (not the flags) so a move parsed from
    // a string or stored in the TT compares equal to the fully-flagged version from the generator.
    public override bool Equals(object? obj) => obj is Move m && ToFrom == m.ToFrom;
    public override int GetHashCode() => ToFrom;
    public static bool operator ==(Move a, Move b) => a.ToFrom == b.ToFrom;
    public static bool operator !=(Move a, Move b) => a.ToFrom != b.ToFrom;
}

/// <summary>
/// Constants, string tables, and the small bit-twiddling helpers that pack/unpack the encodings above.
/// Everything here is pure and allocation-free so it can sit on the hot path of move generation and eval.
/// </summary>
public static class Types
{
    /// <summary>
    /// Glyphs indexed by <see cref="Piece"/>: "PNBRQK" (white), "pnbrqk" (black), "." for empty. The "~&gt;" fillers cover the gap between white King (5) and black Pawn (8).
    /// </summary>
    public const string PIECE_STR = "PNBRQK~>pnbrqk.";

    /// <summary>
    /// The standard chess starting position in FEN.
    /// </summary>
    public const string DEFAULT_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -";

    /// <summary>
    /// "Kiwipete", a famously busy middlegame position used as a perft / move-generation correctness test.
    /// </summary>
    public const string KIWIPETE = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";

    /// <summary>
    /// Square names indexed 0..63 (a1..h8), with "None" at index 64 for <see cref="Square.NoSquare"/>. Used to render moves.
    /// </summary>
    public static readonly string[] SQSTR =
    [
        "a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1",
        "a2", "b2", "c2", "d2", "e2", "f2", "g2", "h2",
        "a3", "b3", "c3", "d3", "e3", "f3", "g3", "h3",
        "a4", "b4", "c4", "d4", "e4", "f4", "g4", "h4",
        "a5", "b5", "c5", "d5", "e5", "f5", "g5", "h5",
        "a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6",
        "a7", "b7", "c7", "d7", "e7", "f7", "g7", "h7",
        "a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8",
        "None"
    ];

    /// <summary>
    /// Human-readable suffixes for each <see cref="MoveFlags"/> value, used when printing a move with its kind (e.g. " O-O", " e.p.", "Q").
    /// </summary>
    public static readonly string[] MOVE_TYPESTR =
    [
        "", "", " O-O", " O-O-O", "N", "B", "R", "Q", " (capture)", "", " e.p.", "",
        "N", "B", "R", "Q"
    ];

    public const int NSQUARES = 64;
    public const int NPIECES = 15;       // 14 coloured pieces + NoPiece; sizes per-piece bitboard arrays
    public const int NPIECE_TYPES = 6;
    public const int NCOLORS = 2;

    /// <summary>
    /// Packs a colour and type into a <see cref="Piece"/> (<c>colour &lt;&lt; 3 | type</c>).
    /// </summary>
    public static Piece MakePiece(Color c, PieceType pt) => (Piece)((int)c << 3 | (int)pt);

    /// <summary>
    /// Extracts the <see cref="PieceType"/> from a <see cref="Piece"/> (low three bits).
    /// </summary>
    public static PieceType TypeOf(Piece pc) => (PieceType)((int)pc & 0b111);

    /// <summary>
    /// Extracts the <see cref="Color"/> from a <see cref="Piece"/> (bit 3).
    /// </summary>
    public static Color ColorOf(Piece pc) => (Color)(((int)pc & 0b1000) >> 3);

    /// <summary>
    /// The rank of a square (bits 3..5 of the index).
    /// </summary>
    public static Rank RankOf(Square s) => (Rank)((int)s >> 3);

    /// <summary>
    /// The file of a square (low three bits of the index).
    /// </summary>
    public static File FileOf(Square s) => (File)((int)s & 0b111);

    /// <summary>
    /// The index 0..14 of the "/" diagonal a square sits on (constant along a1-h8 direction). Used to index <see cref="Bitboard.MASK_DIAGONAL"/>.
    /// </summary>
    public static int DiagonalOf(Square s) => 7 + (int)RankOf(s) - (int)FileOf(s);

    /// <summary>
    /// The index 0..14 of the "\" anti-diagonal a square sits on. Used to index <see cref="Bitboard.MASK_ANTI_DIAGONAL"/>.
    /// </summary>
    public static int AntiDiagonalOf(Square s) => (int)RankOf(s) + (int)FileOf(s);

    /// <summary>
    /// Builds a <see cref="Square"/> from a file and rank (<c>rank &lt;&lt; 3 | file</c>).
    /// </summary>
    public static Square CreateSquare(File f, Rank r) => (Square)((int)r << 3 | (int)f);

    /// <summary>
    /// Maps a rank into the moving side's frame of reference: white sees ranks unchanged, black has them
    /// mirrored (rank 1 ↔ rank 8). Lets colour-agnostic code reason about "the 7th rank from here".
    /// </summary>
    public static Rank RelativeRank(Color c, Rank r) =>
        c == Color.White ? r : (Rank)((int)Rank.Rank8 - (int)r);

    /// <summary>
    /// Mirrors a direction for the given side: for black, "forward" (North) becomes South, etc.
    /// </summary>
    public static Direction RelativeDir(Color c, Direction d) =>
        (Direction)(c == Color.White ? (int)d : -(int)d);
}
