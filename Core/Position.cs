using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Core;

/// <summary>
/// The per-ply state that a move changes but cannot be reconstructed from the board alone, so it must be
/// saved to undo the move: castling rights, the captured piece, the en-passant target square, the Zobrist
/// hash, and the halfmove (fifty-move-rule) clock. <see cref="Position"/> keeps an array of these, one per ply.
/// </summary>
public struct UndoInfo
{
    public CastlingRights Castling;
    public Piece Captured;
    public Square Epsq;
    public ulong Hash;
    public int HalfMoveClock;

    /// <summary>
    /// A cleared record (no rights, no capture, no en-passant), used for the initial position.
    /// </summary>
    public UndoInfo()
    {
        Castling = CastlingRights.None;
        Captured = Piece.NoPiece;
        Epsq = Square.NoSquare;
        Hash = 0;
        HalfMoveClock = 0;
    }

    /// <summary>
    /// Seeds the next ply's record from the previous one: rights and the halfmove clock carry forward
    /// (the mover then adjusts them), while the per-move fields (captured piece, en-passant, hash) reset.
    /// </summary>
    public UndoInfo(UndoInfo prev)
    {
        Castling = prev.Castling;
        Captured = Piece.NoPiece;
        Epsq = Square.NoSquare;
        Hash = 0;
        HalfMoveClock = prev.HalfMoveClock;
    }
}

/// <summary>
/// A chess position and the make/unmake machinery that drives it. The board is stored redundantly for speed:
/// one bitboard per piece type/colour (<c>pieceBB</c>), a cached occupancy bitboard per colour
/// (<c>colorBB</c>), and a plain 64-entry "mailbox" array (<c>board</c>) for "what's on this square?". A
/// running Zobrist <c>hash</c>, the side to move, the ply counter, and a per-ply <see cref="UndoInfo"/>
/// history complete the state. <see cref="Play"/>/<see cref="Undo"/> mutate the position incrementally
/// (updating the hash as they go); a parallel hash-free <see cref="PlayPerft"/>/<see cref="UndoPerft"/> pair
/// exists purely so perft, which never reads the hash, can skip that work.
/// </summary>
public class Position
{
    private readonly ulong[] pieceBB = new ulong[Types.NPIECES];

    // Per-colour occupancy, maintained incrementally by the piece mutators below. AllPieces(c) is on the
    // hot path of every move-generation node (twice) plus eval/search; recomputing it by OR-ing six piece
    // bitboards each call was pure overhead, so we keep it cached. Index = colour (0 = White, 1 = Black).
    private readonly ulong[] colorBB = new ulong[Types.NCOLORS];

    private readonly Piece[] board = new Piece[Types.NSQUARES];

    private Color sideToPlay;

    private int gamePly;

    private ulong hash;

    /// <summary>
    /// Per-ply undo records, indexed by <see cref="Ply"/>. Entry [ply] holds the state needed to undo the move that produced it (and its post-move hash, used for repetition detection).
    /// </summary>
    // Self-play caps games at MaxPlies=512 and MCTS adds tree-descent depth on top, so 256 was
    // not enough; any game past ply ~220 blew this array. 2048 covers the absolute worst case
    // (5949-ply 75-move-rule game) plus ample MCTS headroom; memory cost is negligible.
    public readonly UndoInfo[] History = new UndoInfo[2048];

    /// <summary>
    /// Bitboard of enemy pieces currently giving check. Refreshed by <see cref="MoveGeneration.GenerateLegalsInto"/>.
    /// </summary>
    public ulong Checkers { get; internal set; }

    /// <summary>
    /// Bitboard of our own pieces pinned to the king. Refreshed by <see cref="MoveGeneration.GenerateLegalsInto"/>.
    /// </summary>
    public ulong Pinned { get; internal set; }

    // Castling rights revoked when a piece leaves (king/rook) or lands on (rook capture) a given square.
    // Every other square maps to None, so OR-ing from+to and clearing those bits is always correct.
    private static readonly CastlingRights[] CastleClear = BuildCastleClear();
    private static CastlingRights[] BuildCastleClear()
    {
        var m = new CastlingRights[Types.NSQUARES];
        m[(int)Square.e1] = CastlingRights.White;
        m[(int)Square.a1] = CastlingRights.WhiteOOO;
        m[(int)Square.h1] = CastlingRights.WhiteOO;
        m[(int)Square.e8] = CastlingRights.Black;
        m[(int)Square.a8] = CastlingRights.BlackOOO;
        m[(int)Square.h8] = CastlingRights.BlackOO;
        return m;
    }

    /// <summary>
    /// Creates an empty board (no pieces), white to move. Load a real position with <see cref="Set"/>.
    /// </summary>
    public Position()
    {
        sideToPlay = Color.White;
        gamePly = 0;
        hash = 0;
        Pinned = 0;
        Checkers = 0;

        for (int i = 0; i < 64; i++)
            board[i] = Piece.NoPiece;

        for (int i = 0; i < History.Length; i++)
            History[i] = new UndoInfo();
    }

    // The piece mutators below keep all three representations (mailbox, piece bitboards, colour occupancy)
    // and the Zobrist hash in sync. The colour index `((int)pc >> 3) & 1` extracts the colour bit of the piece.
    /// <summary>
    /// Places <paramref name="pc"/> on an empty square <paramref name="s"/>, updating the hash.
    /// </summary>
    private void PutPiece(Piece pc, Square s)
    {
        board[(int)s] = pc;
        ulong bb = 1UL << (int)s;
        pieceBB[(int)pc] |= bb;
        colorBB[((int)pc >> 3) & 1] |= bb;
        hash ^= Zobrist.Piece(pc, s);
    }
    /// <summary>
    /// Removes whatever piece is on <paramref name="s"/>, updating the hash.
    /// </summary>
    private void RemovePiece(Square s)
    {
        Piece pc = board[(int)s];
        ulong bb = 1UL << (int)s;
        hash ^= Zobrist.Piece(pc, s);
        pieceBB[(int)pc] &= ~bb;
        colorBB[((int)pc >> 3) & 1] &= ~bb;
        board[(int)s] = Piece.NoPiece;
    }
    /// <summary>
    /// Moves a piece from <paramref name="from"/> to <paramref name="to"/>, removing any piece captured on the destination, and updates the hash. Use <see cref="MovePieceQuiet"/> when the destination is known empty.
    /// </summary>
    private void MovePiece(Square from, Square to)
    {
        var movingPiece = board[(int)from];
        var capturedPiece = board[(int)to];

        hash ^= Zobrist.Piece(movingPiece, from)
             ^ Zobrist.Piece(movingPiece, to);

        if (capturedPiece != Piece.NoPiece)
            hash ^= Zobrist.Piece(capturedPiece, to);

        ulong toBB = 1UL << (int)to;
        ulong fromTo = (1UL << (int)from) | toBB;

        pieceBB[(int)movingPiece] ^= fromTo;
        colorBB[((int)movingPiece >> 3) & 1] ^= fromTo;
        if (capturedPiece != Piece.NoPiece)
        {
            pieceBB[(int)capturedPiece] &= ~toBB;
            colorBB[((int)capturedPiece >> 3) & 1] &= ~toBB;
        }

        board[(int)to] = movingPiece;
        board[(int)from] = Piece.NoPiece;
    }

    /// <summary>
    /// Plays a "null move": passes the turn to the opponent without moving a piece. Used by null-move pruning
    /// in the search to test whether the side to move is so far ahead that even doing nothing holds the bound.
    /// Clears the en-passant square (it would not survive a real pass) and bumps the halfmove clock.
    /// </summary>
    public void MakeNullMove()
    {
        hash ^= StateHash(History[gamePly].Castling, History[gamePly].Epsq);
        hash ^= Zobrist.SideToMove;
        sideToPlay = sideToPlay.Flip();
        ++gamePly;
        History[gamePly] = new UndoInfo(History[gamePly - 1]) { Epsq = Square.NoSquare };
        History[gamePly].HalfMoveClock++;

        hash ^= StateHash(History[gamePly].Castling, History[gamePly].Epsq);
        History[gamePly].Hash = hash;
    }

    /// <summary>
    /// Reverts the most recent <see cref="MakeNullMove"/>.
    /// </summary>
    public void UnmakeNullMove()
    {
        sideToPlay = sideToPlay.Flip();
        --gamePly;
        hash = History[gamePly].Hash;
    }

    // Hash-free piece mutators used only by PlayPerft/UndoPerft. Perft never reads the Zobrist hash, so the
    // ~6 hash XORs per move are pure overhead there; these keep the board/bitboards in sync without them.
    // The search keeps the hashing PutPiece/RemovePiece/MovePiece/MovePieceQuiet above, untouched.
    private void MovePieceQuietNoHash(Square from, Square to)
    {
        Piece pc = board[(int)from];
        ulong fromTo = (1UL << (int)from) | (1UL << (int)to);
        pieceBB[(int)pc] ^= fromTo;
        colorBB[((int)pc >> 3) & 1] ^= fromTo;
        board[(int)to] = pc;
        board[(int)from] = Piece.NoPiece;
    }
    private void PutPieceNoHash(Piece pc, Square s)
    {
        board[(int)s] = pc;
        ulong bb = 1UL << (int)s;
        pieceBB[(int)pc] |= bb;
        colorBB[((int)pc >> 3) & 1] |= bb;
    }
    private void RemovePieceNoHash(Square s)
    {
        Piece pc = board[(int)s];
        ulong bb = 1UL << (int)s;
        pieceBB[(int)pc] &= ~bb;
        colorBB[((int)pc >> 3) & 1] &= ~bb;
        board[(int)s] = Piece.NoPiece;
    }

    // Removes the piece on s when the caller already read it
    private void RemovePieceKnownNoHash(Square s, Piece pc)
    {
        ulong bb = 1UL << (int)s;
        pieceBB[(int)pc] &= ~bb;
        colorBB[((int)pc >> 3) & 1] &= ~bb;
        board[(int)s] = Piece.NoPiece;
    }

    // Capture move where the captured piece is already known
    private void MovePieceKnownCaptureNoHash(Square from, Square to, Piece captured)
    {
        var movingPiece = board[(int)from];
        ulong toBB = 1UL << (int)to;
        ulong fromTo = (1UL << (int)from) | toBB;
        pieceBB[(int)movingPiece] ^= fromTo;
        colorBB[((int)movingPiece >> 3) & 1] ^= fromTo;
        pieceBB[(int)captured] &= ~toBB;
        colorBB[((int)captured >> 3) & 1] &= ~toBB;
        board[(int)to] = movingPiece;
        board[(int)from] = Piece.NoPiece;
    }
    private void MovePieceNoHash(Square from, Square to)
    {
        var movingPiece = board[(int)from];
        var capturedPiece = board[(int)to];
        ulong toBB = 1UL << (int)to;
        ulong fromTo = (1UL << (int)from) | toBB;
        pieceBB[(int)movingPiece] ^= fromTo;
        colorBB[((int)movingPiece >> 3) & 1] ^= fromTo;
        if (capturedPiece != Piece.NoPiece)
        {
            pieceBB[(int)capturedPiece] &= ~toBB;
            colorBB[((int)capturedPiece >> 3) & 1] &= ~toBB;
        }
        board[(int)to] = movingPiece;
        board[(int)from] = Piece.NoPiece;
    }
    
    /// <summary>
    /// Deep-copies another position (board, bitboards, hash, and history up to the current ply) into a fresh independent instance.
    /// </summary>
    public Position(Position other)
    {
        Array.Copy(other.pieceBB, pieceBB, Types.NPIECES);
        Array.Copy(other.colorBB, colorBB, Types.NCOLORS);
        Array.Copy(other.board, board, Types.NSQUARES);
        sideToPlay = other.sideToPlay;
        gamePly = other.gamePly;
        hash = other.hash;
        Array.Copy(other.History, History, gamePly + 1);
        Checkers = other.Checkers;
        Pinned = other.Pinned;
    }

    /// <summary>
    /// Moves a piece to a known-empty square (no capture handling), updating the hash. The workhorse for quiet moves, pushes, and the rook hop in castling.
    /// </summary>
    private void MovePieceQuiet(Square from, Square to)
    {
        Piece pc = board[(int)from];
        hash ^= Zobrist.Piece(pc, from)
             ^ Zobrist.Piece(pc, to);
        ulong fromTo = (1UL << (int)from) | (1UL << (int)to);
        pieceBB[(int)pc] ^= fromTo;
        colorBB[((int)pc >> 3) & 1] ^= fromTo;
        board[(int)to] = pc;
        board[(int)from] = Piece.NoPiece;
    }

    /// <summary>
    /// The bitboard of all pieces of a specific coloured kind (e.g. white knights).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong BitboardOf(Piece pc) => Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(pieceBB), (int)pc);
    /// <summary>
    /// The bitboard of all pieces of the given colour and type.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong BitboardOf(Color c, PieceType pt) =>
        Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(pieceBB), (int)Types.MakePiece(c, pt));
    /// <summary>
    /// The piece on a square, or <see cref="Piece.NoPiece"/> if empty.
    /// </summary>
    public Piece At(Square sq) => board[(int)sq];
    /// <summary>
    /// The side to move.
    /// </summary>
    public Color Turn => sideToPlay;
    /// <summary>
    /// The current ply index (also the index of the live <see cref="History"/> record).
    /// </summary>
    public int Ply => gamePly;
    /// <summary>
    /// The current Zobrist hash of the position.
    /// </summary>
    public ulong GetHash() => hash;

    /// <summary>
    /// The hash contribution of the "state" fields that live outside the piece placement: castling rights and the en-passant file. XOR-ed in and out around each move.
    /// </summary>
    private static ulong StateHash(CastlingRights castling, Square epsq)
    {
        ulong stateHash = Zobrist.Castling[(int)castling & 0xF];
        if (epsq != Square.NoSquare)
            stateHash ^= Zobrist.EnPassantFile[(int)Types.FileOf(epsq)];
        return stateHash;
    }

    /// <summary>
    /// The colour's bishops and queens (the pieces that attack along diagonals).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong DiagonalSliders(Color c)
    {
        ref ulong pb = ref MemoryMarshal.GetArrayDataReference(pieceBB);
        return c == Color.White ?
            Unsafe.Add(ref pb, (int)Piece.WhiteBishop) | Unsafe.Add(ref pb, (int)Piece.WhiteQueen) :
            Unsafe.Add(ref pb, (int)Piece.BlackBishop) | Unsafe.Add(ref pb, (int)Piece.BlackQueen);
    }
    /// <summary>
    /// The colour's rooks and queens (the pieces that attack along ranks and files).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong OrthogonalSliders(Color c)
    {
        ref ulong pb = ref MemoryMarshal.GetArrayDataReference(pieceBB);
        return c == Color.White ?
            Unsafe.Add(ref pb, (int)Piece.WhiteRook) | Unsafe.Add(ref pb, (int)Piece.WhiteQueen) :
            Unsafe.Add(ref pb, (int)Piece.BlackRook) | Unsafe.Add(ref pb, (int)Piece.BlackQueen);
    }
    /// <summary>
    /// The cached occupancy bitboard for a colour (all of its pieces).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public ulong AllPieces(Color c) => Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(colorBB), (int)c);

    /// <summary>
    /// The set of pieces of colour <paramref name="c"/> that attack square <paramref name="s"/> under
    /// occupancy <paramref name="occ"/>. Found by firing each piece type's attack pattern <em>from</em> the
    /// target square and intersecting with that piece type's bitboard (attacks are symmetric, so "what does X
    /// attack from s" also answers "what attacks s"). Used for check detection and capture resolution.
    /// </summary>
    public ulong AttackersFrom(Color c, Square s, ulong occ)
    {
        return c == Color.White ?
            (Tables.PawnAttacks(Color.Black, s) & pieceBB[(int)Piece.WhitePawn]) |
            (Tables.Attacks(PieceType.Knight, s, occ) & pieceBB[(int)Piece.WhiteKnight]) |
            (Tables.Attacks(PieceType.Bishop, s, occ) & (pieceBB[(int)Piece.WhiteBishop] | pieceBB[(int)Piece.WhiteQueen])) |
            (Tables.Attacks(PieceType.Rook, s, occ) & (pieceBB[(int)Piece.WhiteRook] | pieceBB[(int)Piece.WhiteQueen])) :
            (Tables.PawnAttacks(Color.White, s) & pieceBB[(int)Piece.BlackPawn]) |
            (Tables.Attacks(PieceType.Knight, s, occ) & pieceBB[(int)Piece.BlackKnight]) |
            (Tables.Attacks(PieceType.Bishop, s, occ) & (pieceBB[(int)Piece.BlackBishop] | pieceBB[(int)Piece.BlackQueen])) |
            (Tables.Attacks(PieceType.Rook, s, occ) & (pieceBB[(int)Piece.BlackRook] | pieceBB[(int)Piece.BlackQueen]));
    }
    /// <summary>
    /// True if colour <paramref name="c"/>'s king is currently attacked (in check).
    /// </summary>
    public bool InCheck(Color c)
    {
        var kingSquare = Bitboard.Bsf(BitboardOf(c, PieceType.King));
        return AttackersFrom(c.Flip(), kingSquare, AllPieces(Color.White) | AllPieces(Color.Black)) != 0;
    }

    /// <summary>
    /// Applies move <paramref name="m"/> by side <paramref name="us"/>, advancing one ply and updating the
    /// board, hash, castling rights, en-passant square, halfmove clock, and history record. The
    /// <see cref="MoveFlags"/> on the move select how it is applied (quiet, capture, castle, promotion, etc.).
    /// Every <see cref="Play"/> must be paired with an <see cref="Undo"/> of the same move.
    /// </summary>
    public void Play(Color us, Move m)
    {
        hash ^= StateHash(History[gamePly].Castling, History[gamePly].Epsq);
        sideToPlay = sideToPlay.Flip();
        ++gamePly;
        var type = m.Flags;

        History[gamePly] = new UndoInfo(History[gamePly - 1]);
        var piece = board[(int)m.From];

        if (Types.TypeOf(piece) == PieceType.Pawn || m.IsCapture)
        {
            History[gamePly].HalfMoveClock = 0;
        }
        else
        {
            History[gamePly].HalfMoveClock++;
        }

        // A king or rook leaving its home square, or any piece landing on a rook's home square (capturing
        // it), revokes the matching castling rights. Table-driven so it's two array loads instead of an
        // eight-way comparison chain on every move; non-home squares map to None so the AND is a no-op.
        History[gamePly].Castling &= ~(CastleClear[(int)m.From] | CastleClear[(int)m.To]);

        switch (type)
        {
            case MoveFlags.Quiet:
                MovePieceQuiet(m.From, m.To);
                break;

            case MoveFlags.DoublePush:
                MovePieceQuiet(m.From, m.To);
                History[gamePly].Epsq = (Square)((int)m.From + (int)Types.RelativeDir(us, Direction.North));
                break;

            case MoveFlags.OO:
                if (us == Color.White)
                {
                    MovePieceQuiet(Square.e1, Square.g1);
                    MovePieceQuiet(Square.h1, Square.f1);
                }
                else
                {
                    MovePieceQuiet(Square.e8, Square.g8);
                    MovePieceQuiet(Square.h8, Square.f8);
                }
                break;

            case MoveFlags.OOO:
                if (us == Color.White)
                {
                    MovePieceQuiet(Square.e1, Square.c1);
                    MovePieceQuiet(Square.a1, Square.d1);
                }
                else
                {
                    MovePieceQuiet(Square.e8, Square.c8);
                    MovePieceQuiet(Square.a8, Square.d8);
                }
                break;

            case MoveFlags.EnPassant:
                MovePieceQuiet(m.From, m.To);
                RemovePiece((Square)((int)m.To + (int)Types.RelativeDir(us, Direction.South)));
                break;

            case MoveFlags.PrKnight:
                RemovePiece(m.From);
                PutPiece(Types.MakePiece(us, PieceType.Knight), m.To);
                break;

            case MoveFlags.PrBishop:
                RemovePiece(m.From);
                PutPiece(Types.MakePiece(us, PieceType.Bishop), m.To);
                break;

            case MoveFlags.PrRook:
                RemovePiece(m.From);
                PutPiece(Types.MakePiece(us, PieceType.Rook), m.To);
                break;

            case MoveFlags.PrQueen:
                RemovePiece(m.From);
                PutPiece(Types.MakePiece(us, PieceType.Queen), m.To);
                break;

            case MoveFlags.PcKnight:
                RemovePiece(m.From);
                History[gamePly].Captured = board[(int)m.To];
                RemovePiece(m.To);
                PutPiece(Types.MakePiece(us, PieceType.Knight), m.To);
                break;

            case MoveFlags.PcBishop:
                RemovePiece(m.From);
                History[gamePly].Captured = board[(int)m.To];
                RemovePiece(m.To);
                PutPiece(Types.MakePiece(us, PieceType.Bishop), m.To);
                break;

            case MoveFlags.PcRook:
                RemovePiece(m.From);
                History[gamePly].Captured = board[(int)m.To];
                RemovePiece(m.To);
                PutPiece(Types.MakePiece(us, PieceType.Rook), m.To);
                break;

            case MoveFlags.PcQueen:
                RemovePiece(m.From);
                History[gamePly].Captured = board[(int)m.To];
                RemovePiece(m.To);
                PutPiece(Types.MakePiece(us, PieceType.Queen), m.To);
                break;

            case MoveFlags.Capture:
                History[gamePly].Captured = board[(int)m.To];
                MovePiece(m.From, m.To);
                break;
        }

        hash ^= Zobrist.SideToMove;
        hash ^= StateHash(History[gamePly].Castling, History[gamePly].Epsq);
        History[gamePly].Hash = hash;
    }
    /// <summary>
    /// Reverts the move <paramref name="m"/> last played by <paramref name="us"/>, restoring the prior ply exactly (board, hash, rights, en-passant, and any captured piece from the history record).
    /// </summary>
    public void Undo(Color us, Move m)
    {
        hash = History[gamePly].Hash;
        hash ^= StateHash(History[gamePly].Castling, History[gamePly].Epsq);

        var type = m.Flags;
        switch (type)
        {
            case MoveFlags.Quiet:
                MovePieceQuiet(m.To, m.From);
                break;

            case MoveFlags.DoublePush:
                MovePieceQuiet(m.To, m.From);
                break;

            case MoveFlags.OO:
                if (us == Color.White)
                {
                    MovePieceQuiet(Square.g1, Square.e1);
                    MovePieceQuiet(Square.f1, Square.h1);
                }
                else
                {
                    MovePieceQuiet(Square.g8, Square.e8);
                    MovePieceQuiet(Square.f8, Square.h8);
                }
                break;

            case MoveFlags.OOO:
                if (us == Color.White)
                {
                    MovePieceQuiet(Square.c1, Square.e1);
                    MovePieceQuiet(Square.d1, Square.a1);
                }
                else
                {
                    MovePieceQuiet(Square.c8, Square.e8);
                    MovePieceQuiet(Square.d8, Square.a8);
                }
                break;

            case MoveFlags.EnPassant:
                MovePieceQuiet(m.To, m.From);
                PutPiece(Types.MakePiece(us.Flip(), PieceType.Pawn),
                        (Square)((int)m.To + (int)Types.RelativeDir(us, Direction.South)));
                break;

            case MoveFlags.PrKnight:
            case MoveFlags.PrBishop:
            case MoveFlags.PrRook:
            case MoveFlags.PrQueen:
                RemovePiece(m.To);
                PutPiece(Types.MakePiece(us, PieceType.Pawn), m.From);
                break;

            case MoveFlags.PcKnight:
            case MoveFlags.PcBishop:
            case MoveFlags.PcRook:
            case MoveFlags.PcQueen:
                RemovePiece(m.To);
                PutPiece(Types.MakePiece(us, PieceType.Pawn), m.From);
                PutPiece(History[gamePly].Captured, m.To);
                break;

            case MoveFlags.Capture:
                MovePieceQuiet(m.To, m.From);
                PutPiece(History[gamePly].Captured, m.To);
                break;
        }
        hash ^= Zobrist.SideToMove;
        hash ^= StateHash(History[gamePly - 1].Castling, History[gamePly - 1].Epsq);

        sideToPlay = sideToPlay.Flip();
        --gamePly;
    }

    // Hash-free make/unmake for perft. Identical board/castling/ep/captured bookkeeping to Play/Undo (so
    // the move generator sees the same state) but skips all Zobrist work, which perft never uses. Kept as
    // a separate pair rather than gating Play/Undo so the search's make/unmake stays byte-for-byte the same.
    /// <summary>
    /// Hash-free counterpart to <see cref="Play"/> for perft (see the note above). Keeps board/castling/ep/captured in sync but skips all Zobrist work.
    /// </summary>
    public void PlayPerft(Color us, Move m)
    {
        sideToPlay = sideToPlay.Flip();
        ++gamePly;
        var type = m.Flags;
        var from = m.From;
        var to = m.To;

        // Perft needs only Castling/Epsq/Captured in the undo record (not Hash/HalfMoveClock), so set those
        // three directly instead of constructing+copying a full UndoInfo from the previous ply.
        ref CastlingRights cc = ref MemoryMarshal.GetArrayDataReference(CastleClear);
        History[gamePly].Castling = History[gamePly - 1].Castling & ~(Unsafe.Add(ref cc, (int)from) | Unsafe.Add(ref cc, (int)to));
        History[gamePly].Epsq = Square.NoSquare;
        History[gamePly].Captured = Piece.NoPiece;

        switch (type)
        {
            case MoveFlags.Quiet:
                MovePieceQuietNoHash(from, to);
                break;
            case MoveFlags.DoublePush:
                MovePieceQuietNoHash(from, to);
                History[gamePly].Epsq = (Square)((int)from + (int)Types.RelativeDir(us, Direction.North));
                break;
            case MoveFlags.OO:
                if (us == Color.White) { MovePieceQuietNoHash(Square.e1, Square.g1); MovePieceQuietNoHash(Square.h1, Square.f1); }
                else { MovePieceQuietNoHash(Square.e8, Square.g8); MovePieceQuietNoHash(Square.h8, Square.f8); }
                break;
            case MoveFlags.OOO:
                if (us == Color.White) { MovePieceQuietNoHash(Square.e1, Square.c1); MovePieceQuietNoHash(Square.a1, Square.d1); }
                else { MovePieceQuietNoHash(Square.e8, Square.c8); MovePieceQuietNoHash(Square.a8, Square.d8); }
                break;
            case MoveFlags.EnPassant:
                MovePieceQuietNoHash(from, to);
                RemovePieceNoHash((Square)((int)to + (int)Types.RelativeDir(us, Direction.South)));
                break;
            case MoveFlags.PrKnight:
                RemovePieceNoHash(from); PutPieceNoHash(Types.MakePiece(us, PieceType.Knight), to);
                break;
            case MoveFlags.PrBishop:
                RemovePieceNoHash(from); PutPieceNoHash(Types.MakePiece(us, PieceType.Bishop), to);
                break;
            case MoveFlags.PrRook:
                RemovePieceNoHash(from); PutPieceNoHash(Types.MakePiece(us, PieceType.Rook), to);
                break;
            case MoveFlags.PrQueen:
                RemovePieceNoHash(from); PutPieceNoHash(Types.MakePiece(us, PieceType.Queen), to);
                break;
            case MoveFlags.PcKnight:
                {
                    Piece cap = board[(int)to]; History[gamePly].Captured = cap;
                    RemovePieceNoHash(from); RemovePieceKnownNoHash(to, cap);
                    PutPieceNoHash(Types.MakePiece(us, PieceType.Knight), to);
                    break;
                }
            case MoveFlags.PcBishop:
                {
                    Piece cap = board[(int)to]; History[gamePly].Captured = cap;
                    RemovePieceNoHash(from); RemovePieceKnownNoHash(to, cap);
                    PutPieceNoHash(Types.MakePiece(us, PieceType.Bishop), to);
                    break;
                }
            case MoveFlags.PcRook:
                {
                    Piece cap = board[(int)to]; History[gamePly].Captured = cap;
                    RemovePieceNoHash(from); RemovePieceKnownNoHash(to, cap);
                    PutPieceNoHash(Types.MakePiece(us, PieceType.Rook), to);
                    break;
                }
            case MoveFlags.PcQueen:
                {
                    Piece cap = board[(int)to]; History[gamePly].Captured = cap;
                    RemovePieceNoHash(from); RemovePieceKnownNoHash(to, cap);
                    PutPieceNoHash(Types.MakePiece(us, PieceType.Queen), to);
                    break;
                }
            case MoveFlags.Capture:
                {
                    Piece cap = board[(int)to]; History[gamePly].Captured = cap;
                    MovePieceKnownCaptureNoHash(from, to, cap);
                    break;
                }
        }
    }

    /// <summary>
    /// Hash-free counterpart to <see cref="Undo"/> for perft.
    /// </summary>
    public void UndoPerft(Color us, Move m)
    {
        var from = m.From;
        var to = m.To;
        switch (m.Flags)
        {
            case MoveFlags.Quiet:
            case MoveFlags.DoublePush:
                MovePieceQuietNoHash(to, from);
                break;
            case MoveFlags.OO:
                if (us == Color.White) { MovePieceQuietNoHash(Square.g1, Square.e1); MovePieceQuietNoHash(Square.f1, Square.h1); }
                else { MovePieceQuietNoHash(Square.g8, Square.e8); MovePieceQuietNoHash(Square.f8, Square.h8); }
                break;
            case MoveFlags.OOO:
                if (us == Color.White) { MovePieceQuietNoHash(Square.c1, Square.e1); MovePieceQuietNoHash(Square.d1, Square.a1); }
                else { MovePieceQuietNoHash(Square.c8, Square.e8); MovePieceQuietNoHash(Square.d8, Square.a8); }
                break;
            case MoveFlags.EnPassant:
                MovePieceQuietNoHash(to, from);
                PutPieceNoHash(Types.MakePiece(us.Flip(), PieceType.Pawn),
                        (Square)((int)to + (int)Types.RelativeDir(us, Direction.South)));
                break;
            case MoveFlags.PrKnight:
            case MoveFlags.PrBishop:
            case MoveFlags.PrRook:
            case MoveFlags.PrQueen:
                RemovePieceNoHash(to); PutPieceNoHash(Types.MakePiece(us, PieceType.Pawn), from);
                break;
            case MoveFlags.PcKnight:
            case MoveFlags.PcBishop:
            case MoveFlags.PcRook:
            case MoveFlags.PcQueen:
                RemovePieceNoHash(to); PutPieceNoHash(Types.MakePiece(us, PieceType.Pawn), from);
                PutPieceNoHash(History[gamePly].Captured, to);
                break;
            case MoveFlags.Capture:
                MovePieceQuietNoHash(to, from);
                PutPieceNoHash(History[gamePly].Captured, to);
                break;
        }
        sideToPlay = sideToPlay.Flip();
        --gamePly;
    }

    /// <summary>
    /// Threefold-repetition test for the search: true once the current position has been seen twice before
    /// (three times total). Only every second ply is checked (same side to move) and the scan stops at the
    /// last irreversible move (halfmove clock reset), since nothing before it can repeat.
    /// </summary>
    public bool IsRepetition()
    {
        if (gamePly < 4) return false;

        int count = 0;
        for (int i = gamePly - 2; i >= 0; i -= 2)
        {
            if (i < 0) break;

            if (History[i].Hash == hash)
            {
                count++;
                if (count >= 2) return true;
            }

            if (History[i].HalfMoveClock == 0)
                break;
        }

        return false;
    }

    /// <summary>
    /// True if the current position has occurred at least once earlier in the relevant
    /// history (back to the last irreversible move). Used by the network encoder so the
    /// model can distinguish a position's 1st visit from its 2nd visit and choose to play
    /// a different move; without this signal, identical inputs produce identical policies
    /// and self-play settles into a repetition-draw equilibrium.
    /// </summary>
    public bool HasRepeated()
    {
        if (gamePly < 4) return false;

        for (int i = gamePly - 2; i >= 0; i -= 2)
        {
            if (i < 0) break;
            if (History[i].Hash == hash) return true;
            if (History[i].HalfMoveClock == 0) break;
        }

        return false;
    }

    /// <summary>
    /// True once 100 halfmoves (50 full moves) have passed without a pawn move or capture (the fifty-move draw rule).
    /// </summary>
    public bool IsFiftyMoveRule()
    {
        return History[gamePly].HalfMoveClock >= 100;
    }

    /// <summary>
    /// Renders the board as an ASCII grid plus the FEN and hash, for debugging and the CLI.
    /// </summary>
    public override string ToString()
    {
        var sb = new StringBuilder();
        const string s = "   +---+---+---+---+---+---+---+---+\n";
        const string t = "     A   B   C   D   E   F   G   H\n";

        sb.Append(t);
        for (int i = 56; i >= 0; i -= 8)
        {
            sb.Append(s);
            sb.Append($" {i / 8 + 1} ");
            for (int j = 0; j < 8; j++)
                sb.Append($"| {Types.PIECE_STR[(int)board[i + j]]} ");
            sb.Append($"| {i / 8 + 1}\n");
        }
        sb.Append(s);
        sb.Append(t);
        sb.Append('\n');

        sb.Append($"FEN: {Fen()}\n");
        sb.Append($"Hash: 0x{hash:X}\n");

        return sb.ToString();
    }

    /// <summary>
    /// Serialises the position to FEN: piece placement, side to move, castling rights, and en-passant square (the halfmove/fullmove counters are omitted).
    /// </summary>
    public string Fen()
    {
        var fen = new StringBuilder();
        int empty;

        for (int i = 56; i >= 0; i -= 8)
        {
            empty = 0;
            for (int j = 0; j < 8; j++)
            {
                Piece p = board[i + j];
                if (p == Piece.NoPiece)
                {
                    empty++;
                }
                else
                {
                    if (empty != 0)
                    {
                        fen.Append(empty);
                        empty = 0;
                    }
                    fen.Append(Types.PIECE_STR[(int)p]);
                }
            }

            if (empty != 0) fen.Append(empty);
            if (i > 0) fen.Append('/');
        }

        fen.Append(sideToPlay == Color.White ? " w " : " b ");

        var rights = History[gamePly].Castling;
        if (rights == CastlingRights.None)
        {
            fen.Append('-');
        }
        else
        {
            if ((rights & CastlingRights.WhiteOO) != 0) fen.Append('K');
            if ((rights & CastlingRights.WhiteOOO) != 0) fen.Append('Q');
            if ((rights & CastlingRights.BlackOO) != 0) fen.Append('k');
            if ((rights & CastlingRights.BlackOOO) != 0) fen.Append('q');
        }
        fen.Append(' ');

        fen.Append(History[gamePly].Epsq == Square.NoSquare ? "-" : Types.SQSTR[(int)History[gamePly].Epsq]);

        return fen.ToString();
    }

    /// <summary>
    /// Resets <paramref name="p"/> and loads the position described by <paramref name="fen"/>. Parses the
    /// four core FEN fields (placement, side to move, castling, en-passant); the optional halfmove/fullmove
    /// counters are ignored. Rebuilds the hash from scratch so the loaded position is ready to search.
    /// </summary>
    public static void Set(string fen, Position p)
    {
        for (int i = 0; i < Types.NSQUARES; i++)
            p.board[i] = Piece.NoPiece;
        for (int i = 0; i < Types.NPIECES; i++)
            p.pieceBB[i] = 0;
        p.colorBB[0] = 0;
        p.colorBB[1] = 0;
        p.sideToPlay = Color.White;
        p.gamePly = 0;
        p.hash = 0;
        p.Checkers = 0;
        p.Pinned = 0;
        p.History[0] = new UndoInfo();

        int square = (int)Square.a8;
        int fenIdx = 0;

        while (fenIdx < fen.Length && fen[fenIdx] != ' ')
        {
            char ch = fen[fenIdx++];
            if (char.IsDigit(ch))
            {
                square += (ch - '0') * (int)Direction.East;
            }
            else if (ch == '/')
            {
                square += 2 * (int)Direction.South;
            }
            else
            {
                int pieceIdx = Types.PIECE_STR.IndexOf(ch);
                if (pieceIdx >= 0)
                {
                    p.PutPiece((Piece)pieceIdx, (Square)square);
                    square++;
                }
            }
        }

        if (fenIdx < fen.Length) fenIdx++;

        if (fenIdx < fen.Length)
        {
            p.sideToPlay = fen[fenIdx] == 'w' ? Color.White : Color.Black;
            fenIdx++;
            if (fenIdx < fen.Length) fenIdx++;
        }

        p.History[p.gamePly].Castling = CastlingRights.None;
        while (fenIdx < fen.Length && fen[fenIdx] != ' ')
        {
            switch (fen[fenIdx++])
            {
                case 'K': p.History[p.gamePly].Castling |= CastlingRights.WhiteOO; break;
                case 'Q': p.History[p.gamePly].Castling |= CastlingRights.WhiteOOO; break;
                case 'k': p.History[p.gamePly].Castling |= CastlingRights.BlackOO; break;
                case 'q': p.History[p.gamePly].Castling |= CastlingRights.BlackOOO; break;
            }
        }

        if (fenIdx < fen.Length) fenIdx++;

        if (fenIdx < fen.Length && fen[fenIdx] != '-')
        {
            if (fenIdx + 1 < fen.Length)
            {
                File f = (File)(fen[fenIdx] - 'a');
                Rank r = (Rank)(fen[fenIdx + 1] - '1');
                p.History[p.gamePly].Epsq = Types.CreateSquare(f, r);
            }
        }

        if (p.sideToPlay == Color.Black)
            p.hash ^= Zobrist.SideToMove;

        p.hash ^= StateHash(p.History[p.gamePly].Castling, p.History[p.gamePly].Epsq);
        p.History[p.gamePly].Hash = p.hash;
    }
}
