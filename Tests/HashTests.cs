using Core;
using Xunit;

namespace Tests;

/// <summary>
/// Tests that the Zobrist hash captures the full game state, not just piece placement. Two positions with
/// the same pieces but different castling rights or en-passant targets must hash differently (otherwise the
/// transposition table and repetition detection would conflate genuinely distinct positions), and a move
/// followed by its undo must restore the hash exactly.
/// </summary>
public class HashTests
{
    /// <summary>
    /// Same pieces, different castling rights must produce different hashes.
    /// </summary>
    [Fact]
    public void Castling_rights_change_the_hash()
    {
        var withRights = FromFen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -");
        var withoutRights = FromFen("r3k2r/8/8/8/8/8/8/R3K2R w - -");
        Assert.NotEqual(withRights.GetHash(), withoutRights.GetHash());
    }

    /// <summary>
    /// An available en-passant capture must change the hash (the file is folded in).
    /// </summary>
    [Fact]
    public void En_passant_file_changes_the_hash()
    {
        var withEp = FromFen("8/8/8/3pP3/8/8/8/4K2k w - d6");
        var withoutEp = FromFen("8/8/8/3pP3/8/8/8/4K2k w - -");
        Assert.NotEqual(withEp.GetHash(), withoutEp.GetHash());
    }

    /// <summary>
    /// Playing a sequence of moves (including a double push and an en-passant capture) and undoing them all must restore both the hash and the FEN exactly.
    /// </summary>
    [Fact]
    public void Play_then_undo_restores_hash_and_fen()
    {
        var p = FromFen(Types.DEFAULT_FEN);
        ulong initial = p.GetHash();

        var played = new List<(Color Side, Move Move)>();
        foreach (string uci in new[] { "e2e4", "a7a6", "e4e5", "d7d5", "e5d6" })
        {
            Move move = FindLegal(p, uci);
            Color side = p.Turn;
            p.Play(side, move);
            played.Add((side, move));
        }
        for (int i = played.Count - 1; i >= 0; i--)
            p.Undo(played[i].Side, played[i].Move);

        Assert.Equal(initial, p.GetHash());
        Assert.Equal(Types.DEFAULT_FEN, p.Fen());
    }

    /// <summary>
    /// Shuffling the rooks back and forth reaches the same piece placement repeatedly, but the very first
    /// rook move spent the castling rights, so those positions are NOT true repetitions. Guards against a
    /// hash that ignores rights wrongly flagging a draw.
    /// </summary>
    [Fact]
    public void Changed_castling_rights_prevent_false_repetition()
    {
        var p = FromFen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq -");
        foreach (string uci in new[] { "h1h2", "h8h7", "h2h1", "h7h8", "h1h2", "h8h7", "h2h1", "h7h8" })
            p.Play(p.Turn, FindLegal(p, uci));

        Assert.False(p.IsRepetition());
    }

    /// <summary>
    /// Builds a position from a FEN string.
    /// </summary>
    static Position FromFen(string fen)
    {
        var p = new Position();
        Position.Set(fen, p);
        return p;
    }

    /// <summary>
    /// Finds the legal move matching a UCI coordinate string (so tests can drive the board by notation). Throws if no such legal move exists.
    /// </summary>
    static Move FindLegal(Position p, string uci)
    {
        var moves = new Move[256];
        int n = p.Turn == Color.White
            ? p.GenerateLegalsInto<White>(moves)
            : p.GenerateLegalsInto<Black>(moves);
        for (int i = 0; i < n; i++)
            if (moves[i].ToString() == uci) return moves[i];
        throw new InvalidOperationException($"legal move not found: {uci} in {p.Fen()}");
    }
}
