using Core;

namespace Engine;

/// <summary>
/// Plays complete games for end-to-end testing: the engine against itself (or two depth settings).
/// Reuses the real <see cref="Searcher"/> and <see cref="Position"/>, so it exercises the full
/// search/eval/make-unmake path through to a genuine terminal result. Doubles as the position
/// source for Stage-1 NNUE training data.
/// </summary>
public static class GameRunner
{
    /// <summary>
    /// The terminal outcome of a game, always reported from White's point of view.
    /// </summary>
    public enum Result { WhiteWins, BlackWins, Draw }

    /// <summary>
    /// Play one self-play game at fixed depth. Returns the result and prints a transcript.
    /// </summary>
    public static Result PlayGame(int depth, int maxPlies = 400, bool print = true)
    {
        var pos = new Position();
        Position.Set(Types.DEFAULT_FEN, pos);
        var white = new Searcher { Quiet = true };
        var black = new Searcher { Quiet = true };
        white.NewGame();
        black.NewGame();

        var moves = new List<string>();
        Result result = Result.Draw;
        string reason = "max plies";

        Span<Move> buf = stackalloc Move[256];

        for (int ply = 0; ply < maxPlies; ply++)
        {
            // Terminal checks before moving.
            int legalCount = Searcher.GenerateLegal(pos, buf);
            if (legalCount == 0)
            {
                if (pos.InCheck(pos.Turn))
                {
                    result = pos.Turn == Color.White ? Result.BlackWins : Result.WhiteWins;
                    reason = "checkmate";
                }
                else { result = Result.Draw; reason = "stalemate"; }
                break;
            }
            if (pos.IsRepetition()) { result = Result.Draw; reason = "repetition"; break; }
            if (pos.IsFiftyMoveRule()) { result = Result.Draw; reason = "fifty-move"; break; }

            var searcher = pos.Turn == Color.White ? white : black;
            Move m = searcher.Think(pos, SearchLimits.Depth(depth));
            if (m.ToFrom == 0) { result = Result.Draw; reason = "no move"; break; }

            moves.Add(m.ToString());
            pos.Play(pos.Turn, m);
        }

        if (print)
        {
            Console.WriteLine();
            PrintMoves(moves);
            Console.WriteLine($"\nResult: {ResultString(result)} ({reason}) in {moves.Count} plies");
            Console.WriteLine($"Final FEN: {pos.Fen()}");
        }
        return result;
    }

    /// <summary>
    /// Play one game between two configured searchers (e.g. NNUE vs hand-crafted), with a few random
    /// opening plies for variety. Returns the result from White's POV.
    /// </summary>
    public static Result PlayGame(Searcher white, Searcher black, int depth, Random rng, int randomPlies, int maxPlies = 400)
    {
        var pos = new Position();
        Position.Set(Types.DEFAULT_FEN, pos);
        white.NewGame();
        black.NewGame();
        Span<Move> buf = stackalloc Move[256];

        for (int ply = 0; ply < maxPlies; ply++)
        {
            int n = Searcher.GenerateLegal(pos, buf);
            if (n == 0)
            {
                if (pos.InCheck(pos.Turn))
                    return pos.Turn == Color.White ? Result.BlackWins : Result.WhiteWins;
                return Result.Draw;
            }
            if (pos.IsRepetition() || pos.IsFiftyMoveRule()) return Result.Draw;

            Move m;
            if (ply < randomPlies)
            {
                m = buf[rng.Next(n)];
            }
            else
            {
                var s = pos.Turn == Color.White ? white : black;
                m = s.Think(pos, SearchLimits.Depth(depth));
                if (m.ToFrom == 0) return Result.Draw;
            }
            pos.Play(pos.Turn, m);
        }
        return Result.Draw;
    }

    private static void PrintMoves(List<string> moves)
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < moves.Count; i += 2)
        {
            sb.Append($"{i / 2 + 1}. {moves[i]}");
            if (i + 1 < moves.Count) sb.Append($" {moves[i + 1]}");
            sb.Append("   ");
            if ((i / 2 + 1) % 6 == 0) sb.Append('\n');
        }
        Console.WriteLine(sb.ToString().TrimEnd());
    }

    private static string ResultString(Result r) => r switch
    {
        Result.WhiteWins => "1-0",
        Result.BlackWins => "0-1",
        _ => "1/2-1/2",
    };
}
