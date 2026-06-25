using System.Globalization;
using Core;
using Engine;
using Engine.Nnue;

namespace Cli;

/// <summary>
/// UCI driver so the engine plays in GUIs (Lucas Chess, Fritz, Arena, CuteChess). UCI (Universal Chess
/// Interface) is the standard text protocol GUIs use to talk to engines over stdin/stdout. Load an NNUE
/// net with <c>setoption name EvalFile value &lt;path&gt;</c> (relative paths resolve under
/// workspace/models); with no net it uses the hand-crafted evaluation.
/// </summary>
public static class Uci
{
    /// <summary>
    /// The protocol loop: read one command per line from stdin and respond on stdout until "quit".
    /// Maintains a single <see cref="Position"/> and <see cref="Searcher"/> across commands, which is
    /// what lets "position" set up the board and a later "go" search it.
    /// </summary>
    public static void Run(string? initialNet = null)
    {
        var pos = new Position();
        Position.Set(Types.DEFAULT_FEN, pos);
        var searcher = new Searcher();
        if (!string.IsNullOrEmpty(initialNet)) LoadNet(searcher, initialNet);
        else AutoLoadNet(searcher);

        string? line;
        while ((line = Console.ReadLine()) != null)
        {
            var t = line.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (t.Length == 0) continue;

            // Dispatch on the first token (the UCI verb). Unrecognised commands are silently ignored,
            // as the protocol expects. The string responses below are protocol text and must stay exact.
            switch (t[0])
            {
                case "uci":
                    Console.WriteLine("id name ChessEngine");
                    Console.WriteLine("id author Chess");
                    Console.WriteLine("option name EvalFile type string default <none>");
                    Console.WriteLine("option name Hash type spin default 64 min 1 max 4096");
                    Console.WriteLine("option name Threads type spin default 1 min 1 max 1");
                    Console.WriteLine("uciok");
                    break;
                case "isready": Console.WriteLine("readyok"); break;
                case "ucinewgame": searcher.NewGame(); Position.Set(Types.DEFAULT_FEN, pos); break;
                case "setoption": HandleSetOption(searcher, t); break;
                case "position": ParsePosition(pos, t); break;
                case "go": HandleGo(pos, searcher, t); break;
                case "quit": return;
            }
        }
    }

    /// <summary>Load an NNUE net into the searcher and report the outcome as a UCI "info string". A load
    /// failure is reported but not fatal, the engine just keeps using whatever eval it had.</summary>
    private static void LoadNet(Searcher s, string path)
    {
        try { s.SetNnue(NnueNetwork.Load(Paths.InModels(path))); Console.WriteLine($"info string loaded net {path}"); }
        catch (Exception e) { Console.WriteLine($"info string failed to load net: {e.Message}"); }
    }

    private static void AutoLoadNet(Searcher s)
    {
        foreach (var name in new[] { "train/best.nnue", "train/latest.nnue" })
        {
            string full = Paths.InModels(name);
            if (!System.IO.File.Exists(full)) continue;
            try { s.SetNnue(NnueNetwork.Load(full)); Console.WriteLine($"info string auto-loaded net {full}"); return; }
            catch (Exception e) { Console.WriteLine($"info string failed to auto-load net {full}: {e.Message}"); }
        }
        Console.WriteLine("info string no net found under workspace/models/train, using hand-crafted eval");
    }

    /// <summary>
    /// Handle "setoption". Only EvalFile is supported: it points at an NNUE net to load.
    /// </summary>
    private static void HandleSetOption(Searcher s, string[] t)
    {
        // setoption name EvalFile value <path>
        // Locate the "name" and "value" keywords by position rather than assuming a fixed layout, since
        // the option name itself can be multiple tokens.
        int ni = Array.IndexOf(t, "name"), vi = Array.IndexOf(t, "value");
        if (ni < 0 || vi <= ni || vi + 1 >= t.Length) return;
        string name = string.Join(' ', t[(ni + 1)..vi]);
        string value = t[vi + 1];

        if (name.Equals("EvalFile", StringComparison.OrdinalIgnoreCase))
            LoadNet(s, value);
        else if (name.Equals("Hash", StringComparison.OrdinalIgnoreCase) && int.TryParse(value, out int mb))
            s.SetHashSize(Math.Clamp(mb, 1, 4096));
        // "Threads" is accepted and ignored (single-threaded search); see the uci handshake.
    }

    /// <summary>
    /// Handle "position": set the board to the start position or a given FEN, then optionally replay a
    /// list of moves after the "moves" keyword. This is how a GUI communicates the current game state,
    /// it sends the whole move history each time rather than incremental updates.
    /// </summary>
    private static void ParsePosition(Position pos, string[] t)
    {
        int idx;
        if (t.Length > 1 && t[1] == "startpos") { Position.Set(Types.DEFAULT_FEN, pos); idx = 2; }
        else if (t.Length > 1 && t[1] == "fen")
        {
            // A FEN is six space-separated fields, so rejoin tokens until we hit "moves" (or run out).
            var fen = new System.Text.StringBuilder();
            idx = 2;
            while (idx < t.Length && t[idx] != "moves") fen.Append(t[idx++]).Append(' ');
            Position.Set(fen.ToString().Trim(), pos);
        }
        else return;

        // Replay the move list (if any) on top of the position we just set.
        if (idx < t.Length && t[idx] == "moves")
            for (int i = idx + 1; i < t.Length; i++)
            {
                Move m = ParseMove(pos, t[i]);
                if (m.ToFrom == 0) break;
                pos.Play(pos.Turn, m);
            }
    }

    /// <summary>
    /// Resolve a UCI move string (e.g. "e2e4", "e7e8q") to a real <see cref="Move"/> by generating the
    /// legal moves and matching on their string form. Going through legal generation this way means we
    /// recover all the move's internal flags (capture, promotion, en passant, castling) for free, and
    /// implicitly reject anything illegal. Returns the default (empty) move if nothing matches.
    /// </summary>
    private static Move ParseMove(Position pos, string uci)
    {
        Span<Move> moves = stackalloc Move[256];
        int n = Searcher.GenerateLegal(pos, moves);
        for (int i = 0; i < n; i++)
            if (moves[i].ToString() == uci) return moves[i];
        return default;
    }

    /// <summary>
    /// Handle "go": translate the time-control words into <see cref="SearchLimits"/>, search, and print
    /// the chosen move. Supports fixed depth, fixed move-time, "infinite", and clock-based time controls
    /// (wtime/btime/winc/binc); with nothing specified it falls back to a fixed depth.
    /// </summary>
    private static void HandleGo(Position pos, Searcher searcher, string[] t)
    {
        var limits = new SearchLimits();
        long wtime = 0, btime = 0, winc = 0, binc = 0, movetime = 0;
        int depth = 0;
        for (int i = 1; i < t.Length - 1; i++)
        {
            switch (t[i])
            {
                case "depth": int.TryParse(t[i + 1], out depth); break;
                case "movetime": long.TryParse(t[i + 1], out movetime); break;
                case "wtime": long.TryParse(t[i + 1], out wtime); break;
                case "btime": long.TryParse(t[i + 1], out btime); break;
                case "winc": long.TryParse(t[i + 1], out winc); break;
                case "binc": long.TryParse(t[i + 1], out binc); break;
            }
        }
        // Pick a limit in priority order. Explicit instructions (infinite, movetime, depth) win;
        // otherwise derive a move-time budget from our remaining clock.
        if (Array.IndexOf(t, "infinite") >= 0) limits.Infinite = true;
        else if (movetime > 0) limits.MoveTimeMs = movetime;
        else if (depth > 0) limits.MaxDepth = depth;
        else if (wtime > 0 || btime > 0)
        {
            // Simple budget: spend ~1/20th of the time left plus half the increment, with a 10ms floor.
            // The /20 assumes roughly 20 more moves to play, a crude but robust default.
            long time = pos.Turn == Color.White ? wtime : btime;
            long inc = pos.Turn == Color.White ? winc : binc;
            limits.MoveTimeMs = Math.Max(10, time / 20 + inc / 2);
        }
        else limits.MaxDepth = 8;

        // !!Bug
        Move best = searcher.Think(pos, limits);
        Console.WriteLine($"bestmove {(best.ToFrom == 0 ? "0000" : best.ToString())}");
    }
}
