using Core;
using Engine;

namespace Cli;

/// <summary>
/// Interactive control center (shown when the CLI launches with no arguments). Pick a workflow, it
/// prompts for parameters (Enter accepts the default in brackets), then runs it with live output and
/// Ctrl-C to cancel. Everything here is also available as a one-shot command; see <see cref="PrintUsage"/>.
/// This is the human-facing front end. The actual work all lives in <see cref="Commands"/>.
/// </summary>
public static class Menu
{
    /// <summary>The main read-eval-print loop: draw the menu, read a choice, dispatch to a workflow,
    /// repeat until the user quits. Choices accept either the number or the command name.</summary>
    public static void Run()
    {
        while (true)
        {
            Console.WriteLine();
            Console.WriteLine("=== Chess Engine - control center ===");
            Console.WriteLine($"workspace: {Paths.Workspace}");
            Console.WriteLine();
            Console.WriteLine("   1) train       continuous self-improving training  (the main one)");
            Console.WriteLine("   2) perft       move-generation performance test");
            Console.WriteLine("   3) nps         search throughput (optionally with a net)");
            Console.WriteLine("   4) gen         generate a self-play dataset");
            Console.WriteLine("   5) trainfile   train a net on a fixed dataset");
            Console.WriteLine("   6) match       net vs hand-crafted eval");
            Console.WriteLine("   7) show        watch one net-vs-hand-crafted game");
            Console.WriteLine("   8) selfplay    one self-play game");
            Console.WriteLine("   9) nnueinit    create a random net");
            Console.WriteLine("  10) nnuetest    verify net plumbing");
            Console.WriteLine("  11) uci         UCI mode (for chess GUIs)");
            Console.WriteLine("   q) quit");
            Console.Write("\n> ");

            string? choice = Console.ReadLine()?.Trim().ToLowerInvariant();
            Console.WriteLine();
            switch (choice)
            {
                case "1": case "train": DoTrain(); break;
                case "2": case "perft": DoPerft(); break;
                case "3": case "nps": DoNps(); break;
                case "4": case "gen": DoGen(); break;
                case "5": case "trainfile": DoTrainFile(); break;
                case "6": case "match": DoMatch(); break;
                case "7": case "show": DoShow(); break;
                case "8": case "selfplay": Commands.SelfPlay(Commands.AskInt("depth", 6)); break;
                case "9": case "nnueinit": Commands.NnueInit(Commands.AskStr("output net", "net.nnue"), Commands.AskInt("H", 256)); break;
                case "10": case "nnuetest": Commands.NnueTest(Commands.AskStr("net", "net.nnue")); break;
                case "11": case "uci": Uci.Run(Commands.AskStr("net (blank = hand-crafted)", "")); break;
                case "q": case "quit": case "exit": return;
                default: Console.WriteLine("unknown choice."); break;
            }
        }
    }

    // ---- per-workflow prompt builders --------------------------------------------------------
    // Each Do* method asks for that workflow's parameters (with sensible defaults) and then hands off to
    // the matching Commands.* method. Kept separate from Run() only to keep the dispatch switch readable.

    private static void DoTrain() =>
        Commands.TrainLoop(
            Commands.AskInt("cores (more = faster iterations; 1 = lightweight background)", Math.Min(8, Environment.ProcessorCount)),
            Commands.AskInt("depth (4 = fastest gen, 6 = best labels)", 6),
            Commands.AskInt("iterations (0 = run until Ctrl-C)", 0));

    private static void DoPerft() =>
        Commands.Perft(Commands.AskInt("depth", Commands.PerftDepth), Commands.AskStr("fen", Types.DEFAULT_FEN));

    private static void DoNps()
    {
        string net = Commands.AskStr("net (blank = hand-crafted)", "");
        Commands.Nps(net == "" ? null : net, Commands.AskInt("depth", 8));
    }

    private static void DoGen() =>
        Commands.Gen(
            Commands.AskStr("output .trd", "selfplay.trd"),
            Commands.AskInt("games", Commands.GenGames),
            Commands.AskInt("depth", Commands.GenDepth),
            Commands.AskInt("random opening plies", Commands.GenRandomPlies),
            Commands.AskInt("seed", 1234),
            Commands.AskInt("threads (0 = all cores)", 0));

    private static void DoTrainFile() =>
        Commands.TrainFile(
            Commands.AskStr("data (file or dir under workspace/data)", "."),
            Commands.AskStr("output net", "net.nnue"),
            Commands.AskInt("epochs", Commands.TrainEpochs),
            Commands.AskInt("H (accumulator width)", Commands.TrainH),
            Commands.AskFloat("lr", Commands.TrainLr),
            Commands.AskFloat("lambda (eval vs result blend)", Commands.TrainLambda),
            Commands.AskInt("threads (0 = all cores)", 0));

    private static void DoMatch() =>
        Commands.Match(Commands.AskStr("net", "net.nnue"), Commands.AskInt("games", Commands.MatchGames), Commands.AskInt("depth", Commands.MatchDepth));

    private static void DoShow() =>
        Commands.Show(Commands.AskStr("net", "net.nnue"), Commands.AskInt("depth", 6));

    /// <summary>Print the one-shot command cheat-sheet. Shown for the menu's help and for an unrecognised
    /// command-line verb, so users can discover the scripting interface.</summary>
    public static void PrintUsage()
    {
        Console.WriteLine("""
            Chess Engine CLI

            Run with no arguments for the interactive control center, or one-shot:

              chess train     [cores] [depth] [iterations]  continuous self-improving training
              chess perft     [depth] [fen]
              chess perftsuite [runs]                       single-core movegen throughput (Mnps)
              chess perftscale [maxThreads]                 multi-core move-gen scaling sweep
              chess perftbulkscale [maxThreads]             multi-core bulk-count perft scaling sweep
              chess nps       [depth] | [net] [depth]
              chess gen       [out.trd] [games] [depth] [randomPlies] [seed] [threads]
              chess trainfile [data] [outNet] [epochs] [H] [lr] [lambda] [threads]
              chess match     [net] [games] [depth]
              chess show      [net] [depth]
              chess selfplay  [depth]
              chess nnueinit  [out] [H]
              chess nnuetest  [net]
              chess uci       [net]

            Paths are relative to workspace/data (datasets) or workspace/models (nets) unless absolute.
            """);
    }
}
