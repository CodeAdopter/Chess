// ---------------------------------------------------------------------------
// Program entry point: the CLI is the control center for the NNUE chess engine.
//
// Two ways in. Run with no arguments and you get the interactive menu (a human
// at a terminal). Run with arguments and the first one selects a single command
// to run and exit (for scripts, batch jobs, and CI). Both routes funnel into the
// same Commands.* workflows, so behaviour is identical either way.
//
// Before anything dispatches we do the one-time global setup: lock formatting to
// the invariant culture and build the engine's static lookup tables.
// ---------------------------------------------------------------------------

using System.Globalization;
using Core;
using Cli;
using Engine.Nnue;

// Uniform, locale-independent number formatting in all output (dots for decimals, no stray separators).
// Parsing and printing must not depend on the machine's regional settings, otherwise a comma-decimal
// locale would mangle datasets and logged numbers.
CultureInfo.CurrentCulture = CultureInfo.InvariantCulture;

// Build the move-generation lookup tables and Zobrist hash keys once, up front. Every command below
// (perft, search, training) assumes these are already initialised.
Tables.Init();
Zobrist.Init();

// No args → interactive control center. With args → run one command and exit (for scripting).
if (args.Length == 0)
{
    Menu.Run();
    return;
}

// Small typed-argument helpers: read positional arg i as int/float/string, falling back to a default
// when the argument is missing or doesn't parse. Parsing uses the invariant culture for the same
// locale-independence reason as above.
var inv = CultureInfo.InvariantCulture;
int I(int i, int def) => args.Length > i && int.TryParse(args[i], NumberStyles.Integer, inv, out var v) ? v : def;
float F(int i, float def) => args.Length > i && float.TryParse(args[i], NumberStyles.Float, inv, out var v) ? v : def;
string S(int i, string def) => args.Length > i ? args[i] : def;

// Command dispatch: args[0] is the verb, the rest are its parameters. Each case just maps the
// positional arguments onto a Commands.* workflow using the typed helpers and shared defaults.
switch (args[0].ToLowerInvariant())
{
    case "gen":
        Commands.Gen(S(1, "selfplay.trd"), I(2, Commands.GenGames), I(3, Commands.GenDepth), I(4, Commands.GenRandomPlies), I(5, 1234), I(6, 0));
        break;
    case "genbench":
        Commands.GenBench(S(1, "train/best.nnue"), I(2, 256), I(3, 4), I(4, Environment.ProcessorCount), I(5, 10), I(6, 1));
        break;
    case "train":
        Commands.TrainLoop(I(1, 0), I(2, 6), I(3, 0));
        break;
    case "trainfile":
        Commands.TrainFile(S(1, "."), S(2, "net.nnue"), I(3, Commands.TrainEpochs), I(4, Commands.TrainH), F(5, Commands.TrainLr), F(6, Commands.TrainLambda), I(7, 0));
        break;
    case "match":
        Commands.Match(S(1, "net.nnue"), I(2, Commands.MatchGames), I(3, Commands.MatchDepth));
        break;
    case "optmatch":
        Commands.OptMatch(S(1, "train/best.nnue"), I(2, 200), I(3, 4), S(4, "base"), S(5, "sound"));
        break;
    case "show":
        Commands.Show(S(1, "net.nnue"), I(2, 6));
        break;
    case "selfplay":
        Commands.SelfPlay(I(1, 6));
        break;
    case "perft":
        Commands.Perft(I(1, Commands.PerftDepth), args.Length > 2 ? string.Join(' ', args[2..]) : Types.DEFAULT_FEN);
        break;
    case "perftsuite":
        Commands.PerftSuite(I(1, 3));
        break;
    case "perftsuitebulk":
        Commands.PerftSuite(I(1, 3), bulk: true);
        break;
    case "perftdoc":
        Commands.PerftDoc(I(1, 5));
        break;
    case "perftscale":
        Commands.PerftScale(I(1, 0));
        break;
    case "perftbulkscale":
        Commands.PerftScale(I(1, 0), bulk: true);
        break;
    case "nps":
    {
        // The first argument after "nps" is optional and overloaded: a number means "depth only"
        // (hand-crafted eval), anything else is treated as a net name with depth in the next slot.
        string a1 = S(1, "");
        if (a1 == "" || int.TryParse(a1, out _)) Commands.Nps(null, I(1, 8));
        else Commands.Nps(a1, I(2, 8));
        break;
    }
    case "nnueinit":
        Commands.NnueInit(S(1, "net.nnue"), I(2, 256));
        break;
    case "nnuetest":
        Commands.NnueTest(S(1, "net.nnue"));
        break;
    case "nnueacctest":
        NnueTools.AccTest();
        break;
    case "quanttest":
        NnueTools.QuantTest(S(1, "net.nnue"));
        break;
    case "uci":
        Uci.Run(S(1, ""));
        break;
    default:
        // Unknown verb: print the usage cheat-sheet rather than failing silently.
        Menu.PrintUsage();
        break;
}
