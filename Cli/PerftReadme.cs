using System.Text;
using Engine;

namespace Cli;

/// <summary>
/// Autogenerates new perft results for the readme.md 
/// Requires the start and end tags to parse properly
/// </summary>
public static class PerftReadme
{
    public static void Update(int runs, CancellationToken ct)
    {
        string path = Path.Combine(Paths.Root, "README.md");
        if (!File.Exists(path)) { Console.WriteLine($"perftdoc: README not found at {path}"); return; }

        Console.WriteLine($"perftdoc: benchmarking best of {runs} to regenerate {path}");
        Console.WriteLine("  move generation (emit path the search uses)...");
        var searchRows = Perft.RunSuite(runs, bulk: false, ct);
        Console.WriteLine("  bulk count (popcount leaves, perft-only)...");
        var bulkRows = Perft.RunSuite(runs, bulk: true, ct);

        Report("search", searchRows);
        Report("bulk", bulkRows);
        if (!searchRows.TrueForAll(r => r.Ok) || !bulkRows.TrueForAll(r => r.Ok))
        {
            Console.WriteLine("  node-count MISMATCH, refusing to write the README.");
            return;
        }

        string text = File.ReadAllText(path);
        text = ReplaceSection(text, "search", BuildTable(searchRows));
        text = ReplaceSection(text, "bulk", BuildTable(bulkRows));
        File.WriteAllText(path, text);
        Console.WriteLine("  README updated.");
    }

    private static void Report(string label, List<Perft.SuiteResult> rows)
    {
        long tn = 0, tm = 0;
        foreach (var r in rows) { tn += r.Nodes; tm += r.Ms; }
        Console.WriteLine($"  {label,-7} total {tn,14:n0} nodes  {tm,5} ms  {Log.Nps(tn, tm)}");
    }

    private static string BuildTable(List<Perft.SuiteResult> rows)
    {
        var sb = new StringBuilder();
        sb.AppendLine("| Position | Depth | Nodes | NPS |");
        sb.AppendLine("|:--|--:|--:|--:|");
        long tn = 0, tm = 0;
        foreach (var r in rows)
        {
            sb.AppendLine($"| {r.Name} | {r.Depth} | {r.Nodes:n0} | {Log.Nps(r.Nodes, r.Ms)} |");
            tn += r.Nodes;
            tm += r.Ms;
        }
        sb.Append($"| **Total** | | **{tn:n0}** | **{Log.Nps(tn, tm)}** |");
        return sb.ToString();
    }

    private static string ReplaceSection(string text, string key, string table)
    {
        string begin = $"<!-- perftdoc:{key}:begin -->";
        string end = $"<!-- perftdoc:{key}:end -->";
        int i = text.IndexOf(begin, StringComparison.Ordinal);
        int j = text.IndexOf(end, StringComparison.Ordinal);
        if (i < 0 || j < 0 || j < i)
            throw new InvalidOperationException($"perftdoc: README is missing the '{begin} ... {end}' anchors.");
        return text[..(i + begin.Length)] + "\n" + table + "\n" + text[j..];
    }
}
