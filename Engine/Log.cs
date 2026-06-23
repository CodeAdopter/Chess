using System.Text;

namespace Engine;

/// <summary>
/// Shared formatting helpers so every workflow reports throughput identically.
/// </summary>
public static class Log
{
    /// <summary>
    /// Format a nodes-per-second figure consistently across perft, search, and self-play.
    /// </summary>
    public static string Nps(long nodes, long ms)
    {
        double nps = ms > 0 ? nodes * 1000.0 / ms : nodes;
        return nps >= 1_000_000 ? $"{nps / 1e6:F2} Mnps" : $"{nps / 1e3:F0} knps";
    }
}

/// <summary>
/// Transparently mirrors all console output of a workflow run to a timestamped file under
/// <c>workspace/logs/</c>. Workflows just use <c>Console.WriteLine</c>; wrapping a run in
/// <c>using RunLog.Begin("train")</c> captures it to disk for later review without threading a logger
/// through every method.
/// </summary>
public sealed class RunLog : IDisposable
{
    private readonly TextWriter original;
    private readonly StreamWriter file;
    public string FilePath { get; }

    private RunLog(TextWriter original, StreamWriter file, string path)
    {
        this.original = original;
        this.file = file;
        FilePath = path;
    }

    public static RunLog Begin(string name)
    {
        string path = Path.Combine(Paths.Logs, $"{name}_{DateTime.Now:yyyyMMdd_HHmmss}.log");
        var file = new StreamWriter(path, append: false) { AutoFlush = true };
        var original = Console.Out;
        Console.SetOut(new TeeWriter(original, file));
        return new RunLog(original, file, path);
    }

    public void Dispose()
    {
        Console.SetOut(original);
        file.Dispose();
    }

    private sealed class TeeWriter(TextWriter a, TextWriter b) : TextWriter
    {
        public override Encoding Encoding => a.Encoding;
        public override void Write(char value) { a.Write(value); b.Write(value); }
        public override void Write(string? value) { a.Write(value); b.Write(value); }
        public override void WriteLine(string? value) { a.WriteLine(value); b.WriteLine(value); }
        public override void Flush() { a.Flush(); b.Flush(); }
    }
}
