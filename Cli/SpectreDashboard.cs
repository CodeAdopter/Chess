using Engine.Nnue;
using Spectre.Console;
using Spectre.Console.Rendering;

namespace Cli;

/// <summary>
/// Live training dashboard. Subscribes to the loop's <see cref="LoopEvents"/> and renders a panel that
/// refreshes in place: generation + teacher-phase header, the current stage with a progress bar + ETA,
/// buffer fill, and a rolling history of generation speed / train time / measured score. A single
/// refresher thread does all rendering (events only mutate shared state), so it's safe against the
/// parallel generation threads firing progress updates.
/// </summary>
public static class SpectreDashboard
{
    /// <summary>
    /// All the mutable display state, shared between the loop's event callbacks (which write it) and the
    /// refresher thread (which reads it to draw). The fields are deliberately plain. Only
    /// <see cref="History"/> is guarded by a lock because it is a collection that both sides touch.
    /// </summary>
    private sealed class State
    {
        public int Cores, Generation;
        public bool NetTeacher;
        public string Stage = "starting", Info = "";
        public int Done, Total;            // progress within the current stage
        public DateTime StageStart = DateTime.Now;  // when the current stage began, used for ETA
        public long Buffer;                // positions currently in the training buffer
        public readonly List<IterReport> History = [];  // recent per-generation reports for the table
    }

    /// <summary>
    /// Entry point for the dashboard. Picks the live (in-place, ANSI) renderer when attached to a real
    /// terminal, or a plain line-by-line renderer when output is redirected (background runs, CI, log
    /// capture), then runs the training loop and feeds its events to whichever renderer was chosen.
    /// </summary>
    public static void Run(LoopConfig cfg, CancellationToken ct, TextWriter log)
    {
        // The live display needs a real terminal (cursor control). When output is redirected/piped
        // (background runs, CI, capture), fall back to plain line output.
        if (Console.IsOutputRedirected || !AnsiConsole.Profile.Capabilities.Ansi)
        {
            RunPlain(cfg, ct, log);
            return;
        }
        RunLive(cfg, ct, log);
    }

    /// <summary>Format one finished generation as a single log/console line. Used by both renderers and
    /// written verbatim to the log file so a run is reconstructable from the log alone.</summary>
    private static string Summary(IterReport r) =>
        $"gen {r.Generation,4} | buf {r.BufferPositions,9:n0} | gen {r.GenSec,5:F1}s ({r.GamesPerSec:F0} g/s) | " +
        $"train {r.TrainSec,5:F1}s | mse {r.Mse:F4}" +
        (r.Score is double sc ? $" | vs-anchor {sc:P0}" : "") + (r.NetTeacher ? " | [net teacher]" : "");

    /// <summary>The redirected-output fallback: no cursor tricks, just print stage changes, occasional
    /// progress, and a summary line per generation. Safe to pipe to a file.</summary>
    private static void RunPlain(LoopConfig cfg, CancellationToken ct, TextWriter log)
    {
        var events = new LoopEvents
        {
            Info = m => Console.WriteLine($"  - {m}"),
            Stage = st => Console.WriteLine($"  [{st}]"),
            // Throttle progress to roughly the quarter marks so a piped log doesn't fill with noise.
            Progress = (d, t) => { if (t > 0 && d % Math.Max(1, t / 4) == 0) Console.WriteLine($"    {d}/{t}"); },
            Iteration = r => { string line = Summary(r); Console.WriteLine(line); log.WriteLine(line); },
        };
        new ContinuousLoop(cfg, events).Run(ct);
    }

    /// <summary>
    /// The interactive renderer. One background "refresher" task redraws the panel from shared
    /// <see cref="State"/> at a fixed cadence; the loop's event callbacks only mutate that state. Keeping
    /// all drawing on a single thread means the parallel generation threads firing progress events can
    /// never race on the console.
    /// </summary>
    private static void RunLive(LoopConfig cfg, CancellationToken ct, TextWriter log)
    {
        var s = new State { Cores = cfg.Cores };
        bool finished = false;

        AnsiConsole.Live(Render(s)).AutoClear(false).Overflow(VerticalOverflow.Ellipsis).Start(ctx =>
        {
            // Redraw ~5x/second from the latest state. A render can briefly race a state mutation; that
            // just yields a slightly stale frame, so the exception is swallowed and we redraw next tick.
            var refresher = Task.Run(() =>
            {
                while (!Volatile.Read(ref finished))
                {
                    try { ctx.UpdateTarget(Render(s)); } catch { /* transient render race */ }
                    Thread.Sleep(200);
                }
            });

            // Event callbacks only update shared state. The refresher above turns that state into frames.
            var events = new LoopEvents
            {
                Info = m => s.Info = m,
                Stage = st => { s.Stage = st; s.StageStart = DateTime.Now; s.Done = 0; s.Total = 0; },
                Progress = (d, t) => { s.Done = d; s.Total = t; },
                Iteration = r =>
                {
                    // Keep only the last dozen generations so the history table stays a fixed height.
                    lock (s.History) { s.History.Add(r); if (s.History.Count > 12) s.History.RemoveAt(0); }
                    s.Generation = r.Generation; s.Buffer = r.BufferPositions; s.NetTeacher = r.NetTeacher;
                    log.WriteLine(Summary(r));
                },
            };

            try { new ContinuousLoop(cfg, events).Run(ct); }
            finally
            {
                // Stop the refresher, wait for it to exit, then draw one final frame so the last state is
                // what stays on screen after the live display releases the terminal.
                Volatile.Write(ref finished, true);
                refresher.Wait();
                ctx.UpdateTarget(Render(s));
            }
        });
    }

    /// <summary>Build the full dashboard panel (header, current stage with progress bar, buffer fill,
    /// optional info line, and the rolling history table) from a snapshot of the shared state.</summary>
    private static IRenderable Render(State s)
    {
        string teacher = s.NetTeacher ? "[lime]net teacher[/]" : "[grey]hand-crafted teacher[/]";
        var header = new Markup($"[bold yellow]Chess Trainer[/]    generation [aqua]{s.Generation}[/]    {s.Cores} cores    {teacher}");

        var rows = new List<IRenderable>
        {
            header,
            new Rule().RuleStyle("grey"),
            new Markup(StageLine(s)),
            new Markup($"buffer: [green]{s.Buffer:n0}[/] positions"),
        };
        if (!string.IsNullOrEmpty(s.Info)) rows.Add(new Markup($"[grey]{Markup.Escape(s.Info)}[/]"));

        rows.Add(new Rule("[grey]history[/]").RuleStyle("grey").LeftJustified());
        var table = new Table().Border(TableBorder.Minimal).BorderColor(Color.Grey);
        table.AddColumns("[grey]gen[/]", "[grey]gen speed[/]", "[grey]train[/]", "[grey]mse[/]", "[grey]vs anchor[/]");
        lock (s.History)
            foreach (var r in s.History)
                table.AddRow(r.Generation.ToString(), $"{r.GamesPerSec:F0} g/s", $"{r.TrainSec:F1}s", $"{r.Mse:F4}",
                             r.Score is double sc ? $"{sc:P0}" : "-");
        rows.Add(table);

        return new Panel(new Rows(rows)).Expand().BorderColor(Color.Grey);
    }

    /// <summary>Render the "stage" line: the stage name plus, when the stage reports a total, a text
    /// progress bar and a linear ETA extrapolated from elapsed time and fraction done.</summary>
    private static string StageLine(State s)
    {
        string bar = "", eta = "";
        if (s.Total > 0)
        {
            const int width = 24;
            int filled = Math.Clamp((int)Math.Round(width * (double)s.Done / s.Total), 0, width);
            bar = $" [green]{new string('#', filled)}[/][grey]{new string('-', width - filled)}[/] {s.Done}/{s.Total}";
            // ETA assumes the remaining work takes as long per item as the work done so far.
            double el = (DateTime.Now - s.StageStart).TotalSeconds;
            if (s.Done > 0 && el > 0)
                eta = $"  ETA {TimeSpan.FromSeconds((s.Total - s.Done) * el / s.Done):m\\:ss}";
        }
        return $"stage: [aqua]{s.Stage,-10}[/]{bar}{eta}";
    }
}
