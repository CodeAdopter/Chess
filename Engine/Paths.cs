namespace Engine;

/// <summary>
/// Central resolver for all generated I/O. Everything the engine reads or writes lives under a single
/// <c>workspace/</c> folder (data, models, logs) so generated artifacts never mix with source. The
/// workspace is found by walking up from the current directory (then the executable's directory), so
/// the CLI works whether launched from the repo root or from its build output.
/// </summary>
public static class Paths
{
    public static string Workspace { get; }
    public static string Data => Ensure(Path.Combine(Workspace, "data"));
    public static string Models => Ensure(Path.Combine(Workspace, "models"));
    public static string Logs => Ensure(Path.Combine(Workspace, "logs"));

    static Paths()
    {
        Workspace = Find() ?? Path.Combine(Environment.CurrentDirectory, "workspace");
        Directory.CreateDirectory(Workspace);
    }

    private static string? Find()
    {
        foreach (var start in new[] { Environment.CurrentDirectory, AppContext.BaseDirectory })
        {
            var dir = new DirectoryInfo(start);
            while (dir != null)
            {
                string candidate = Path.Combine(dir.FullName, "workspace");
                if (Directory.Exists(candidate)) return candidate;
                dir = dir.Parent;
            }
        }
        return null;
    }

    private static string Ensure(string path)
    {
        Directory.CreateDirectory(path);
        return path;
    }

    /// <summary>
    /// Resolve a user-supplied path: absolute as-is, otherwise relative to a workspace subfolder.
    /// </summary>
    public static string InData(string name) => Path.IsPathRooted(name) ? name : Path.Combine(Data, name);
    public static string InModels(string name) => Path.IsPathRooted(name) ? name : Path.Combine(Models, name);
}
