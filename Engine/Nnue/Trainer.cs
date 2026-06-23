using System.Diagnostics;
using Core;

namespace Engine.Nnue;

/// <summary>
/// File-based training entry point (the <c>train</c> command): load a dataset, build features, train a
/// fresh net from scratch, save it. The network learns a centipawn correction over the hand-crafted
/// evaluation, not the whole evaluation, so sparse HalfKP king buckets do not have to rediscover basic
/// material from self-play alone. The actual optimisation lives in <see cref="NnueTrainer"/>, which is
/// also what the continuous loop drives (warm-start), so both paths share identical math.
/// </summary>
public static class Trainer
{
    /// <summary>
    /// The centipawn-to-win-probability scale used by the sigmoid target, and also the net's output scale.
    /// </summary>
    public const float K = 300f;   // centipawn -> win-prob scale, also the net's output scale

    /// <summary>
    /// The <c>train</c> command end to end: load a dataset, build features and residual targets, train a
    /// fresh net for <paramref name="epochs"/> epochs, fold away the factoriser, save, and sanity-check.
    /// </summary>
    public static void Train(string dataPath, string netOut, int epochs, int h, float lr, float lambda,
        int threads = 0, CancellationToken ct = default)
    {
        threads = threads <= 0 ? Environment.ProcessorCount : threads;
        dataPath = Paths.InData(dataPath);
        netOut = Paths.InModels(netOut);

        var samples = TrainingData.ReadAll(dataPath);
        Console.WriteLine($"loaded {samples.Count} samples; building features (H={h}, lr={lr}, lambda={lambda}, threads={threads})...");
        var examples = BuildExamples(samples, lambda);

        var net = NnueNetwork.CreateRandom(h, 32, K, seed: 7);
        var trainer = new NnueTrainer(net, lr);
        var sw = Stopwatch.StartNew();
        trainer.Run(examples, epochs, threads, ct,
            (epoch, mse) => Console.WriteLine($"epoch {epoch + 1,3}/{epochs}: mse={mse:F5}  ({sw.Elapsed.TotalSeconds:F1}s)"));

        trainer.FoldVirtualIntoL0();   // collapse king-independent virtual weights into L0 → plain HalfKP net
        net.Save(netOut);
        Console.WriteLine($"\nsaved {netOut}");
        SanityCheck(net);
    }

    /// <summary>
    /// Pre-extract features and compute the residual win-prob target for each sample.
    /// </summary>
    public static Example[] BuildExamples(IReadOnlyList<Sample> samples, float lambda)
    {
        var list = new Example[samples.Count];
        var pos = new Position();
        var fw = new List<int>(40);
        var fb = new List<int>(40);
        for (int i = 0; i < samples.Count; i++)
        {
            var s = samples[i];
            Position.Set(s.Fen, pos);
            fw.Clear(); fb.Clear();
            FeatureSet.ActiveFeatures(pos, Color.White, fw);
            FeatureSet.ActiveFeatures(pos, Color.Black, fb);

            float wpEval = Sigmoid(s.ScoreCp / K);
            float wpRes = (s.Result + 1) / 2f;
            float targetAbs = lambda * wpEval + (1 - lambda) * wpRes;
            float desiredCp = K * Logit(ClampProbability(targetAbs));
            float residualCp = Math.Clamp(desiredCp - Eval.Evaluate(pos), -10000f, 10000f);
            float target = Sigmoid(residualCp / K);
            list[i] = new Example([.. fw], [.. fb], pos.Turn == Color.White, target);
        }
        return list;
    }

    /// <summary>
    /// Prints the full evaluation (hand-crafted plus NNUE correction) on a few obvious positions, so a trained net's output can be eyeballed for sanity (start near 0, a side up a queen strongly positive, etc.).
    /// </summary>
    public static void SanityCheck(NnueNetwork net)
    {
        var ev = new NnueEvaluator(net);
        var pos = new Position();
        Console.WriteLine("sanity (hand-crafted + NNUE correction):");
        Report(ev, pos, "start (~0)              ", Types.DEFAULT_FEN);
        Report(ev, pos, "white up a queen (>>0)  ", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -");
        Report(ev, pos, "white down a queen (<<0)", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq -");
        Report(ev, pos, "black up a queen (>>0)  ", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq -");
    }

    private static void Report(NnueEvaluator ev, Position pos, string label, string fen)
    {
        Position.Set(fen, pos);
        int correction = ev.Evaluate(pos);
        int full = Math.Clamp(Eval.Evaluate(pos) + correction, -10000, 10000);
        Console.WriteLine($"  {label}: {full,6} cp   (nnue {correction:+0;-0;0})");
    }

    // Sigmoid maps a (scaled) centipawn score to a win probability in (0, 1); Logit is its inverse. The
    // probability is clamped just inside (0, 1) before Logit so the log never sees 0 or 1 (which blow up).
    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
    private static float Logit(float p) => MathF.Log(p / (1f - p));
    private static float ClampProbability(float p) => Math.Clamp(p, 1e-4f, 1f - 1e-4f);
}
