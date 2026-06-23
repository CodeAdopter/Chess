using Core;

namespace Engine.Nnue;

/// <summary>
/// Drives <see cref="NnueNetwork"/> over a <see cref="Position"/> using a full accumulator refresh.
/// The returned centipawns are the learned correction; callers that want the playable evaluation add
/// <see cref="Engine.Eval"/>. One instance per searcher (buffers are not shared).
/// </summary>
public sealed class NnueEvaluator
{
    private readonly NnueNetwork net;
    private readonly float[] accWhite;
    private readonly float[] accBlack;
    private readonly List<int> features = new(40);

    /// <summary>
    /// Binds the evaluator to a network and allocates its two per-perspective accumulator buffers.
    /// </summary>
    public NnueEvaluator(NnueNetwork net)
    {
        this.net = net;
        accWhite = new float[net.H];
        accBlack = new float[net.H];
    }

    /// <summary>
    /// Evaluates the position from scratch, feeding the head [side-to-move accumulator, other accumulator]. Returns the learned correction in centipawns (side-to-move POV).
    /// </summary>
    public int Evaluate(Position pos)
    {
        Refresh(pos, Color.White, accWhite);
        Refresh(pos, Color.Black, accBlack);
        return pos.Turn == Color.White
            ? net.Forward(accWhite, accBlack)
            : net.Forward(accBlack, accWhite);
    }

    /// <summary>
    /// Recompute one colour's accumulator from scratch: bias + sum of active feature columns.
    /// </summary>
    private void Refresh(Position pos, Color persp, float[] acc)
    {
        Array.Copy(net.L0Bias, acc, net.H);

        features.Clear();
        FeatureSet.ActiveFeatures(pos, persp, features);
        foreach (int f in features) net.AddFeature(acc, f);
    }
}
