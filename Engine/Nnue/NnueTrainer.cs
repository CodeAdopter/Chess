using System.Collections.Concurrent;

namespace Engine.Nnue;

/// <summary>
/// One labelled training example with features pre-extracted for both perspectives.
/// </summary>
public readonly struct Example
{
    public readonly int[] FeatsWhite;
    public readonly int[] FeatsBlack;
    public readonly bool StmWhite;
    public readonly float Target;   // win prob in [0,1] for the NNUE correction, side-to-move POV
    public Example(int[] fw, int[] fb, bool stm, float t) { FeatsWhite = fw; FeatsBlack = fb; StmWhite = stm; Target = t; }
}

/// <summary>
/// Stateful NNUE trainer: owns a net plus its Adam moment buffers and step counter, so training can be
/// resumed and continued (warm-start) across many calls, which is the basis of the continuous loop.
/// <see cref="Run"/> does Hogwild-parallel Adam epochs over a set of examples, updating the net in place
/// (lock-free: threads race on the shared weights, and the small amount of lost work is a net speed win).
/// The file-based <see cref="Trainer.Train"/> is a thin wrapper that creates one of these, runs it, and saves.
/// </summary>
public sealed class NnueTrainer
{
    private const float B1 = 0.9f, B2 = 0.999f;
    // Surrogate gradient slope for CReLU outside its live (0,1) band. A true clipped-ReLU derivative is 0
    // when saturated, which traps any unit pinned at 0 or 1 (no gradient can move it back). A small leak
    // lets saturated units recover, the clipped-ReLU analogue of leaky ReLU.
    private const float Leak = 0.01f;
    private const float WClip = 1.98f;

    public NnueNetwork Net { get; }
    public float Lr { get; set; }

    private readonly int H, twoH, L1;
    private readonly float[] mL0, vL0, mb0, vb0, mW1, vW1, mb1, vb1, mW2, vW2, mL2b, vL2b, L2b;
    // Feature factoriser: a king-INDEPENDENT virtual weight per (piece,square), shared across all 64 king
    // buckets. A HalfKP index is f = king*640 + kind*64 + sq, so the virtual index is just f % VF. Every
    // king bucket of the same (piece,square) trains the same virtual column → ~64× the gradient signal, so
    // the net learns common piece-square structure from far less data instead of relearning it per bucket.
    // Folded into L0 at save (FoldVirtualIntoL0), so inference stays plain HalfKP with zero extra cost.
    private readonly int VF;
    private readonly float[] V, mV, vV;
    private long t;

    public NnueTrainer(NnueNetwork net, float lr)
    {
        Net = net;
        Lr = lr;
        H = net.H; twoH = 2 * H; L1 = net.L1Out;
        mL0 = new float[net.L0Weights.Length]; vL0 = new float[net.L0Weights.Length];
        mb0 = new float[H]; vb0 = new float[H];
        mW1 = new float[net.L1Weights.Length]; vW1 = new float[net.L1Weights.Length];
        mb1 = new float[L1]; vb1 = new float[L1];
        mW2 = new float[L1]; vW2 = new float[L1];
        mL2b = new float[1]; vL2b = new float[1];
        L2b = [net.L2Bias];
        VF = FeatureSet.PieceKinds * 64;        // 640 king-independent (piece,square) virtual features
        V = new float[VF * H]; mV = new float[V.Length]; vV = new float[V.Length];
    }

    /// <summary>
    /// Per-thread scratch buffers (accumulators, activations, gradients) so a worker allocates nothing inside the training loop.
    /// </summary>
    private sealed class Buf
    {
        public readonly float[] AccW, AccB, X, Z1, A1, Dx, DStm, DNon;
        public double Loss;
        public Buf(int h, int l1, int twoH)
        {
            AccW = new float[h]; AccB = new float[h];
            X = new float[twoH]; Z1 = new float[l1]; A1 = new float[l1];
            Dx = new float[twoH]; DStm = new float[h]; DNon = new float[h];
        }
    }

    /// <summary>
    /// Run Hogwild-parallel Adam epochs over <paramref name="examples"/>, updating the net in place.
    /// </summary>
    public void Run(Example[] examples, int epochs, int threads, CancellationToken ct, Action<int, double>? onEpoch = null)
    {
        threads = threads <= 0 ? Environment.ProcessorCount : threads;
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        var rng = new Random(99);

        for (int epoch = 0; epoch < epochs; epoch++)
        {
            if (ct.IsCancellationRequested) break;
            Shuffle(examples, rng);

            double lossSum = 0;
            object lossLock = new();
            Parallel.ForEach(Partitioner.Create(0, examples.Length), options,
                () => new Buf(H, L1, twoH),
                (range, _, buf) =>
                {
                    for (int idx = range.Item1; idx < range.Item2; idx++)
                        buf.Loss += Step(in examples[idx], buf);
                    return buf;
                },
                buf => { lock (lossLock) lossSum += buf.Loss; });

            Net.L2Bias = L2b[0];
            onEpoch?.Invoke(epoch, lossSum / Math.Max(1, examples.Length));
        }
        Net.L2Bias = L2b[0];
    }

    /// <summary>
    /// Train on <paramref name="budget"/> samples drawn from <paramref name="pool"/> (a partial epoch when
    /// budget &lt; pool size). Used by the continuous loop to keep per-position reuse bounded. Returns mean loss.
    /// </summary>
    public double RunSamples(Example[] pool, long budget, int threads, CancellationToken ct)
    {
        if (pool.Length == 0 || budget <= 0) return 0;
        threads = threads <= 0 ? Environment.ProcessorCount : threads;
        var options = new ParallelOptions { MaxDegreeOfParallelism = threads };
        Shuffle(pool, new Random(unchecked((int)t)));     // fresh order each call; vary by step count
        long len = pool.Length;

        double lossSum = 0;
        object lossLock = new();
        Parallel.ForEach(Partitioner.Create(0L, budget), options,
            () => new Buf(H, L1, twoH),
            (range, _, buf) =>
            {
                for (long k = range.Item1; k < range.Item2; k++)
                    buf.Loss += Step(in pool[(int)(k % len)], buf);
                return buf;
            },
            buf => { lock (lossLock) lossSum += buf.Loss; });

        Net.L2Bias = L2b[0];
        return lossSum / budget;
    }

    /// <summary>
    /// One training step on a single example: forward pass (accumulators, both CReLU layers, sigmoid), then
    /// backpropagation with an Adam update applied directly to every weight it touches. Both the real L0
    /// columns and their shared virtual (factoriser) columns are updated. Returns the squared error.
    /// </summary>
    private float Step(in Example ex, Buf buf)
    {
        float[] L0 = Net.L0Weights, b0 = Net.L0Bias, W1 = Net.L1Weights, b1 = Net.L1Bias, W2 = Net.L2Weights;
        float[] accW = buf.AccW, accB = buf.AccB, x = buf.X, z1 = buf.Z1, a1 = buf.A1, dx = buf.Dx;

        Array.Copy(b0, accW, H);
        Array.Copy(b0, accB, H);
        foreach (int f in ex.FeatsWhite) { int bse = f * H, vb = (f % VF) * H; for (int i = 0; i < H; i++) accW[i] += L0[bse + i] + V[vb + i]; }
        foreach (int f in ex.FeatsBlack) { int bse = f * H, vb = (f % VF) * H; for (int i = 0; i < H; i++) accB[i] += L0[bse + i] + V[vb + i]; }

        float[] accStm = ex.StmWhite ? accW : accB;
        float[] accNon = ex.StmWhite ? accB : accW;
        for (int i = 0; i < H; i++) { x[i] = CReLU(accStm[i]); x[H + i] = CReLU(accNon[i]); }

        for (int o = 0; o < L1; o++)
        {
            float s = b1[o]; int bse = o * twoH;
            for (int i = 0; i < twoH; i++) s += x[i] * W1[bse + i];
            z1[o] = s; a1[o] = CReLU(s);
        }
        float r = L2b[0];
        for (int o = 0; o < L1; o++) r += a1[o] * W2[o];
        float p = Sigmoid(r);
        float err = p - ex.Target;

        long ti = Interlocked.Increment(ref t);
        float bc1 = ti < 2000 ? 1f - MathF.Pow(B1, ti) : 1f;
        float bc2 = ti < 30000 ? 1f - MathF.Pow(B2, ti) : 1f;
        float gr = 2f * err * p * (1f - p);

        Adam(ref L2b[0], ref mL2b[0], ref vL2b[0], gr, bc1, bc2);

        Array.Clear(dx, 0, twoH);
        for (int o = 0; o < L1; o++)
        {
            float da1 = gr * W2[o];
            Adam(ref W2[o], ref mW2[o], ref vW2[o], gr * a1[o], bc1, bc2);
            W2[o] = Math.Clamp(W2[o], -WClip, WClip);
            float dz1 = (z1[o] > 0f && z1[o] < 1f) ? da1 : Leak * da1;
            if (dz1 != 0f)
            {
                int bse = o * twoH;
                Adam(ref b1[o], ref mb1[o], ref vb1[o], dz1, bc1, bc2);
                for (int i = 0; i < twoH; i++)
                {
                    dx[i] += dz1 * W1[bse + i];                            // backprop uses pre-update W1
                    Adam(ref W1[bse + i], ref mW1[bse + i], ref vW1[bse + i], dz1 * x[i], bc1, bc2);
                    W1[bse + i] = Math.Clamp(W1[bse + i], -WClip, WClip);
                }
            }
        }

        float[] dStm = buf.DStm, dNon = buf.DNon;
        for (int i = 0; i < H; i++)
        {
            dStm[i] = (accStm[i] > 0f && accStm[i] < 1f) ? dx[i] : Leak * dx[i];
            dNon[i] = (accNon[i] > 0f && accNon[i] < 1f) ? dx[H + i] : Leak * dx[H + i];
        }
        float[] dAccW = ex.StmWhite ? dStm : dNon;
        float[] dAccB = ex.StmWhite ? dNon : dStm;

        for (int i = 0; i < H; i++) Adam(ref b0[i], ref mb0[i], ref vb0[i], dAccW[i] + dAccB[i], bc1, bc2);
        foreach (int f in ex.FeatsWhite) { int bse = f * H, vb = (f % VF) * H; for (int i = 0; i < H; i++) { Adam(ref L0[bse + i], ref mL0[bse + i], ref vL0[bse + i], dAccW[i], bc1, bc2); Adam(ref V[vb + i], ref mV[vb + i], ref vV[vb + i], dAccW[i], bc1, bc2); } }
        foreach (int f in ex.FeatsBlack) { int bse = f * H, vb = (f % VF) * H; for (int i = 0; i < H; i++) { Adam(ref L0[bse + i], ref mL0[bse + i], ref vL0[bse + i], dAccB[i], bc1, bc2); Adam(ref V[vb + i], ref mV[vb + i], ref vV[vb + i], dAccB[i], bc1, bc2); } }

        return err * err;
    }

    /// <summary>
    /// Collapse the factorised virtual (king-independent) weights into the real HalfKP columns:
    /// L0[king,kind,sq] += V[kind,sq] for every feature. Afterwards the net is a plain HalfKP net (inference
    /// needs no knowledge of the factoriser), and V (with its Adam moments) is reset so the next training
    /// stretch re-learns a fresh shared component on top of the now-consolidated L0. Call before saving or
    /// using the net for play.
    /// </summary>
    public void FoldVirtualIntoL0()
    {
        var L0 = Net.L0Weights;
        for (int f = 0; f < FeatureSet.NumFeatures; f++)
        {
            int bse = f * H, vb = (f % VF) * H;
            for (int i = 0; i < H; i++) L0[bse + i] += V[vb + i];
        }
        Array.Clear(V); Array.Clear(mV); Array.Clear(vV);
    }

    /// <summary>
    /// One Adam optimiser update of a single weight <paramref name="p"/> given gradient <paramref name="g"/>: update the running mean (<paramref name="m"/>) and variance (<paramref name="v"/>) of the gradient, bias-correct them, and step.
    /// </summary>
    private void Adam(ref float p, ref float m, ref float v, float g, float bc1, float bc2)
    {
        m = 0.9f * m + 0.1f * g;
        v = 0.999f * v + 0.001f * g * g;
        p -= Lr * (m / bc1) / (MathF.Sqrt(v / bc2) + 1e-8f);
    }

    private static void Shuffle(Example[] a, Random rng)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }

    private static float CReLU(float v) => v < 0f ? 0f : (v > 1f ? 1f : v);
    private static float Sigmoid(float v) => 1f / (1f + MathF.Exp(-v));
}
