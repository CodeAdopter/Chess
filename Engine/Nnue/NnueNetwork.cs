using System.Numerics;

namespace Engine.Nnue;

/// <summary>
/// Our own float32 NNUE (Efficiently Updatable Neural Network): a HalfKP feature transformer
/// (40960 inputs to H values per perspective) followed by a small dense head
/// ([2H] to CReLU to L1Out to CReLU to 1). Weights live in a self-describing binary file that the C#
/// trainer (Stage 2) writes and this loads, with no external (Stockfish) format and no native dependency,
/// so the whole net travels into a WASM build unchanged. float32 comes first for correctness; int16/int8
/// quantisation (see <see cref="QuantizedNetwork"/>) is a later optimisation for native speed and a
/// smaller browser payload.
/// </summary>
public sealed class NnueNetwork
{
    /// <summary>
    /// File-format identifier ("NNU2") written at the start of every saved net, checked on load.
    /// </summary>
    public const int Magic = 0x554E4E32; // "NNU2"

    /// <summary>
    /// Accumulator width per perspective (the feature transformer's output size).
    /// </summary>
    public int H { get; private set; }          // accumulator width per perspective
    /// <summary>
    /// Hidden width of the dense head's first layer.
    /// </summary>
    public int L1Out { get; private set; }       // hidden head width
    /// <summary>
    /// Scale factor mapping the network's raw scalar output to centipawns.
    /// </summary>
    public float OutScale { get; private set; }   // maps the raw scalar to centipawns

    // Feature transformer: column-major by feature for cheap accumulation (L0[feature*H + h]).
    public float[] L0Weights = default!;          // [NumFeatures * H]
    public float[] L0Bias = default!;             // [H]

    // Head layer 1: input is [accStm | accNonStm] of width 2H. L1[out*2H + in].
    public float[] L1Weights = default!;          // [L1Out * 2H]
    public float[] L1Bias = default!;             // [L1Out]

    // Head layer 2: L1Out → 1.
    public float[] L2Weights = default!;          // [L1Out]
    public float L2Bias;

    /// <summary>
    /// Creates a freshly-initialised random network of the given shape (used as the starting point for training a net from scratch).
    /// </summary>
    public static NnueNetwork CreateRandom(int h = 256, int l1Out = 32, float outScale = 400f, int seed = 1)
    {
        var net = new NnueNetwork { H = h, L1Out = l1Out, OutScale = outScale };
        var rng = new Random(seed);
        net.L0Weights = RandArray(rng, FeatureSet.NumFeatures * h, 0.02f);
        // Start accumulator activations in the middle of CReLU's live band [0,1]. With a zero bias the
        // accumulator (bias + ~30 tiny feature columns) centres on 0, so ~half the units sit pinned at the
        // 0 clamp where the gradient is gated off (dead at init, dead forever). A 0.5 bias keeps them live.
        net.L0Bias = new float[h];
        Array.Fill(net.L0Bias, 0.5f);
        net.L1Weights = RandArray(rng, l1Out * 2 * h, 0.1f);
        net.L1Bias = new float[l1Out];
        net.L2Weights = RandArray(rng, l1Out, 0.1f);
        net.L2Bias = 0f;
        return net;
    }

    /// <summary>
    /// Deep copy, used to snapshot the best net for net-vs-net gating without sharing weight arrays.
    /// </summary>
    public NnueNetwork Clone() => new()
    {
        H = H,
        L1Out = L1Out,
        OutScale = OutScale,
        L0Weights = (float[])L0Weights.Clone(),
        L0Bias = (float[])L0Bias.Clone(),
        L1Weights = (float[])L1Weights.Clone(),
        L1Bias = (float[])L1Bias.Clone(),
        L2Weights = (float[])L2Weights.Clone(),
        L2Bias = L2Bias,
    };

    private static float[] RandArray(Random rng, int n, float scale)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        return a;
    }

    /// <summary>
    /// Run the dense head over two accumulators (SIMD). Returns centipawns (side-to-move POV).
    /// </summary>
    public int Forward(float[] accStm, float[] accNonStm)
    {
        int twoH = 2 * H, vw = Vector<float>.Count;
        Span<float> x = twoH <= 2048 ? stackalloc float[twoH] : new float[twoH];
        ClampInto(accStm, x, 0, H, vw);          // CReLU(accStm) once, not per output
        ClampInto(accNonStm, x, H, H, vw);

        float outp = L2Bias;
        for (int o = 0; o < L1Out; o++)
        {
            int wbase = o * twoH;
            var vsum = Vector<float>.Zero;
            int i = 0;
            for (; i <= twoH - vw; i += vw)
                vsum += new Vector<float>(x.Slice(i, vw)) * new Vector<float>(L1Weights, wbase + i);
            float sum = L1Bias[o] + Vector.Sum(vsum);
            for (; i < twoH; i++) sum += x[i] * L1Weights[wbase + i];
            outp += CReLU(sum) * L2Weights[o];
        }

        return Math.Clamp((int)(outp * OutScale), -10000, 10000);
    }

    /// <summary>
    /// acc += L0 column for feature f (vectorised). Used by the accumulator refresh/updates.
    /// </summary>
    public void AddFeature(float[] acc, int f)
    {
        int bse = f * H, vw = Vector<float>.Count, i = 0;
        for (; i <= H - vw; i += vw)
            (new Vector<float>(acc, i) + new Vector<float>(L0Weights, bse + i)).CopyTo(acc, i);
        for (; i < H; i++) acc[i] += L0Weights[bse + i];
    }

    /// <summary>
    /// acc -= L0 column for feature f (vectorised).
    /// </summary>
    public void SubFeature(float[] acc, int f)
    {
        int bse = f * H, vw = Vector<float>.Count, i = 0;
        for (; i <= H - vw; i += vw)
            (new Vector<float>(acc, i) - new Vector<float>(L0Weights, bse + i)).CopyTo(acc, i);
        for (; i < H; i++) acc[i] -= L0Weights[bse + i];
    }

    private static void ClampInto(float[] src, Span<float> dst, int off, int n, int vw)
    {
        var zero = Vector<float>.Zero;
        var one = new Vector<float>(1f);
        int i = 0;
        for (; i <= n - vw; i += vw)
            Vector.Min(Vector.Max(new Vector<float>(src, i), zero), one).CopyTo(dst.Slice(off + i, vw));
        for (; i < n; i++) dst[off + i] = CReLU(src[i]);
    }

    /// <summary>
    /// Clamped ReLU: squashes an activation into [0, 1]. The bounded range is what makes later int8/int16 quantisation possible.
    /// </summary>
    private static float CReLU(float x) => x < 0f ? 0f : (x > 1f ? 1f : x);

    /// <summary>
    /// Writes the network to a binary "NNU2" file (magic, shape, then each weight array as raw float bytes).
    /// </summary>
    public void Save(string path)
    {
        using var fs = File.Create(path);
        using var w = new BinaryWriter(fs);
        w.Write(Magic);
        w.Write(H);
        w.Write(L1Out);
        w.Write(OutScale);
        WriteArray(w, L0Weights);
        WriteArray(w, L0Bias);
        WriteArray(w, L1Weights);
        WriteArray(w, L1Bias);
        WriteArray(w, L2Weights);
        w.Write(L2Bias);
    }

    /// <summary>
    /// Reads a network back from an "NNU2" file produced by <see cref="Save"/>. Throws if the magic or length does not match.
    /// </summary>
    public static NnueNetwork Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var r = new BinaryReader(fs);
        if (r.ReadInt32() != Magic) throw new InvalidDataException("not an NNU2 file");
        var net = new NnueNetwork { H = r.ReadInt32(), L1Out = r.ReadInt32(), OutScale = r.ReadSingle() };
        net.L0Weights = ReadArray(r, FeatureSet.NumFeatures * net.H);
        net.L0Bias = ReadArray(r, net.H);
        net.L1Weights = ReadArray(r, net.L1Out * 2 * net.H);
        net.L1Bias = ReadArray(r, net.L1Out);
        net.L2Weights = ReadArray(r, net.L1Out);
        net.L2Bias = r.ReadSingle();
        return net;
    }

    private static void WriteArray(BinaryWriter w, float[] a)
    {
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan());
        w.Write(bytes);
    }

    private static float[] ReadArray(BinaryReader r, int n)
    {
        var a = new float[n];
        var bytes = System.Runtime.InteropServices.MemoryMarshal.AsBytes(a.AsSpan());
        int read = r.Read(bytes);
        if (read != bytes.Length) throw new EndOfStreamException("truncated NNU2 file");
        return a;
    }
}
