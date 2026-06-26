using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using Core;

namespace Engine.Nnue;

/// <summary>
/// int8/int16 quantisation of a trained float <see cref="NnueNetwork"/> for fast inference. The feature
/// transformer (L0) and accumulator are int16 at activation-scale <see cref="QA"/>; the dense head (L1, L2)
/// uses int8 weights at scale <see cref="QW"/> with int32 accumulation. CReLU activations live in [0, QA].
///
/// Quantising the eval is the lever that recovers most of the ~6× speed gap vs the hand-crafted eval once
/// the net is the self-play teacher (the eval is ~83% of per-node search time there). This scalar forward
/// is the correctness reference; the SIMD (AVX-VNNI / AVX2) fast path is validated to match it exactly.
///
/// Quantisation maths (QA=127, QW=64): a quantised activation is round(a·QA) for a∈[0,1]; a quantised weight is
/// round(w·QW). The accumulator stores round(acc·QA) directly, so L0/L0Bias scale by QA. An L1 pre-activation
/// s = QA·QW·z, so its CReLU output is clamp(s ≫ log2(QW), 0, QA); biases scale by QA·QW. The final scalar
/// out = QA·QW·raw, so centipawns = out·OutScale / (QA·QW).
///
/// Head weights are int16, not int8: a residual net trained with a [0,1] CReLU drives the head into a
/// large-weight regime (|w| up to ~175), which int8 (±127, any scale) cannot represent without ruinous
/// clipping. int16 holds them losslessly and still vectorises ~2× via AVX2 madd. (int8 + AVX-VNNI would need
/// quantisation-aware training that clamps head weights into the int8 range; deferred to a quant-aware retrain.)
/// </summary>
public sealed class QuantizedNetwork
{
    public const int QA = 127;   // activation scale: CReLU float [0,1] → int [0, QA]
    public const int QW = 64;    // head weight scale: float w → round(w·QW) (int16)
    public const int Shift = 6;  // log2(QW): L1 requantisation right-shift

    public readonly int H, L1Out, TwoH;
    public readonly float OutScale;

    public readonly short[] L0;      // [NumFeatures · H]   scale QA
    public readonly short[] L0Bias;  // [H]                 scale QA
    public readonly short[] L1;      // [L1Out · 2H]        scale QW
    public readonly int[] L1Bias;    // [L1Out]             scale QA·QW
    public readonly short[] L2;      // [L1Out]             scale QW
    public readonly int L2Bias;      //                     scale QA·QW

    // Diagnostics from quantisation: how many weights saturated the int8 range (signals QW too large).
    public int L1Clipped { get; private set; }
    public int L2Clipped { get; private set; }
    public float MaxAbsL1 { get; private set; }
    public float MaxAbsL2 { get; private set; }

    /// <summary>
    /// Quantises a trained float network into this fixed-point form, recording how many head weights saturated (a signal that the weight scale <see cref="QW"/> is too large).
    /// </summary>
    public QuantizedNetwork(NnueNetwork net)
    {
        H = net.H; L1Out = net.L1Out; TwoH = 2 * H; OutScale = net.OutScale;
        L0 = Q16(net.L0Weights, QA, out _, out _);
        L0Bias = Q16(net.L0Bias, QA, out _, out _);
        L1 = Q16(net.L1Weights, QW, out int c1, out float m1); L1Clipped = c1; MaxAbsL1 = m1;
        L2 = Q16(net.L2Weights, QW, out int c2, out float m2); L2Clipped = c2; MaxAbsL2 = m2;
        L1Bias = Qb(net.L1Bias, QA * QW);
        L2Bias = (int)MathF.Round(net.L2Bias * (QA * QW));
    }

    private static short[] Q16(float[] w, int scale, out int clipped, out float maxAbs)
    {
        var a = new short[w.Length];
        clipped = 0; maxAbs = 0f;
        for (int i = 0; i < w.Length; i++)
        {
            maxAbs = MathF.Max(maxAbs, MathF.Abs(w[i]));
            int q = (int)MathF.Round(w[i] * scale);
            if (q is > short.MaxValue or < short.MinValue) clipped++;
            a[i] = (short)Math.Clamp(q, short.MinValue, short.MaxValue);
        }
        return a;
    }

    private static int[] Qb(float[] b, int scale)
    {
        var a = new int[b.Length];
        for (int i = 0; i < b.Length; i++) a[i] = (int)MathF.Round(b[i] * scale);
        return a;
    }

    /// <summary>
    /// acc += L0 column for feature f (int16, exact; vectorised). Wrapping matches the scalar path.
    /// </summary>
    public void AddFeature(short[] acc, int f)
    {
        int b = f * H, w = Vector<short>.Count, i = 0;
        for (; i <= H - w; i += w) (new Vector<short>(acc, i) + new Vector<short>(L0, b + i)).CopyTo(acc, i);
        for (; i < H; i++) acc[i] += L0[b + i];
    }

    /// <summary>
    /// acc -= L0 column for feature f (int16, exact; vectorised).
    /// </summary>
    public void SubFeature(short[] acc, int f)
    {
        int b = f * H, w = Vector<short>.Count, i = 0;
        for (; i <= H - w; i += w) (new Vector<short>(acc, i) - new Vector<short>(L0, b + i)).CopyTo(acc, i);
        for (; i < H; i++) acc[i] -= L0[b + i];
    }

    // fused updates
    public void AddSub(short[] acc, int fAdd, int fSub)
    {
        int ba = fAdd * H, bs = fSub * H, w = Vector<short>.Count, i = 0;
        for (; i <= H - w; i += w)
            (new Vector<short>(acc, i) + new Vector<short>(L0, ba + i) - new Vector<short>(L0, bs + i)).CopyTo(acc, i);
        for (; i < H; i++) acc[i] += (short)(L0[ba + i] - L0[bs + i]);
    }

    public void AddSubSub(short[] acc, int fAdd, int fSub1, int fSub2)
    {
        int ba = fAdd * H, b1 = fSub1 * H, b2 = fSub2 * H, w = Vector<short>.Count, i = 0;
        for (; i <= H - w; i += w)
            (new Vector<short>(acc, i) + new Vector<short>(L0, ba + i) - new Vector<short>(L0, b1 + i) - new Vector<short>(L0, b2 + i)).CopyTo(acc, i);
        for (; i < H; i++) acc[i] += (short)(L0[ba + i] - L0[b1 + i] - L0[b2 + i]);
    }

    /// <summary>
    /// Seed an accumulator with the feature-transformer bias (start of a full refresh).
    /// </summary>
    public void CopyBias(short[] acc) => Array.Copy(L0Bias, acc, H);

    /// <summary>Quantised forward → centipawns (side-to-move POV correction). AVX2 when available, else scalar.
    /// Both paths do identical exact integer arithmetic, so they agree bit-for-bit.</summary>
    public int Forward(short[] accStm, short[] accNon) =>
        Avx2.IsSupported ? ForwardSimd(accStm, accNon) : ForwardScalar(accStm, accNon);

    /// <summary>
    /// Scalar reference forward.
    /// </summary>
    public int ForwardScalar(short[] accStm, short[] accNon)
    {
        Span<short> x = stackalloc short[TwoH];
        Clamp(accStm, accNon, x);

        Span<short> a1 = stackalloc short[L1Out];
        for (int o = 0; o < L1Out; o++)
        {
            int s = L1Bias[o], baseo = o * TwoH;
            for (int i = 0; i < TwoH; i++) s += x[i] * L1[baseo + i];
            a1[o] = Requantise(s);
        }
        return Output(a1);
    }

    /// <summary>
    /// AVX2 forward: int16 activations × int16 weights via <c>madd</c> (_mm256_madd_epi16), int32 accumulation.
    /// </summary>
    private int ForwardSimd(short[] accStm, short[] accNon)
    {
        Span<short> x = stackalloc short[TwoH];
        Clamp(accStm, accNon, x);

        ref short x0 = ref MemoryMarshal.GetReference(x);
        ref short w0 = ref MemoryMarshal.GetArrayDataReference(L1);

        Span<short> a1 = stackalloc short[L1Out];
        for (int o = 0; o < L1Out; o++)
        {
            int baseo = o * TwoH, i = 0;
            Vector256<int> a = Vector256<int>.Zero, b = Vector256<int>.Zero, c = Vector256<int>.Zero, d = Vector256<int>.Zero;
            for (; i <= TwoH - 64; i += 64)   // 4 independent madd chains
            {
                a = Avx2.Add(a, Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref x0, (nuint)i),        Vector256.LoadUnsafe(ref w0, (nuint)(baseo + i))));
                b = Avx2.Add(b, Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref x0, (nuint)(i + 16)), Vector256.LoadUnsafe(ref w0, (nuint)(baseo + i + 16))));
                c = Avx2.Add(c, Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref x0, (nuint)(i + 32)), Vector256.LoadUnsafe(ref w0, (nuint)(baseo + i + 32))));
                d = Avx2.Add(d, Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref x0, (nuint)(i + 48)), Vector256.LoadUnsafe(ref w0, (nuint)(baseo + i + 48))));
            }
            for (; i <= TwoH - 16; i += 16)   // 16-wide remainder
                a = Avx2.Add(a, Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref x0, (nuint)i), Vector256.LoadUnsafe(ref w0, (nuint)(baseo + i))));
            int s = L1Bias[o] + Vector256.Sum(Avx2.Add(Avx2.Add(a, b), Avx2.Add(c, d)));
            for (; i < TwoH; i++) s += x[i] * L1[baseo + i];              // scalar tail
            a1[o] = Requantise(s);
        }
        return Output(a1);
    }

    private void Clamp(short[] accStm, short[] accNon, Span<short> x)
    {
        // Vectorize the int16 crelu clamp
        var lo = Vector<short>.Zero;
        var hi = new Vector<short>(QA);
        int w = Vector<short>.Count, i = 0;
        for (; i <= H - w; i += w)
        {
            Vector.Min(Vector.Max(new Vector<short>(accStm, i), lo), hi).CopyTo(x.Slice(i, w));
            Vector.Min(Vector.Max(new Vector<short>(accNon, i), lo), hi).CopyTo(x.Slice(H + i, w));
        }
        for (; i < H; i++)
        {
            x[i] = (short)Math.Clamp((int)accStm[i], 0, QA);
            x[H + i] = (short)Math.Clamp((int)accNon[i], 0, QA);
        }
    }

    /// <summary>
    /// CReLU + round-to-nearest requantise of an L1 pre-activation (scale QA·QW) back to [0, QA].
    /// </summary>
    private static short Requantise(int s) => (short)Math.Clamp((s + (1 << (Shift - 1))) >> Shift, 0, QA);

    private int Output(Span<short> a1)
    {
        int outp = L2Bias;
        for (int o = 0; o < L1Out; o++) outp += a1[o] * L2[o];
        double cp = outp * (double)OutScale / (QA * QW);
        return (int)Math.Clamp(Math.Round(cp), -10000, 10000);
    }
}
