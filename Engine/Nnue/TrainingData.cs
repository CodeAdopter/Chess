using Core;

namespace Engine.Nnue;

/// <summary>
/// One labelled training position: the FEN, the engine's search score at that position (centipawns,
/// side-to-move POV, the dense teacher signal), and the eventual game result from that side's POV
/// (-1/0/+1, the sparse outcome signal). The trainer blends the two into a win-probability target.
/// </summary>
public readonly struct Sample
{
    public readonly string Fen;
    public readonly short ScoreCp;
    public readonly sbyte Result;   // -1 loss, 0 draw, +1 win, from side-to-move POV at this position

    public Sample(string fen, short scoreCp, sbyte result)
    {
        Fen = fen;
        ScoreCp = scoreCp;
        Result = result;
    }
}

/// <summary>
/// The on-disk training-data format ("TRD1"): a tiny binary container of <see cref="Sample"/> records, written by self-play and read back by the trainer.
/// </summary>
public static class TrainingData
{
    /// <summary>
    /// File-format identifier ("TRD1") written at the start of every dataset.
    /// </summary>
    public const int Magic = 0x54524431; // "TRD1"

    /// <summary>
    /// Streams samples to a .trd file. The sample count is unknown up front, so it is written as a placeholder and patched in on <see cref="Dispose"/>.
    /// </summary>
    public sealed class Writer : IDisposable
    {
        private readonly BinaryWriter w;
        public int Count { get; private set; }

        public Writer(string path)
        {
            w = new BinaryWriter(System.IO.File.Create(path));
            w.Write(Magic);
            w.Write(0); // count placeholder, patched on Dispose
        }

        /// <summary>
        /// Appends one labelled position (score is clamped to the storable range).
        /// </summary>
        public void Add(string fen, int scoreCp, int result)
        {
            int clamped = Math.Clamp(scoreCp, -10000, 10000);
            w.Write(fen);
            w.Write((short)clamped);
            w.Write((sbyte)result);
            Count++;
        }

        /// <summary>
        /// Back-patches the final sample count into the header and closes the file.
        /// </summary>
        public void Dispose()
        {
            w.Seek(4, SeekOrigin.Begin);
            w.Write(Count);
            w.Dispose();
        }
    }

    /// <summary>
    /// Read a single .trd file, or every .trd in a directory (for merging parallel runs).
    /// </summary>
    public static List<Sample> ReadAll(string path)
    {
        if (System.IO.Directory.Exists(path))
        {
            var all = new List<Sample>();
            foreach (var f in System.IO.Directory.GetFiles(path, "*.trd").OrderBy(x => x))
                all.AddRange(Read(f));
            return all;
        }
        return Read(path);
    }

    /// <summary>
    /// Reads every sample from one .trd file. Throws if the magic header does not match.
    /// </summary>
    public static List<Sample> Read(string path)
    {
        using var r = new BinaryReader(System.IO.File.OpenRead(path));
        if (r.ReadInt32() != Magic) throw new InvalidDataException("not a TRD1 file");
        int count = r.ReadInt32();
        var list = new List<Sample>(count);
        for (int i = 0; i < count; i++)
        {
            string fen = r.ReadString();
            short score = r.ReadInt16();
            sbyte result = r.ReadSByte();
            list.Add(new Sample(fen, score, result));
        }
        return list;
    }
}
