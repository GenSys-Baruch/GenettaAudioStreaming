using System;
using System.Net;
using System.Runtime.InteropServices;
using System.Threading;

// C# example demonstrating how to P/Invoke the GenettaAudioStreaming Rust library
// Build the Rust project first to produce GenettaAudioStreaming.dll
// Then run this C# sample (e.g., `dotnet new console` and add this file, or compile with csc)

internal static class GasNative
{
    private const string Dll = "GenettaAudioStreaming"; // resolves to GenettaAudioStreaming.dll on Windows

    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    public delegate void PcmCb(
        ulong inst,
        IntPtr pcm,
        UIntPtr samples,
        uint sampleRate,
        byte channels
    );

    // Receiver side
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_create_rtp_receiver")]
    public static extern ulong CreateRtpReceiver(ushort bindPort);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_on_pcm")]
    public static extern void OnPcm(ulong inst, PcmCb cb);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_stop")]
    public static extern void Stop(ulong inst);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_destroy")]
    public static extern void Destroy(ulong inst);

    // Sender side
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_create_rtp_sender")]
    public static extern ulong CreateRtpSender(
        ushort localBindPort,
        uint remoteIpv4Be,
        ushort remotePort,
        byte payloadType
    );

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_push_pcm")]
    public static extern void PushPcm(
        ulong senderInst,
        IntPtr pcm,
        UIntPtr samples,
        uint sampleRate,
        byte channels
    );

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_stop_sender")]
    public static extern void StopSender(ulong senderInst);

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_destroy_sender")]
    public static extern void DestroySender(ulong senderInst);

    // Version helpers
    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_version_major")]
    public static extern int VersionMajor();

    [DllImport(Dll, CallingConvention = CallingConvention.Cdecl, EntryPoint = "gas_version_minor")]
    public static extern int VersionMinor();
}

internal static class Program
{
    // Keep a reference to the delegate so it isn't GC'ed while native code stores/calls it
    private static GasNative.PcmCb? _pcmCallbackRef;

    private static void Main(string[] args)
    {
        Console.WriteLine($"GenettaAudioStreaming v{GasNative.VersionMajor()}.{GasNative.VersionMinor()}");
        Console.WriteLine("1 = Receiver demo, 2 = Sender demo");
        Console.Write("Choose: ");
        var key = Console.ReadKey();
        Console.WriteLine();
        if (key.KeyChar == '1')
        {
            ReceiverDemo();
        }
        else if (key.KeyChar == '2')
        {
            SenderDemo();
        }
        else
        {
            Console.WriteLine("Unknown selection.");
        }
    }

    private static void ReceiverDemo()
    {
        // Bind to 0.0.0.0:5004 and print incoming PCM summary
        const ushort bindPort = 5004;
        ulong inst = GasNative.CreateRtpReceiver(bindPort);
        if (inst == 0)
        {
            Console.WriteLine("Failed to create RTP receiver.");
            return;
        }

        // Define the PCM callback
        _pcmCallbackRef = new GasNative.PcmCb((ulong i, IntPtr pcmPtr, UIntPtr samples, uint rate, byte ch) =>
        {
            // Copy PCM i16 buffer to managed array (use Marshal.Copy for Int16)
            long sampleCount = (long)samples;
            if (sampleCount <= 0)
                return;

            short[] buffer = new short[sampleCount];
            Marshal.Copy(pcmPtr, buffer, 0, (int)sampleCount);

            // Compute simple peak for display
            short peak = 0;
            for (int n = 0; n < buffer.Length; n++)
            {
                short val = buffer[n];
                if (val < 0)
                {
                    short abs = (short)(-val);
                    if (abs > peak) peak = abs;
                }
                else if (val > peak) peak = val;
            }
            Console.WriteLine($"[RX] inst={i} got {buffer.Length} samples @ {rate} Hz, ch={ch}, peak={peak}");
        });

        // Register callback and run for a while
        GasNative.OnPcm(inst, _pcmCallbackRef);
        Console.WriteLine($"Receiver created on UDP :{bindPort}. Press any key to stop...");
        Console.ReadKey(true);

        GasNative.Stop(inst);
        GasNative.Destroy(inst);
        Console.WriteLine("Receiver stopped.");
    }

    private static void SenderDemo()
    {
        // Configure destination
        // Example: send to 127.0.0.1:5004 using PCMU (payload type 0)
        const ushort localBindPort = 0; // 0 = OS chooses ephemeral local port
        const string remoteIp = "127.0.0.1";
        const ushort remotePort = 5004;
        const byte payloadType = 0; // 0 = PCMU, 8 = PCMA

        uint remoteIpv4Be = IPv4ToBigEndian(remoteIp);
        ulong sender = GasNative.CreateRtpSender(localBindPort, remoteIpv4Be, remotePort, payloadType);
        if (sender == 0)
        {
            Console.WriteLine("Failed to create RTP sender.");
            return;
        }

        Console.WriteLine($"Sender created -> {remoteIp}:{remotePort} PT={payloadType}. Streaming 5 seconds of 1 kHz tone...");

        // Generate and send a 1 kHz sine at 48 kHz mono for ~5 seconds
        const uint sampleRate = 48000;
        const byte channels = 1;
        const double freq = 1000.0; // Hz
        const double durationSec = 5.0;
        int totalSamples = (int)(sampleRate * durationSec);

        // Send in frames of 20 ms (common RTP frame size)
        int frameSamples = (int)(sampleRate / 50); // 20 ms
        short[] frame = new short[frameSamples * channels];

        double twoPiOverFs = 2.0 * Math.PI / sampleRate;
        double phase = 0;

        var sw = System.Diagnostics.Stopwatch.StartNew();
        int sent = 0;
        while (sent < totalSamples)
        {
            int count = Math.Min(frameSamples, totalSamples - sent);
            for (int n = 0; n < count; n++)
            {
                short s = (short)(Math.Sin(phase) * short.MaxValue * 0.2); // -14 dBFS approx
                frame[n] = s; // mono
                phase += twoPiOverFs * freq;
                if (phase > Math.PI * 2) phase -= Math.PI * 2;
            }

            // Pin and push the frame
            unsafe
            {
                fixed (short* p = frame)
                {
                    IntPtr ptr = (IntPtr)p;
                    GasNative.PushPcm(sender, ptr, (UIntPtr)count, sampleRate, channels);
                }
            }

            sent += count;
            // Pace roughly in real-time
            Thread.Sleep(20);
        }

        GasNative.StopSender(sender);
        GasNative.DestroySender(sender);
        Console.WriteLine($"Done. Elapsed {sw.Elapsed}.");
    }

    private static uint IPv4ToBigEndian(string ipv4)
    {
        var ip = IPAddress.Parse(ipv4);
        byte[] b = ip.GetAddressBytes(); // already network byte order (big-endian)
        if (b.Length != 4) throw new ArgumentException("Only IPv4 is supported here.");
        return (uint)(b[0] << 24 | b[1] << 16 | b[2] << 8 | b[3]);
    }
}
