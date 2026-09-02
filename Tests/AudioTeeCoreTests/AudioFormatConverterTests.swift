import AVFoundation
import XCTest

@testable import AudioTeeCore

/// Regression tests for the resampler.
///
/// The bug these guard against: the converter's input block used to return the
/// same buffer with `.haveData` every time AVAudioConverter asked. Measured
/// 2026-09-02 with an uncapped output buffer, the block was called 4 times for
/// one chunk and produced 4800 frames instead of 1600, with 9 discontinuities
/// in a monotonic ramp — the same audio converted repeatedly.
///
/// It never corrupted a real recording only because `getBuffers` sizes the
/// output buffer to exactly `input * ratio`, so the cap truncated the duplicate
/// passes. That is correctness by accident: any change to the capacity
/// calculation, or a chunk whose frame count differs from the buffer it was
/// sized for, would have let the duplication through.
///
/// These tests do not check that the input block was changed — it was not.
/// They check the invariant that makes the current block safe: output frames
/// must never EXCEED `input * ratio` (which would mean duplication got past the
/// capacity cap) and must not fall far below it (which would mean dropped
/// audio). If `getBuffers` is ever changed to allocate a roomier output buffer,
/// the first assertion fails and the duplication is caught here rather than in
/// someone's meeting recording.
final class AudioFormatConverterTests: XCTestCase {
  /// Bounded allowance for resampler filter priming. Small and paid once for
  /// the stream; it must never scale with the number of chunks.
  private static let primingSlack = 300

  /// AudioTeeCore's converter targets 16-bit, not the source's 32-bit float.
  private func bytesPerOutputFrame(_ converter: AudioFormatConverter) -> Int {
    Int(converter.targetFormatDescription.mBytesPerFrame)
  }

  private func sourceFormat(sampleRate: Double) -> AudioStreamBasicDescription {
    AudioStreamBasicDescription(
      mSampleRate: sampleRate,
      mFormatID: kAudioFormatLinearPCM,
      mFormatFlags: kAudioFormatFlagIsFloat | kAudioFormatFlagIsPacked,
      mBytesPerPacket: 4, mFramesPerPacket: 1, mBytesPerFrame: 4,
      mChannelsPerFrame: 1, mBitsPerChannel: 32, mReserved: 0)
  }

  /// Downsampling 48 kHz to 16 kHz must produce close to a third of the frames.
  /// Duplicated input showed up here as roughly double the expected output.
  func testDownsampleProducesExpectedFrameCount() throws {
    let converter = try AudioFormatConverter.toSampleRate(
      16000, from: sourceFormat(sampleRate: 48000))

    let inputFrames = 4800  // 100 ms at 48 kHz
    var samples = [Float32](repeating: 0, count: inputFrames)
    for i in 0..<inputFrames {
      samples[i] = Float32(sin(Double(i) * 0.05)) * 0.5
    }

    var outputBytes = 0
    samples.withUnsafeBytes { raw in
      converter.transform(from: raw.baseAddress!, count: raw.count) { _, count in
        outputBytes += count
      }
    }

    let outputFrames = outputBytes / bytesPerOutputFrame(converter)
    let ideal = inputFrames / 3
    XCTAssertGreaterThan(outputFrames, 0, "converter produced no output")
    XCTAssertLessThanOrEqual(outputFrames, ideal,
      "more output than input can justify — duplication escaped the capacity cap")
    XCTAssertGreaterThanOrEqual(outputFrames, ideal - Self.primingSlack,
      "audio was dropped: got \(outputFrames), ideal \(ideal)")
  }

  /// Feeding successive chunks must not accumulate duplicated audio: the total
  /// output over several chunks stays proportional to the total input.
  func testRepeatedChunksDoNotDuplicate() throws {
    let converter = try AudioFormatConverter.toSampleRate(
      16000, from: sourceFormat(sampleRate: 48000))

    let chunkFrames = 2400
    let chunks = 5
    let samples = [Float32](repeating: 0.25, count: chunkFrames)

    var outputBytes = 0
    for _ in 0..<chunks {
      samples.withUnsafeBytes { raw in
        converter.transform(from: raw.baseAddress!, count: raw.count) { _, count in
          outputBytes += count
        }
      }
    }

    let outputFrames = outputBytes / bytesPerOutputFrame(converter)
    let ideal = chunkFrames * chunks / 3
    XCTAssertLessThanOrEqual(outputFrames, ideal,
      "duplication accumulated across chunks")
    XCTAssertGreaterThanOrEqual(outputFrames, ideal - Self.primingSlack,
      "shortfall grew with chunk count, so audio is being dropped")
  }
}
