# ADR 0022: Benchmarks sample telemetry during the run and interleave compared passes

- Status: accepted
- Date: 2026-11-21

## Context

ADR 0007 sets how benchmarks are measured and when a result is official. Its telemetry
was read before and after each benchmark, and compared passes ran one after the other.
The independent review of #24 found both weak (issue #25):

- A short benchmark under the power cap can change P-state between the two readings
  (192³: P0 at 9001 MHz before, P2 at 8001 MHz after), and nothing showed when.
- Passes measured in sequence feel different clocks. At 192³, the single sweep ran
  first, in the boost window after an idle gap, and the graph later, under the power
  cap. M0's "single 0.160 ms against graph 0.199 ms" came mostly from that order.
  Timed in one process with equal gaps, sequential and interleaved, the review measured
  stream/graph at 0.91 and 1.06–1.08, and graph at 0.201 and 0.181 ms per sweep.

## Decision

- **Telemetry during the run.** `BenchSession.start` launches one `nvidia-smi -lms 100`
  process and reads its lines on a thread until `record`. The result gains
  `telemetry.during`: min, median, max and count of each numeric field, and how often
  each P-state and clock event reason was seen.
  - An unreadable value drops only itself.
  - Without nvidia-smi there are no samples, and the measurement is never lost.
  - The sampler is stopped by `record`, by starting the same name again, by the
    session's teardown when a benchmark raises, and by an `atexit` kill as a last
    resort. On Windows an orphaned nvidia-smi otherwise polls the GPU until reboot.
    Killing Python outright (`os._exit`, `taskkill /F`, closing its console) skips
    `atexit` and still orphans it; a Windows job object would close it with Python.
- **Interleaving.** Passes that a benchmark compares are timed one call each per round
  (`time_interleaved`), so drift hits them alike. This is for GPU-bound passes only. A
  host-bound pass after a long GPU pass is partly enqueued before its span starts, and
  reads short (measured: 30 %). Host-bound passes keep `time_device`.
- **Not comparable with M0.** The stencil benchmark's results from now on are not
  comparable with the M0 reports; the benchmark README says why.

## Alternatives considered

- **One nvidia-smi process per sample:** spawning costs tens of milliseconds each, and
  10 Hz would perturb the CPU side of short benchmarks.
- **NVML through a Python binding:** a new dependency for the same readings.
- **Randomised pass order instead of round-robin:** it decorrelates drift but needs
  more rounds for the same stability.

## How we would know it is wrong

- An nvidia-smi is left running after a benchmark session that ended normally or on
  an exception.
- Ratios between interleaved passes still swing by more than their spread from session
  to session.
