# macOS OpenCL silently reports success for aborted dispatches

**On macOS, Apple's OpenCL can report that a kernel finished successfully when
the GPU never ran it.** The output buffer comes back untouched, and
`clEnqueueNDRangeKernel`, `clFinish` and `CL_EVENT_COMMAND_EXECUTION_STATUS`
all indicate success. Metal, one layer down, *does* report the failure.

The cause is not the kernel. macOS aborts the Metal command buffer that backs
the dispatch when the compute work starves the display, logging
`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`. Apple's OpenCL
runtime does not propagate that error.

**This repository lets you check it on your own Mac in about a minute, and
contribute the result.**

## Why it matters

An aborted dispatch is **fast**, because it did not do the work:

| 20 000 work-items, same kernel | time | output written |
|---|---|---|
| completed | 61.2 s | 100% |
| aborted, reported as success | 0.5 s | **0%** |

So any benchmark that measures time without verifying results against a
reference will record its **largest speedups for the runs that failed most
completely**. Correctness checks are not a nicety on this platform; they are
the only thing standing between you and a published number that is an artefact
of work never performed.

If you compute on an Apple GPU and do not verify your results, you may have
been affected without knowing.

## Run it

Requires macOS and the Xcode command line tools (`xcode-select --install`).
Nothing else — no dependencies, no network, no credentials.

```sh
git clone https://github.com/nejeoui/opencl-silent-abort
cd opencl-silent-abort
bash apple_opencl_abort_probe.sh --quick --no-mail
```

It compiles two small programs, runs the same kernel arithmetic through OpenCL
and through Metal, and prints one of:

| verdict | meaning |
|---|---|
| `REPRODUCED` | OpenCL reported success for dispatches that did not run, and Metal reported the abort on the same machine |
| `REPRODUCED (OpenCL side)` | OpenCL silently lost work; Metal happened not to abort during its own runs |
| `NOT REPRODUCED` | Metal aborted but OpenCL did not lose work |
| `NOT TRIGGERED` | nothing aborted — your machine tolerated the load. **This is a useful result too**, please still report it |

A full report is written to your Desktop. Read it before sharing: it is plain
text, and the top of the script lists exactly what it contains.

## Please contribute your result

Coverage across Apple silicon generations is what this needs most. Two ways:

1. **Open an issue** using the *Test result* template and paste the report.
2. **Send a pull request** adding your report to `results/` and one row to the
   table below.

Either is welcome. Reports from machines where it does **not** reproduce are
just as valuable — they bound the problem.

### Results so far

| Chip | GPU cores | RAM | macOS | Verdict | Report |
|---|---|---|---|---|---|
| Apple M2 | 10 | 16 GB | 26.3.1 (25D771280a) | `REPRODUCED` | [`m2-10c-...txt`](results/m2-10c-macos26.3.1-reproduced.txt) |

## What the script collects

Machine model, chip, core counts, RAM, macOS version, GPU name, the OpenCL
device and driver strings, the probe output, and macOS log lines matching
**only** the text `command buffer was aborted` during the run.

It does not read your files, your network, or the rest of your system log.

**It never sends anything by itself and contains no mail credentials.** Without
`--no-mail` it offers to open a pre-filled draft in Mail.app with the report
attached, which you review and send yourself. Decline and it just prints the
file path.

## Confirming it without these programs

The operating system logs every abort:

```sh
log show --last 30m --predicate 'eventMessage CONTAINS "command buffer was aborted"'
```

A line containing `kIOGPUCommandBufferCallbackErrorImpactingInteractivity` is
macOS saying it terminated GPU work because it was starving the display.

## Files

| File | Purpose |
|---|---|
| `apple_opencl_abort_probe.sh` | **Start here.** Self-contained: embeds both probes, runs the ladder, writes the report, prints a verdict. `--quick`, `--no-mail`, `--help`. |
| `metalprobe.m` | Standalone Metal probe — the smallest readable program that shows Metal *does* report the abort, with the full command-buffer status and error domain/code. The script embeds an equivalent probe, so this file is for reading and for testing Metal on its own, not a prerequisite. |
| `probe_real_kernel.c` | Optional. Runs a real 2048-bit modular-exponentiation kernel rather than a synthetic loop, which is the evidence that production code is affected. Requires the kernel from [MPA-OpenCL](https://github.com/nejeoui/MPA-OpenCl) via `MPA_KERNEL_DIR`. |

## How it is measured

The output buffer is pre-filled with `0xDEADBEEF` before every dispatch. A
work-item that runs overwrites its slot; a slot still holding the sentinel was
never written. This separates two things that a comparison against a reference
conflates: *the arithmetic is wrong* and *the arithmetic never ran*. It is
always the latter — no run has ever produced a written-but-incorrect value.

A zero-filled buffer cannot make that distinction, because zero is a legitimate
result.

## Expect nondeterminism

Whether a dispatch is aborted depends on contention with the display, which no
launch parameter controls. Identical commands on one machine have produced an
82-second correct run and a 0.5-second empty one. **Run each configuration
several times** before concluding anything, and do not use the machine while a
probe runs.

Only integrated GPUs that drive a display appear to be affected. A headless
datacenter GPU has no interactivity to impact.

## The proposed fix

The information is already available to the runtime. In the command-buffer
completion handler, when `status == MTLCommandBufferStatusError`, map the
non-`nil` `error` onto the OpenCL surface:

| Metal | OpenCL |
|---|---|
| `MTLCommandBufferStatusError` | event status ← negative error code |
| `MTLCommandBufferErrorDomain` | `CL_OUT_OF_RESOURCES` from `clFinish` / `clWaitForEvents` |
| `MTLCommandBufferStatusCompleted` | `CL_COMPLETE` (unchanged) |

This changes no scheduling and no abort policy — only whether the application
is told. The OpenCL specification already provides for it: event status is
defined to carry a negative error code when a command is abnormally terminated,
`clCreateContext` accepts a `pfn_notify` callback for asynchronous errors, and
`clFinish` has `CL_OUT_OF_RESOURCES` available.

Workarounds until then: verify every result against a reference before trusting
a timing; keep each dispatch short; prefer a GPU that is not driving a display.

## Citing

This accompanies a paper describing the diagnosis and the proposed fix.

> A. Nejeoui. *Silently Aborted Dispatches in Apple's OpenCL-on-Metal Runtime:
> Diagnosis, Consequences for Benchmarking, and a Proposed Fix.*

It was found while validating [MPA-OpenCL](https://github.com/nejeoui/MPA-OpenCl),
a portable multiple-precision arithmetic library that checks every result
against GMP before timing it — which is how a fast wrong answer was caught
rather than published.

## Method and disclosure

Everything here was obtained from a retail Mac using documented, publicly
available interfaces.

- **No reverse engineering.** No Apple binary, library or framework was
  disassembled, decompiled or patched. The evidence is return values from
  published OpenCL and Metal calls, documented properties
  (`MTLCommandBuffer.status` and `.error`), and messages the operating system
  writes to its own log, read with the standard `log show` utility.
- **No protection measure was circumvented.**
- **No Apple code is redistributed.** These programs include Apple's public
  SDK headers by reference, as any application does.
- **This is a correctness bug, not a security vulnerability.** It confers no
  privilege, discloses no data and offers no exploitation path. It is published
  because it silently corrupts measurements.
- **Reported to Apple** via Feedback Assistant before publication:
  **FB24808475**.

Two questions are kept separate throughout. Whether macOS *should* terminate
compute work that starves the display is a design decision, and a defensible
one — no position is taken on it here. Whether an API should *report* that
termination to the application blocked waiting for it is not a matter of taste:
the OpenCL specification provides the mechanisms, and Metal demonstrably has
the information. The claim is confined to the second.

## Licence

MIT. See [LICENSE](LICENSE).
