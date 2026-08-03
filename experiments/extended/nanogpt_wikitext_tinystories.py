"""Extended benchmark scaffold: NanoGPT (6-12 layer decoder-only Transformer) on
WikiText-2 / TinyStories (deltagradpaperplan.pdf Sec 4.2.2) -- verifies DeltaGrad-EMA
scales to high-dimensional parameter spaces where the windowed variant's K*d
gradient-history storage would become memory-prohibitive.

NOT implemented -- this is intentionally scaffolding only. Building this out for
real needs, at minimum:
  - A decoder-only Transformer implementation (e.g. Karpathy's nanoGPT, vendored or
    installed) -- not something to hand-roll as a side effect of this reorg.
  - WikiText-2 / TinyStories tokenized and chunked into fixed-length sequences,
    plus a BPE or char-level tokenizer.
  - GPU-only in practice for any non-trivial layer count -- not something to
    attempt on this CPU-only machine, even in smoke mode with 6 layers.

Run with --smoke once implemented (tiny model, short sequence length, a handful of
steps) to sanity-check the wiring before handing off to a real GPU run.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["wikitext2", "tinystories"], default="wikitext2")
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--optimizer", default="ema")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    raise NotImplementedError(
        "nanogpt_wikitext_tinystories.py is a scaffold, not a working benchmark yet. "
        "See this file's module docstring for what's needed (a nanoGPT-style model "
        "implementation and a tokenized dataset pipeline, plus a GPU) before running "
        f"--dataset {args.dataset} --n-layers {args.n_layers} --optimizer {args.optimizer}."
    )


if __name__ == "__main__":
    main()
