"""Extended benchmark scaffold: ResNet-18/CIFAR-100 and ResNet-50/ImageNet-1K
(deltagradpaperplan.pdf Sec 4.2.1) -- convergence, final test accuracy, and label
noise resilience on standard residual architectures.

NOT implemented -- this is intentionally scaffolding only. Building this out for
real needs, at minimum:
  - torchvision.models.resnet18 / resnet50, trained from scratch (the paper's
    "evaluate convergence" framing implies training, not fine-tuning a pretrained
    checkpoint).
  - CIFAR-100: reuse deltagrad.data.cifar.get_cifar100_loaders, but ResNet-18
    expects a 224x224 3-channel input by default -- either resize CIFAR-100 up
    (wasteful) or swap in a CIFAR-sized ResNet stem (3x3 conv, no initial maxpool).
  - ImageNet-1K: roughly 150GB, not something to download/cache casually -- needs
    its own acquisition plan (a pre-existing local copy, WebDataset shards, or
    torchvision.datasets.ImageNet pointed at a manually downloaded devkit).
  - GPU-only in practice (Colab or better) -- a from-scratch ResNet-50/ImageNet-1K
    run is not something to attempt on this CPU-only machine, even in smoke mode.

Run with --smoke once implemented, to sanity-check the wiring on a tiny local
subset before handing off to a real GPU run.
"""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["cifar100", "imagenet"], default="cifar100")
    parser.add_argument("--model", choices=["resnet18", "resnet50"], default="resnet18")
    parser.add_argument("--optimizer", default="windowed")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    raise NotImplementedError(
        "resnet_cifar100_imagenet.py is a scaffold, not a working benchmark yet. "
        "See this file's module docstring for what's needed (a CIFAR-sized ResNet "
        "stem or a real ImageNet-1K data pipeline, plus a GPU) before running "
        f"--dataset {args.dataset} --model {args.model} --optimizer {args.optimizer}."
    )


if __name__ == "__main__":
    main()
