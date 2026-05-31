import argparse

from zklora import batch_verify_proofs


def main():
    parser = argparse.ArgumentParser(
        description="Verify LoRA proof artifacts in a given directory."
    )
    parser.add_argument(
        "--proof_dir",
        type=str,
        default="proof_artifacts",
        help="Directory containing native .zklora proof artifacts.",
    )
    parser.add_argument(
        "--transcript",
        type=str,
        required=True,
        help="Base user transcript JSON captured during inference.",
    )
    parser.add_argument(
        "--expected_adapters",
        type=str,
        required=True,
        help="Pre-inference adapter manifest JSON agreed by the verifier.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print more details during verification."
    )
    args = parser.parse_args()

    total_verify_time, num_proofs = batch_verify_proofs(
        proof_dir=args.proof_dir,
        transcript=args.transcript,
        expected_adapters=args.expected_adapters,
        verbose=args.verbose,
    )
    print(f"Done verifying {num_proofs} proofs. Total time: {total_verify_time:.2f}s")


if __name__ == "__main__":
    main()
