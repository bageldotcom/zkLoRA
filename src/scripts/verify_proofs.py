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
        help="Directory containing proof files (.pf), plus settings, vk, srs.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print more details during verification."
    )
    args = parser.parse_args()

    total_verify_time, num_proofs = batch_verify_proofs(
        proof_dir=args.proof_dir, verbose=args.verbose
    )
    print(f"Done verifying {num_proofs} proofs. Total time: {total_verify_time:.2f}s")


if __name__ == "__main__":
    main()
