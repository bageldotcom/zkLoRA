import argparse

from zklora import BaseModelClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_a", default="127.0.0.1")
    parser.add_argument("--port_a", type=int, default=30000)
    parser.add_argument(
        "--contributors",
        nargs="*",
        help="Additional LoRA contributors as host:port",
    )
    parser.add_argument("--base_model", default="distilgpt2")
    parser.add_argument(
        "--combine_mode", choices=["replace", "add_delta"], default="add_delta"
    )
    parser.add_argument("--transcript", default="b-transcript.json")
    parser.add_argument("--proof_dir", default="proof_artifacts")
    parser.add_argument("--expected_adapters")
    parser.add_argument("--verify_proofs", action="store_true")
    args = parser.parse_args()
    if args.verify_proofs and not args.expected_adapters:
        parser.error("--verify_proofs requires --expected_adapters")

    contributors = [(args.host_a, args.port_a)]
    if args.contributors:
        for item in args.contributors:
            host, port = item.split(":")
            contributors.append((host, int(port)))

    client = BaseModelClient(
        base_model=args.base_model,
        combine_mode=args.combine_mode,
        contributors=contributors,
    )
    client.init_and_patch()

    # forward pass => triggers submodule calls => A records transcript-bound proof inputs
    text = "Hello World, this is a LoRA test."

    loss_val = client.forward_loss(text)
    print(f"[B] final loss => {loss_val:.4f}")

    client.end_inference()
    client.transcript.write(args.transcript)
    print(
        "[B] done. B can now fetch proof files from A and verify them "
        f"against {args.transcript} and the pre-agreed adapter manifest."
    )

    if args.verify_proofs:
        from zklora import batch_verify_proofs

        total_verify_time, num_proofs = batch_verify_proofs(
            proof_dir=args.proof_dir,
            transcript=args.transcript,
            expected_adapters=args.expected_adapters,
            verbose=True,
        )
        print(f"[B] verified {num_proofs} proofs in {total_verify_time:.2f}s")


if __name__ == "__main__":
    main()
