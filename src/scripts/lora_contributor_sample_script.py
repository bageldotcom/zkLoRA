import argparse
import threading
import time

from zklora import LoRAServer, LoRAServerSocket


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Run a sample LoRA contributor server and write the adapter manifest "
            "that the verifier should pin out-of-band before inference."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default="127.0.0.1", help="Contributor bind host.")
    parser.add_argument("--port_a", type=int, default=30000, help="Contributor port.")
    parser.add_argument(
        "--base_model",
        default="distilgpt2",
        help="Base model name expected for the LoRA adapter.",
    )
    parser.add_argument(
        "--lora_model_id",
        default="ng0-k1/distilgpt2-finetuned-es",
        help="LoRA model ID or local path served by this contributor.",
    )
    parser.add_argument(
        "--out_dir",
        default="proof_artifacts",
        help="Directory where native .zklora proof artifacts are written.",
    )
    parser.add_argument(
        "--adapter_manifest",
        default="adapter-manifest.json",
        help=(
            "Convenience manifest handoff path. The verifier must obtain and pin "
            "this manifest out-of-band before inference; a post-inference manifest "
            "is not trusted expected_adapters input."
        ),
    )
    args = parser.parse_args()

    stop_event = threading.Event()
    server_obj = LoRAServer(args.base_model, args.lora_model_id, args.out_dir)
    server_obj.write_adapter_manifest(args.adapter_manifest)
    print(f"[A-Server] wrote adapter manifest => {args.adapter_manifest}")
    print(
        "[A-Server] verifier must pin this manifest out-of-band before inference; "
        "post-inference manifests are not trusted expected_adapters."
    )
    t = LoRAServerSocket(
        args.host, args.port_a, server_obj, stop_event, stop_timeout=1.0
    )
    t.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("[A-Server] stopping.")
    stop_event.set()
    t.join()


if __name__ == "__main__":
    main()
