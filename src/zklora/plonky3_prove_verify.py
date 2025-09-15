import plonky3_py as pl

from zklora import export_lora_submodules, generate_proofs, batch_verify_proofs

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import asyncio
import ezkl

# Patch ezkl.gen_witness to be awaitable so generate_proofs can await it
if not asyncio.iscoroutinefunction(getattr(ezkl, "gen_witness", None)):
    _orig_gen_witness = ezkl.gen_witness

    async def _gen_witness_async(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _orig_gen_witness(*args, **kwargs))

    ezkl.gen_witness = _gen_witness_async  # type: ignore

v = [1, 2, 3]
A = [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]

proof = pl.vector_matrix_multiplication_prove(3, 3, v, A)
assert pl.vector_matrix_multiplication_verify(3, 3, proof)
print("proof length:", len(proof))




async def main():
     base_model_name = "distilgpt2"
     lora_model_name = "q1e123/peft-starcoder-lora-a100"
     base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
     lora_model = PeftModel.from_pretrained(base_model, lora_model_name)
     tokenizer = AutoTokenizer.from_pretrained(base_model_name)
     lora_model.eval()

     texts = ["Hello from LoRA", "And another test", "One more line..."]

     export_lora_submodules(
          model=lora_model,
          tokenizer=tokenizer,
          input_texts=texts,
          submodule_key="attn.c_attn",
     )

     await generate_proofs(verbose=True)

if __name__ == "__main__":
    asyncio.run(main())
    print("Done")