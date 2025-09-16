import plonky3_py as pl

from zklora import export_lora_submodules, generate_proofs, batch_verify_proofs

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import asyncio
import ezkl

# Fixed point encoding function for 16-bit floating point numbers
import numpy as np

def fixed_point_encode(x, fractional_bits=10):
    """
    Encodes an array (or scalar) of 16-bit floats into fixed-point representation as u32 integers.

    Parameters:
        x: array-like of np.float16 (or convertible to np.float16)
        fractional_bits: Number of bits for the fractional part (default is 10)

    Returns:
        A numpy array of type np.uint32 containing the fixed-point encoded values.
    """
    arr = np.asarray(x, dtype=np.float16)
    multiplier = 2 ** fractional_bits
    fixed_point = np.round(arr.astype(np.float32) * multiplier)
    return fixed_point.astype(np.uint32).tolist()

# Fixed point decoding function for 16-bit floating point numbers from u32 fixed-point representation
def fixed_point_decode(y, fractional_bits=10):
    """
    Decodes an array (or scalar) of u32 fixed-point encoded numbers back into 16-bit floating point numbers.

    Parameters:
        y: array-like of np.uint32 (or convertible to np.uint32)
        fractional_bits: Number of bits for the fractional part (default is 10)

    Returns:
        A numpy array of type np.float16 containing the decoded values.
    """
    arr = np.asarray(y, dtype=np.uint32)
    multiplier = 2 ** fractional_bits
    decoded = arr.astype(np.float32) / multiplier
    return decoded.astype(np.float16).tolist()

# Patch ezkl.gen_witness to be awaitable so generate_proofs can await it
if not asyncio.iscoroutinefunction(getattr(ezkl, "gen_witness", None)):
    _orig_gen_witness = ezkl.gen_witness

    async def _gen_witness_async(*args, **kwargs):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, lambda: _orig_gen_witness(*args, **kwargs))

    ezkl.gen_witness = _gen_witness_async  # type: ignore

v = [0.1, 0.2, 0.3]
A = [[0.1, 0.2, 0.3],
     [0.4, 0.5, 0.6],
     [0.7, 0.8, 0.9]]

v2 = [200, 200, 300]
A2 = [[1, 4, 3],
     [4, 5, 6],
     [7, 8, 9]]

v_encoded = fixed_point_encode(v)
A_encoded = [fixed_point_encode(row) for row in A]

print("v:", v_encoded)
print("A:", A_encoded)

print("v decoded:", fixed_point_decode(v_encoded))
print("A decoded:", [fixed_point_decode(row) for row in A_encoded])

proof = pl.vector_matrix_multiplication_prove(3, 3, v2, A2)
assert pl.vector_matrix_multiplication_verify(3, 3, proof)
print("proof length:", len(proof))



"""
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
"""