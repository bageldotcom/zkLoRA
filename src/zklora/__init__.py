__version__ = "0.2.0"

_LAZY_EXPORTS = {
    "BaseModelClient": ("zklora.base_model_user_mpi", "BaseModelClient"),
    "BaseModelToLoRAComm": ("zklora.base_model_user_mpi", "BaseModelToLoRAComm"),
    "RemoteLoRAWrappedModule": (
        "zklora.base_model_user_mpi",
        "RemoteLoRAWrappedModule",
    ),
    "LoRAServer": ("zklora.lora_contributor_mpi", "LoRAServer"),
    "LoRAServerSocket": ("zklora.lora_contributor_mpi", "LoRAServerSocket"),
    "batch_verify_proofs": ("zklora.zk_proof_generator", "batch_verify_proofs"),
    "generate_proofs": ("zklora.zk_proof_generator", "generate_proofs"),
    "adapter_manifest_entry": ("zklora.proof_contract", "adapter_manifest_entry"),
    "write_adapter_manifest": ("zklora.proof_contract", "write_adapter_manifest"),
    "commit_activations": ("zklora.polynomial_commit", "commit_activations"),
    "verify_commitment": ("zklora.polynomial_commit", "verify_commitment"),
}

__all__ = [*_LAZY_EXPORTS, "__version__"]


def __getattr__(name):
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module 'zklora' has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    from importlib import import_module

    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
