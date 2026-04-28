"""
Convert a CellWhisperer Lightning checkpoint saved with transformers==4.39
so it can be loaded with transformers>=4.57.

The incompatibility: newer transformers turned PretrainedConfig._attn_implementation
into a property backed by _attn_implementation_internal, and added a dtype attribute.
Old pickled configs lack these, causing AttributeError on unpickle.

Strategy: monkey-patch PretrainedConfig with a __setstate__ that injects the missing
attributes, load the checkpoint, then re-save it so the new pickle includes them.

Usage (from SNAP compute node):
    cd /sailhome/moritzs/cellwhisperer_private
    pixi run python /sailhome/moritzs/patientwhisperer/experiments/agent_lbcl_bench_with_spatial/scripts/convert_checkpoint.py

Or pass a custom path:
    pixi run python .../convert_checkpoint.py /path/to/old.ckpt [/path/to/output.ckpt]
"""

import sys
from pathlib import Path

DEFAULT_CKPT = Path("/dfs/user/moritzs/cellwhisperer/checkpoints/best_cxg.ckpt")


def patch_pretrained_config():
    """
    Monkey-patch PretrainedConfig so that unpickling old checkpoints
    (saved with transformers<4.45) works with transformers>=4.57.

    The problem: _attn_implementation is now a property whose getter reads
    _attn_implementation_internal.  Old pickles store _attn_implementation
    as a plain dict entry.  When pickle restores __dict__ and any subsequent
    code accesses config._attn_implementation, the property getter fires and
    reads _attn_implementation_internal, which doesn't exist in the old state.

    This can also be imported and called from other code to enable loading
    old checkpoints at runtime without re-saving:

        from convert_checkpoint import patch_pretrained_config
        patch_pretrained_config()
        model = load_cellwhisperer_model(path)
    """
    from transformers.configuration_utils import PretrainedConfig

    if getattr(PretrainedConfig, "_compat_patched", False):
        return  # already patched

    def _compat_setstate(self, state):
        # Move old _attn_implementation dict entry to _attn_implementation_internal
        # which is the backing store for the new property.
        if "_attn_implementation_internal" not in state:
            state["_attn_implementation_internal"] = state.pop("_attn_implementation", None)

        # dtype was added in transformers ~4.50; old configs may have torch_dtype instead.
        if "dtype" not in state:
            state["dtype"] = state.get("torch_dtype", None)

        self.__dict__.update(state)

    PretrainedConfig.__setstate__ = _compat_setstate
    PretrainedConfig._compat_patched = True


def main():
    import torch

    ckpt_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CKPT
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ckpt_path.with_name(ckpt_path.stem + "_converted.ckpt")

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found at {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    if out_path.exists():
        print(f"Output already exists at {out_path}, skipping.", file=sys.stderr)
        sys.exit(0)

    print("Patching PretrainedConfig for backward compatibility...")
    patch_pretrained_config()

    print(f"Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # Walk the checkpoint and fix config objects so re-save includes new attributes
    if isinstance(ckpt, dict):
        print(f"Checkpoint keys: {list(ckpt.keys())}")
        if "hyper_parameters" in ckpt:
            hp = ckpt["hyper_parameters"]
            print(f"  hyper_parameters keys: {list(hp.keys()) if isinstance(hp, dict) else type(hp)}")
            if isinstance(hp, dict) and "model_config" in hp:
                _fix_config(hp["model_config"], "model_config")

    print(f"Saving converted checkpoint to {out_path} ...")
    torch.save(ckpt, out_path)
    print(f"Done. Converted checkpoint: {out_path}")
    print(f"Size: {out_path.stat().st_size / 1e9:.2f} GB")


def _fix_config(config, label="config"):
    """Ensure a PretrainedConfig instance (and any nested sub-configs) has
    the attributes expected by newer transformers."""
    from transformers.configuration_utils import PretrainedConfig

    if not isinstance(config, PretrainedConfig):
        return

    # _attn_implementation_internal
    if not hasattr(config, "_attn_implementation_internal"):
        val = config.__dict__.pop("_attn_implementation", None)
        config._attn_implementation_internal = val
        print(f"  Fixed {label}: added _attn_implementation_internal = {val}")

    # dtype
    if not hasattr(config, "dtype"):
        config.dtype = config.__dict__.get("torch_dtype", None)
        print(f"  Fixed {label}: added dtype = {config.dtype}")

    # Recurse into nested configs
    for attr_name in ("transcriptome_config", "text_config", "image_config"):
        sub = getattr(config, attr_name, None)
        if sub is not None:
            _fix_config(sub, f"{label}.{attr_name}")


if __name__ == "__main__":
    main()
