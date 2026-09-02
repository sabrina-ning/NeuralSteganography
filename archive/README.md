# Archive

Previous experimental scripts. These are preserved for reference but are **not** part of the active codebase.

## Exploratory Tests

| File | Original Name | Description |
|------|---------------|-------------|
| `emu3_source_model_tests.py` | `test_2.py` | Token sampling using the original (non-HF) Emu3 source code models |
| `entropy_stress_test_emu3.py` | `test_3.py` | Stress-tests text generation entropy dynamics on Emu3-Chat |
| `entropy_stress_test_gpt2.py` | `test_4.py` | Same entropy experiments using GPT-2 |
| `cache_truncation_gpt2xl.py` | `test_5.py` | KV cache truncation experiments with GPT-2-XL |
| `long_generation_olmo.py` | `test_6.py` | Long-form generation experiments with OLMo-7B |
| `text_ac_roundtrip_test.py` | `test_arithmetic.py` | Text-only arithmetic coding encode→decode roundtrip test |
| `image_text_ac_test.py` | `test_arithmetic_2.py` | Image + text arithmetic coding pipeline test (with breakpoints) |
| `token_comparison.py` | `test_load_image.py` | Utility to compare VQ tokens between original and recovered images |
| `unit_tests.py` | `unit_tests.py` | Collection of tests exploring different Emu3 model checkpoints |
| `run_arithmetic.py` | `run_arithmetic.py` | Text-only steganography test across multiple models |

## Upstream Baselines (from original STEGASURAS repo)

| File | Description |
|------|-------------|
| `block_baseline.py` | Binning-based steganography algorithm |
| `huffman.py` | Huffman coding implementation |
| `huffman_baseline.py` | Huffman-based steganography algorithm |
| `sample.py` | Simple sampling baseline |

## Data

| Directory | Description |
|-----------|-------------|
| `vocab/` | Vocabulary CSV exports for different tokenizers |
