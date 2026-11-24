# MorphLaM

This repository is the official implementation of **"MorphLaM: Morphable Latent Masking for Compact Tensor Factorization"** (KDD 2027).
The supplementary material is available [here](material/supplementary.pdf).

## Files

- `morphlam.py`: MorphLaM, CP, Tucker, and Tensor-Train. It includes Adam-based fitting, CP-ALS, HOSVD+HOOI, and permutation-aware TT-SVD.
- `mulcp.py`: Hadamard-product CP baseline.
- `mmf.py`: fixed-mask mixture factorization baseline.
- `tensorcodec.py`: integrated TensorCodec-style neural Tensor-Train baseline.
- `puzzletensor.py`: integrated PuzzleTensor-style transformation followed by CP, Tucker, or TT.
- `synthetic.py`: dense-random and block-diagonal tensor generators.
- `common.py`, `classical.py`: shared tensor operations, budget allocation, logging, and classical solvers.

## Requirements

```text
Python >= 3.10
PyTorch >= 2.3
NumPy >= 1.26
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Quick start

Compare MorphLaM with classical decompositions:

```bash
python morphlam.py \
  --dims 64,64,64 \
  --data random \
  --parameter_ratio 0.125 \
  --rank 24 \
  --methods morphlam,cp,cp_als,tucker,tucker_hooi,tt,tt_svd_perm
```

Run the block-diagonal setting:

```bash
python morphlam.py \
  --dims 60,60,60 \
  --data block \
  --num_blocks 2 \
  --parameter_ratio 0.0625 \
  --rank 16
```

Run the additional baselines with the same tensor shape and parameter budget:

```bash
python mulcp.py        --dims 64,64,64 --data random --target_params 32768
python mmf.py          --dims 64,64,64 --data random --target_params 32768 --rank 24
python tensorcodec.py  --dims 64,64,64 --data random --target_params 32768
python puzzletensor.py --dims 64,64,64 --data random --target_params 32768
```

Use multiple seeds with `--seeds 1,2,3,4,5`. Each script writes `results.csv` under its output directory. Training checkpoints can also be written locally:

```bash
python morphlam.py --log_every 100 --log_progress_csv
```

Pass a path after `--log_progress_csv` to choose a custom location.

Run `python <script>.py --help` for the complete set of options.

## Reference

If you use this code, please cite the following paper.
```bibtex
@inproceedings{park2027morphlam,
  title={MorphLaM: Morphable Latent Masking for Compact Tensor Factorization},
  author={Park, Yong-chan and Lee, SeungJoo and Lee, Jeongyoung and Kang, U},
  booktitle={Proceedings of the 33rd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1},
  year={2027}
}
