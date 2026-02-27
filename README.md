## AADN: Usage and Dataset

### AADN usage

- **Prepare Gwilliams MEG dataset** (from the project root):

  ```bash
  dora run download_only=true 'dset.selections=[gwilliams2022]'
  ```

- **Train AADN** (example configuration name):

  ```bash
  dora run model=aadn_clip
  ```

- **Evaluate zero-shot audio–MEG retrieval** (if you use the original `brainmagick` evaluation script in your project):

  ```bash
  python -m scripts.run_eval_probs grid_name="main_table"
  ```

### Gwilliams2022 dataset

- **Gwilliams2022 (MEG-MASC)** is a large-scale auditory MEG dataset with natural speech.  
- Following the original `brainmagick` setting, we perform **zero-shot audio–MEG retrieval over 1,000 candidate word-level segments**.

Our AADN implementation is built on top of, and extends, the excellent open-source **`brainmagick`** framework for speech decoding from non-invasive brain recordings.

### Citation

If you use AADN in your work, please cite:

```bibtex
@INPROCEEDINGS{11209914,
  author={Xiao, Yi and Qiao, XuYi and Zhang, Yu-Xuan and Yu, Xianchuan},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)}, 
  title={Zero-Shot Speech Perception Decoding via Advancing Representation Consistency}, 
  year={2025},
  volume={},
  number={},
  pages={1-6},
  keywords={Accuracy;Noise;Semantics;Redundancy;Neural activity;Contrastive learning;Brain modeling;Data models;Decoding;Cross modal retrieval;Multi-modal Contrastive Learning;Magnetoencephalography (MEG);Zero-shot Neural Decoding;Speech Perception;Representation Consistency},
  doi={10.1109/ICME59968.2025.11209914}}
```

