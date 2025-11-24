# Code Repo for Compositional Neural Textures (SIGGRAPH Asia 2024)

## Installation

Run
```bash setup_env.sh```

The python environment will be installed in ```.venv/```, you can activate it by executing

```source .venv/bin/activate```


Next, download model checkpoint from [Google Drive](https://drive.google.com/file/d/1kIGSZlI6O_rEssw9IYaZyBA0wycD8JBG/view?usp=sharing) and place it inside ```ckpts/```.

## Inference

Place your images under ```dataset/content_imgs``` and ```dataset/textures```.

### Style Transfer
```python src/inference/style_transfer.py```

### Texture Transfer
```python src/inference/texture_transfer.py```

### More applications are coming soon!

- [x] Style transfer 
- [x] Texture transfer
- [ ] Texture interpolation
- [ ] Editing texture variations
- [ ] Texture animation
- [ ] Edit propagation

The above setup is tested on macOS with Apple M3 Pro.

If you have any questions, feel free to open an issue or email peihan.tu@gmail.com.


## Citation
If you find our code useful to you, please cite our paper. Thanks!

```
@inproceedings{Tu:2024:CNT,
  title={Compositional neural textures},
  author={Tu, Peihan and Wei, Li-Yi and Zwicker, Matthias},
  booktitle={SIGGRAPH Asia 2024 Conference Papers},
  pages={1--11},
  year={2024}
}
```