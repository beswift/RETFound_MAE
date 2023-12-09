## RETFound - A Playground for Pushing EyeCare Bounds


This is not the official repo for [RETFound: a foundation model for generalizable disease detection from retinal images](https://www.nature.com/articles/s41586-023-06555-x), which is based on [MAE](https://github.com/facebookresearch/mae):
(Or Keras version implemented by Yuka Kihara can be found [here](https://github.com/uw-biomedical-ml/RETFound_MAE))

This is a repo for playing with RETFound. 

Currently, it supports:
- Fine-tuning RETFound on your own data
- Using the wonderful [Transformer-MM-Explainability Repo by @hila-chefer](https://github.com/hila-chefer/Transformer-MM-Explainability) to explain the predictions of RETFound
- An attempt at using the retfound weights to train a Color Fundus Photo (cfp) to Fluorescein Angiography (FA) encoder-decoder model

()

### Key features of RETFound

- RETFound is pre-trained on 1.6 million retinal images with self-supervised learning
- RETFound has been validated in multiple disease detection tasks
- RETFound can be efficiently adapted to customised tasks


### Install environment

Before you do anything else, you need to get a clean environment set up.

1. Create environment with conda:

```
conda create -n retfound python=3.7.5 -y
conda activate retfound
```

2. Install dependencies

```
git clone https://github.com/beswift/RETFound_MAE.git
cd RETFound_MAE
pip install -r requirements.txt
```





### Fine-Tuning with your own data using the retfound weights

To fine tune RETFound on your own data:
<br> _Did you follow the steps above to get your environment set up?  Do that first!_
<br>
<br> 1. Download the RETFound pre-trained weights
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom"></th>
<th valign="bottom">ViT-Large</th>
<!-- TABLE BODY -->
<tr><td align="left">Colour fundus image</td>
<td align="center"><a href="https://drive.google.com/file/d/1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE/view?usp=sharing">download</a></td>
</tr>
<!-- TABLE BODY -->
<tr><td align="left">OCT</td>
<td align="center"><a href="https://drive.google.com/file/d/1m6s7QYkjyjJDlpEuXm7Xp3PmjN-elfW2/view?usp=sharing">download</a></td>
</tr>
</tbody></table>

2. Organise your images into folders so that each folder is the name of the "class" the images belong to, like:
```
├── data
│   ├── class_1
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   ├── image_3.jpg
│   ├── class_2
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   ├── image_3.jpg
│   ├── class_3
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   ├── image_3.jpg
```
3. In a terminal, run the following command to start fine-tuning (use IDRiD as example). A fine-tuned checkpoint will be saved during training. Evaluation will be run after training.
```
python train.py
```

4. To evaluate the fine-tuned model, run the following command:
```
python test.py
```


### Using the Retfound weights to train a Color Fundus Photo (cfp) to Fluorescein Angiography (FA) encoder-decoder model

To train a cfp to FA encoder-decoder model using the retfound weights:
<br> _Did you follow the steps above to get your environment set up?  Do that first!_

1. Download the retfound weights
2. Organise your images into folders so that each folder is the name of the "class" the images belong to, like:
```
├── data
│   ├── cfp
│   │   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   ├── image_3.jpg
│   ├── fa
|   |   ├── image_1.jpg
│   │   ├── image_2.jpg
│   │   ├── image_3.jpg
```
3. In a terminal, run the following command to start training:
```
python leaky.py
```

4. To evaluate the model, run the following command:
```
python leakyTest.py
```


<br>
<br>


### See the [original README](retfound_readme.md) for more details on how to use the base retfound scripts
### Citation

If you find the RETFound repository useful, please consider citing this paper:
```
@article{zhou2023foundation,
  title={A foundation model for generalizable disease detection from retinal images},
  author={Zhou, Yukun and Chia, Mark A and Wagner, Siegfried K and Ayhan, Murat S and Williamson, Dominic J and Struyven, Robbert R and Liu, Timing and Xu, Moucheng and Lozano, Mateo G and Woodward-Court, Peter and others},
  journal={Nature},
  pages={1--8},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```

If you find this repo useful, that's amazing and unexpected! If you are interested in working or partnering on eye care, chronic disease or wellness related projects, we'd love to work with you over at [Unified Imaging](https://github.com/unifiedimaging)! 
