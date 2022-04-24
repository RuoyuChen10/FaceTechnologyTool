## 1. Face alignment

please refer to [tutorial_face_alignment.ipynb](../FaceRecognition/tutorial_face_alignment.ipynb), conduct face alignment.

## 2. Generate List

Please refer to `generate_pairs.py`，modify `path`，`positive_pairs` and `negative_pairs`.

Then

```
python generate_pairs.py
```

See `text/tutorial.txt` for more details.

```
<img1> <img2> <ver?>
```

in `<ver?>`, 0 indicates non-mate, there are 2000 non-mate pairs. 1 indicate mate, there are 1000 mate pairs.

We didn't offer models, you can use the given example for trying.

## 3. compute score

see the parameters in [verification.py](verification.py), then

```
python verification.py
```

you can select the give txt file [results/text/Celeb-A-results2.txt](results/text/Celeb-A-results2.txt)

Generate `args.save_List`, see example in [text/CelebA.txt](text/CelebA.txt)

## 4. AUC

`args.save_List` is same as above

Then:

```
python AUC.py
```

```
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| 1e-06  | 1e-05  | 0.0001 | 0.001  |  0.01  |  0.1   |  0.2   |  0.4   |  0.6   |  0.8   |   1    |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
| 0.9251 | 0.9251 | 0.9251 | 0.9331 | 0.9421 | 0.9570 | 0.9710 | 0.9780 | 0.9870 | 0.9940 | 1.0000 |
+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+--------+
ROC AUC: 0.9800934433250592
```

![](./Celeb-A2-AUC.jpg)
