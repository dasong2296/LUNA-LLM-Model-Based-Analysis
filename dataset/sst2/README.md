# OOD Evaluation

Folder `data` contain the original set of data (`dev.tsv`), and the style transfer data. 

For word-level transformations, we have `Augment` and `Shake-W`. For sentence-level style transformations, we have `Tweet`, `Shake`, `Bible`, and `Romantic Poetry`.

`p = 0` means more deterministic, while `p = 0.6` is more stochastics.

Label: `0` is negative, while `1` is positive.

There is one task
```shell
SST-2: sentiment classification 
```
