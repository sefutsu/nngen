# NNgen with Software Training

NNgenで生成した回路を使って順伝播を行い、CPU上で32bit floatによる逆伝播を行います。

## 実行手順
### 必要なライブラリ
- pynq
- scikit-learn
- pandas

### ビットストリーム生成
KV260用のbitstream(`mlp.bit`, `mlp.hwh`)を用意しています。これを利用する場合にはこの手順をスキップできます。

1. `python generate_hardware.py`でハードウェアを生成する
2. https://www.acri.c.titech.ac.jp/wordpress/archives/5576 などを参考にbitstreamを生成する
3. PYNQ環境にbitstreamをアップロードする

### 学習の実行
MNISTの1~9で学習した重みをロードし、0を含めた1000枚の学習データでfine-tuningを行います。
```
cd ~/nngen
pip install .
cd training/float_backward
python main.py
```

なお、PYNQの環境によってはJupyter Notebookから実行する必要があります。その場合は`main.ipynb`を実行してください。
