# NNgen with Software Training

## 実行手順
KV260用のbitstream(`mlp.bit`, `mlp.hwh`)を用意しています。これを利用する場合には4から手順を開始してください。

1. `python generate_hardware.py`でハードウェアを生成する
2. https://www.acri.c.titech.ac.jp/wordpress/archives/5576 などを参考にbitstreamを生成する
3. PYNQ環境にbitstreamをアップロードする
4. データセットをアップロードする
5. PYNQ上で`python main.py`を実行する
