# NNgen with Software Training

## 実行手順
1. `python generate_hardware.py`でハードウェアを生成する
2. https://www.acri.c.titech.ac.jp/wordpress/archives/5576 などを参考にbitstreamを生成する
3. PYNQ環境にbitstreamと`nngen_ctrl.py`とデータセットをアップロードする
4. PYNQ上で`python main.py`を実行する
