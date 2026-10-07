# MNIST Neural Network in C

C言語でニューラルネットワークを実装し、MNIST 手書き数字データセットを用いて数字認識を行うプロジェクトです。機械学習ライブラリには依存せず、順伝播、誤差逆伝播、Adam によるパラメータ更新を実装しています。

この README は **Windows PowerShell** と **MSYS2/MinGW-w64 の GCC** での利用を前提にしています。

## 概要

28×28 ピクセルのグレースケール画像を 784 次元のベクトルとして入力し、0〜9 の数字に分類します。

```text
784 → Fully Connected (50) → ReLU → Fully Connected (100) → ReLU
    → Fully Connected (10) → Softmax → 0〜9 の分類結果
```

ネットワークの主な設定は次のとおりです。

| 項目 | 設定 |
| --- | --- |
| 入力サイズ | 784 |
| 隠れ層 | 50 → 100 |
| 出力サイズ | 10 |
| 活性化関数 | ReLU |
| 出力層 | Softmax |
| 損失関数 | 交差エントロピー誤差 |
| Optimizer | Adam |
| Learning Rate | 0.005 |
| Batch Size | 100 |
| Epochs | 10 |
| 重み初期化 | He 初期化 |

## 動作環境

- Windows 10 / 11
- PowerShell
- MSYS2/MinGW-w64 の GCC

PowerShell で次のコマンドが実行できることを確認してください。

```powershell
gcc --version
```

> Visual Studio の `cl.exe` はこの README のコンパイルコマンドとは互換ではありません。`-o` と `-lm` を使うため、MSYS2/MinGW-w64 GCC を使用してください。

## 必要なデータ

学習には MNIST の次の4ファイルが必要です。いずれも Git 管理対象外のため、リポジトリを clone しただけでは含まれません。

```text
data\
├── train-images.idx3-ubyte
├── train-labels.idx1-ubyte
├── t10k-images.idx3-ubyte
└── t10k-labels.idx1-ubyte
```

[CVDF の MNIST データセット](https://github.com/cvdfoundation/mnist?utm_) から4ファイルを取得して展開し、プロジェクト直下の `data\` に配置してください。データはコード中で `./data` として参照されます。

## ファイル構成

```text
MNIST_recognize\
├── data\                         # MNIST データセット（Git 管理外）
├── results\
│   └── training_log.png
├── training.c                     # 学習プログラム
├── inference.c                    # 推論プログラム
├── mnist_loader.h                 # MNIST / BMP の読み込み処理
├── README.md
└── .gitignore
```

学習を実行すると、プロジェクト直下に次の学習済みパラメータが作成されます。

```text
fc1_parameter_adam.dat
fc2_parameter_adam.dat
fc3_parameter_adam.dat
```

## 実行方法

以降のコマンドは、PowerShell でこの README があるプロジェクト直下に移動してから実行してください。

### 1. 学習

学習プログラムをコンパイルします。

```powershell
gcc .\training.c -o .\training.exe -lm
```

実行します。

```powershell
.\training.exe
```

学習中には各 epoch の学習データ・テストデータの損失と正解率が表示され、終了後に上記の `.dat` ファイルが保存されます。

### 2. 推論

先に学習済みパラメータ（`fc1_parameter_adam.dat`、`fc2_parameter_adam.dat`、`fc3_parameter_adam.dat`）を用意してください。通常は前節の学習を完了すると作成されます。

推論プログラムをコンパイルします。

```powershell
gcc .\inference.c -o .\inference.exe -lm
```

BMP 画像を指定して実行します。

```powershell
.\inference.exe .\input.bmp
```

実行すると、次のように推論結果が表示されます。

```text
inference answer: 7
```

推論用画像は自分で用意してください。読み込み処理は非圧縮の 8 ビット、24 ビット、32 ビット BMP に対応しています。`input.bmp` はサンプル名であり、リポジトリには含まれません。

## 実装内容

- 全結合層（Fully Connected Layer）
- ReLU
- Softmax
- 交差エントロピー誤差
- 誤差逆伝播法（Backpropagation）
- ミニバッチ学習とデータシャッフル
- He 初期化
- Adam によるパラメータ更新
- 学習済みパラメータの保存と読み込み
- BMP 画像を用いた推論

## 学習結果

学習中の損失（loss）と精度（accuracy）の推移です。

![Training Results](results/training_log.png)

最終的な精度は 97.33% でした。
