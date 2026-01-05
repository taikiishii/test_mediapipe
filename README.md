# MediaPipe マルチモード検出プログラム

MediaPipeを使用したリアルタイム姿勢・顔・手・ジェスチャー検出プログラムです。

## 機能

**mediapipe_webcam.py** - Webカメラでのリアルタイム検出（4つのモード切り替え可能）

### 4つの検出モード

1. **姿勢検出（Pose Detection）**
   - 全身の33個の関節点を検出
   - 骨格構造を描画

2. **顔検出（Face Detection）**
   - 顔の478個の細部ランドマークを検出
   - 顔の輪郭、目、鼻、唇、眉毛、虹彩を色分けして描画

3. **手検出（Hand Detection）**
   - 1本の手につき21個の関節点を検出
   - 左手（青）右手（赤）を色分け表示
   - 最大2本の手を同時検出

4. **ジェスチャー認識（Gesture Recognition）**
   - 手のジェスチャーを認識
   - 対応ジェスチャー：Open_Palm、Closed_Fist、Pointing_Up、Thumbs_Up、Thumbs_Down、Victory、Love

## インストール

### 必要なパッケージ

```bash
pip install mediapipe opencv-python numpy
```

### バージョン確認

```bash
python -c "import mediapipe; print(mediapipe.__version__)"
```

## 使用方法

### 基本的な実行

```bash
# デフォルト：姿勢検出で開始
python mediapipe_webcam.py

# または明示的にモードを指定
python mediapipe_webcam.py --mode pose      # 姿勢検出
python mediapipe_webcam.py --mode face      # 顔検出
python mediapipe_webcam.py --mode hand      # 手検出
python mediapipe_webcam.py --mode gesture   # ジェスチャー認識
```

### キーボード操作

実行中のキーボード操作：

| キー | 機能 |
|------|------|
| `p` | 姿勢検出モードに切り替え |
| `f` | 顔検出モードに切り替え |
| `h` | 手検出モードに切り替え |
| `g` | ジェスチャー認識モードに切り替え |
| `q` | プログラムを終了 |

### 各モードの詳細

#### 姿勢検出モード（Pose Detection）

- 全身33個の関節を検出
- 関節を結ぶ骨格線を描画
- 関節位置を信頼度スコア付きで表示

検出される関節：
- **頭部**: 鼻、両目、両耳
- **上半身**: 両肩、両肘、両手首、両手、手指
- **下半身**: 両腰、両膝、両足首、両足

#### 顔検出モード（Face Detection）

- 顔の478個のランドマークを検出
- カラーコード化された特徴部位：
  - 黄色：顔の輪郭
  - 青色：鼻
  - 緑色：目
  - 赤色：唇
  - 紫色：眉毛
  - オレンジ色：虹彩

#### 手検出モード（Hand Detection）

- 最大2本の手を同時検出
- 1本の手につき21個の関節点
- 左手は青、右手は赤で表示
- 手の骨格構造を描画

#### ジェスチャー認識モード（Gesture Recognition）

- 手のジェスチャーを認識
- ジェスチャー名と信頼度スコア（0.0-1.0）を表示
- 左手のジェスチャーは画面左側に、右手は右側に表示

## 検出される関節（ランドマーク）

### 姿勢検出（33個）

```
0: Nose, 1: Left Eye Inner, 2: Left Eye, 3: Left Eye Outer
4: Right Eye Inner, 5: Right Eye, 6: Right Eye Outer
7: Left Ear, 8: Right Ear
9: Mouth Left, 10: Mouth Right
11: Left Shoulder, 12: Right Shoulder
13: Left Elbow, 14: Right Elbow
15: Left Wrist, 16: Right Wrist
17-21: Left Hand Fingers
22-26: Right Hand Fingers
27: Left Hip, 28: Right Hip
29: Left Knee, 30: Right Knee
31: Left Ankle, 32: Right Ankle
```

### 顔検出（478個）

- 顔の輪郭（68点）
- 両目（42点）
- 両眉（18点）
- 鼻（18点）
- 唇（20点）
- その他の顔特性点

### 手検出（21個/手）

```
0: Wrist
1-4: Thumb
5-8: Index Finger
9-12: Middle Finger
13-16: Ring Finger
17-20: Pinky Finger
```

## システム要件

- Python 3.7以上
- Webカメラ付きPC
- 推奨メモリ：4GB以上
- 推奨CPU：Intel Core i5以上（CPU処理時）
- GPU対応で高速化可能

## パフォーマンス最適化

### 処理速度が低い場合

1. モデルの複雑度を下げる
2. 入力画像の解像度を下げる
3. GPU対応版をインストール：
   ```bash
   pip install mediapipe-gpu
   ```

## トラブルシューティング

### Webカメラが認識されない

```bash
# ビデオデバイスの確認（Linux）
ls /dev/video*
```

### モデルファイルが見つからない

プログラム実行時に自動的にダウンロードされます。インターネット接続を確認してください。

- pose_landmarker_heavy.task（30MB）
- face_landmarker.task（3.6MB）
- hand_landmarker.task（7.5MB）
- gesture_recognizer.task（8.0MB）

### ジェスチャーが認識されない

- 手をはっきりとカメラに写してください
- 照明が十分であることを確認してください
- 信頼度スコアが低い場合は、別のジェスチャーの組み合わせを試してください

## 応用例

このプログラムを基に以下のような応用が可能です：

1. **フィットネス分析** - 運動時の姿勢チェック
2. **手話認識** - ジェスチャーの組み合わせ分析
3. **仮想キャラクター操作** - ARアプリケーション
4. **行動分析** - 統計的な姿勢パターン分析
5. **ジェスチャーコントロール** - PCやデバイスの操作

## 参考資料

- MediaPipe公式ドキュメント：https://mediapipe.dev/
- Google MediaPipe GitHub：https://github.com/google/mediapipe

## ライセンス

このコードはMediaPipeの公式モデルを使用しています。
詳細はMediaPipeの公式ライセンスを参照してください。
