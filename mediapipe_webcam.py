#!/usr/bin/env python3
"""
MediaPipeを使ったWebカメラでのリアルタイム姿勢・顔・手検出プログラム
コマンドラインオプションで切り替え可能
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
import numpy as np
import argparse
import os

# モデルファイルの保存ディレクトリ
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'models')

# modelsディレクトリが存在しない場合は作成
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)


def draw_pose_landmarks(image, landmarks):
    """姿勢ランドマークを画像に描画"""
    h, w, _ = image.shape
    
    # 接続点のペア（関節の接続関係）
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),  # 右腕
        (0, 5), (5, 6), (6, 7), (7, 8),  # 左腕
        (9, 10),  # 体
        (11, 12),  # 肩
        (11, 13), (13, 15),  # 左腕
        (12, 14), (14, 16),  # 右腕
        (11, 23), (12, 24),  # 胴体から腰
        (23, 25), (25, 27),  # 左脚
        (24, 26), (26, 28),  # 右脚
    ]
    
    # 接続線を描画
    for start, end in connections:
        if start < len(landmarks) and end < len(landmarks):
            start_pos = landmarks[start]
            end_pos = landmarks[end]
            
            if start_pos.presence > 0.5 and end_pos.presence > 0.5:
                start_x = int(start_pos.x * w)
                start_y = int(start_pos.y * h)
                end_x = int(end_pos.x * w)
                end_y = int(end_pos.y * h)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
    
    # ランドマークを描画
    for landmark in landmarks:
        if landmark.presence > 0.5:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)


def draw_face_landmarks(image, face_landmarks_list):
    """顔ランドマークを画像に描画"""
    from mediapipe.tasks.python import vision
    
    h, w, _ = image.shape
    
    for face_landmarks in face_landmarks_list:
        # 各部位の接続を正確に描画
        connections_map = {
            'face_oval': (vision.FaceLandmarksConnections.FACE_LANDMARKS_FACE_OVAL, (0, 255, 255)),  # 黄色
            'contours': (vision.FaceLandmarksConnections.FACE_LANDMARKS_CONTOURS, (255, 0, 0)),      # 青色
            'left_eye': (vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYE, (0, 255, 0)),      # 緑色
            'right_eye': (vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYE, (0, 255, 0)),    # 緑色
            'lips': (vision.FaceLandmarksConnections.FACE_LANDMARKS_LIPS, (0, 0, 255)),              # 赤色
            'left_eyebrow': (vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_EYEBROW, (128, 0, 255)),   # 紫色
            'right_eyebrow': (vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_EYEBROW, (128, 0, 255)),  # 紫色
            'left_iris': (vision.FaceLandmarksConnections.FACE_LANDMARKS_LEFT_IRIS, (0, 165, 255)),  # オレンジ
            'right_iris': (vision.FaceLandmarksConnections.FACE_LANDMARKS_RIGHT_IRIS, (0, 165, 255)),  # オレンジ
        }
        
        # 各部位の接続線を描画
        for part_name, (connections, color) in connections_map.items():
            for connection in connections:
                start_idx = connection.start
                end_idx = connection.end
                
                if start_idx < len(face_landmarks) and end_idx < len(face_landmarks):
                    start_landmark = face_landmarks[start_idx]
                    end_landmark = face_landmarks[end_idx]
                    
                    start_x = int(start_landmark.x * w)
                    start_y = int(start_landmark.y * h)
                    end_x = int(end_landmark.x * w)
                    end_y = int(end_landmark.y * h)
                    
                    cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
        
        # すべてのランドマークを小さい点で表示
        for landmark in face_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 2, (200, 200, 200), -1)  # グレー


def draw_hand_landmarks(image, hand_landmarks_list, handedness_list):
    """手ランドマークを画像に描画"""
    from mediapipe.tasks.python import vision
    
    h, w, _ = image.shape
    
    # 左手と右手で色を分ける
    hand_colors = {
        'Left': (255, 0, 0),   # 左手：青
        'Right': (0, 0, 255),  # 右手：赤
    }
    
    for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
        # 手の種類を取得
        hand_type = handedness[0].category_name
        color = hand_colors.get(hand_type, (0, 255, 0))
        
        # 手の接続線を描画
        hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in hand_connections:
            start_idx = connection.start
            end_idx = connection.end
            
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
        
        # ランドマークポイントを描画
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 4, color, -1)


def draw_gesture_landmarks(image, hand_landmarks_list, gestures_list, handedness_list):
    """ジェスチャーと手のランドマークを画像に描画"""
    from mediapipe.tasks.python import vision
    
    h, w, _ = image.shape
    
    # 左手と右手で色を分ける
    hand_colors = {
        'Left': (255, 0, 0),   # 左手：青
        'Right': (0, 0, 255),  # 右手：赤
    }
    
    for hand_landmarks, gesture_list, handedness in zip(hand_landmarks_list, gestures_list, handedness_list):
        # 手の種類を取得
        hand_type = handedness[0].category_name
        color = hand_colors.get(hand_type, (0, 255, 0))
        
        # 手の接続線を描画
        hand_connections = vision.HandLandmarksConnections.HAND_CONNECTIONS
        for connection in hand_connections:
            start_idx = connection.start
            end_idx = connection.end
            
            if start_idx < len(hand_landmarks) and end_idx < len(hand_landmarks):
                start_landmark = hand_landmarks[start_idx]
                end_landmark = hand_landmarks[end_idx]
                
                start_x = int(start_landmark.x * w)
                start_y = int(start_landmark.y * h)
                end_x = int(end_landmark.x * w)
                end_y = int(end_landmark.y * h)
                
                cv2.line(image, (start_x, start_y), (end_x, end_y), color, 2)
        
        # ランドマークポイントを描画
        for landmark in hand_landmarks:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            cv2.circle(image, (x, y), 4, color, -1)
        
        # ジェスチャー情報を表示
        if len(gesture_list) > 0:
            gesture = gesture_list[0]
            gesture_name = gesture.category_name if gesture.category_name else gesture.display_name
            confidence = gesture.score
            
            # 画面上に表示（英語）
            if gesture_name:
                text = f"{hand_type}: {gesture_name} ({confidence:.2f})"
                text_position = (50, 80) if hand_type == 'Left' else (w - 400, 80)
                cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)


def run_pose_detection(args):
    """姿勢検出モード"""
    # PoseLandmarkerオプション
    model_path = os.path.join(MODEL_DIR, 'pose_landmarker_heavy.task')
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=False
    )
    
    # Webカメラの初期化
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("エラー: Webカメラを開くことができません")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"姿勢検出モード開始")
    print(f"フレームサイズ: {frame_width}x{frame_height}, FPS: {fps}")
    print("'q'キーで終了、'f'キーで顔検出に切り替え、'h'キーで手検出に切り替え、'g'キーでジェスチャー認識に切り替え\n")
    
    frame_count = 0
    
    try:
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                
                if not success:
                    break
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                detection_result = landmarker.detect(mp_image)
                
                if detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]
                    draw_pose_landmarks(frame, landmarks)
                    
                    if len(landmarks) > 0:
                        nose = landmarks[0]
                        text = f"Frame: {frame_count}, Nose: ({nose.x:.2f}, {nose.y:.2f})"
                        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No pose detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, "Pose Detection | Press 'q' to quit, 'f' for face, 'h' for hand, 'g' for gesture", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("MediaPipe Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('f'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_face_detection(args)
                    return
                elif key == ord('h'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_hand_detection(args)
                    return
                elif key == ord('g'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_gesture_detection(args)
                    return
    
    except Exception as e:
        print(f"エラー: {e}")
        print("\n姿勢検出モデルをダウンロード中...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
        model_path = os.path.join(MODEL_DIR, 'pose_landmarker_heavy.task')
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✓ モデルをダウンロードしました")
            print("もう一度実行してください")
        except Exception as e2:
            print(f"✗ ダウンロード失敗: {e2}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("姿勢検出を終了しました")


def run_face_detection(args):
    """顔検出モード"""
    # FaceLandmarkerオプション
    model_path = os.path.join(MODEL_DIR, 'face_landmarker.task')
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False
    )
    
    # Webカメラの初期化
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("エラー: Webカメラを開くことができません")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"顔検出モード開始")
    print(f"フレームサイズ: {frame_width}x{frame_height}, FPS: {fps}")
    print("'q'キーで終了、'p'キーで姿勢検出に切り替え、'h'キーで手検出に切り替え、'g'キーでジェスチャー認識に切り替え\n")
    
    frame_count = 0
    
    try:
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                
                if not success:
                    break
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                detection_result = landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    draw_face_landmarks(frame, detection_result.face_landmarks)
                    text = f"Detected {len(detection_result.face_landmarks)} face(s)"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, "Face Detection | Press 'q' to quit, 'p' for pose, 'h' for hand", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("MediaPipe Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_pose_detection(args)
                    return
                elif key == ord('h'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_hand_detection(args)
                    return
                elif key == ord('g'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_gesture_detection(args)
                    return
    
    except Exception as e:
        print(f"エラー: {e}")
        print("\n顔検出モデルをダウンロード中...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
        model_path = os.path.join(MODEL_DIR, 'face_landmarker.task')
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✓ モデルをダウンロードしました")
            print("もう一度実行してください")
        except Exception as e2:
            print(f"✗ ダウンロード失敗: {e2}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("顔検出を終了しました")


def run_hand_detection(args):
    """手検出モード"""
    # HandLandmarkerオプション
    model_path = os.path.join(MODEL_DIR, 'hand_landmarker.task')
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=2
    )
    
    # Webカメラの初期化
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("エラー: Webカメラを開くことができません")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"手検出モード開始")
    print(f"フレームサイズ: {frame_width}x{frame_height}, FPS: {fps}")
    print("'q'キーで終了、'p'キーで姿勢検出に切り替え、'f'キーで顔検出に切り替え、'g'キーでジェスチャー認識に切り替え\n")
    
    frame_count = 0
    
    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                
                if not success:
                    break
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                detection_result = landmarker.detect(mp_image)
                
                if detection_result.hand_landmarks:
                    draw_hand_landmarks(frame, detection_result.hand_landmarks, detection_result.handedness)
                    text = f"Detected {len(detection_result.hand_landmarks)} hand(s)"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, "Hand Detection | Press 'q' to quit, 'p' for pose, 'f' for face, 'g' for gesture", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("MediaPipe Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_pose_detection(args)
                    return
                elif key == ord('f'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_face_detection(args)
                    return
                elif key == ord('g'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_gesture_detection(args)
                    return
    
    except Exception as e:
        print(f"エラー: {e}")
        print("\n手検出モデルをダウンロード中...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
        model_path = os.path.join(MODEL_DIR, 'hand_landmarker.task')
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✓ モデルをダウンロードしました")
            print("もう一度実行してください")
        except Exception as e2:
            print(f"✗ ダウンロード失敗: {e2}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("手検出を終了しました")


def run_gesture_detection(args):
    """ジェスチャー認識モード"""
    # GestureRecognizerオプション
    model_path = os.path.join(MODEL_DIR, 'gesture_recognizer.task')
    base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(base_options=base_options)
    
    # Webカメラの初期化
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("エラー: Webカメラを開くことができません")
        return
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    print(f"ジェスチャー認識モード開始")
    print(f"フレームサイズ: {frame_width}x{frame_height}, FPS: {fps}")
    print("'q'キーで終了、'p'キーで姿勢検出に切り替え、'f'キーで顔検出に切り替え、'h'キーで手検出に切り替え\n")
    
    frame_count = 0
    
    try:
        with vision.GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                success, frame = cap.read()
                
                if not success:
                    break
                
                frame_count += 1
                frame = cv2.flip(frame, 1)
                h, w, c = frame.shape
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                recognition_result = recognizer.recognize(mp_image)
                
                if recognition_result.gestures:
                    draw_gesture_landmarks(
                        frame,
                        recognition_result.hand_landmarks,
                        recognition_result.gestures,
                        recognition_result.handedness
                    )
                    text = f"Detected {len(recognition_result.gestures)} gesture(s)"
                    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "No gesture detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                cv2.putText(frame, "Gesture Recognition | Press 'q' to quit, 'p' for pose, 'f' for face, 'h' for hand", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.imshow("MediaPipe Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_pose_detection(args)
                    return
                elif key == ord('f'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_face_detection(args)
                    return
                elif key == ord('h'):
                    cv2.destroyAllWindows()
                    cap.release()
                    run_hand_detection(args)
                    return
    
    except Exception as e:
        print(f"エラー: {e}")
        print("\nジェスチャー認識モデルをダウンロード中...")
        import urllib.request
        url = 'https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task'
        model_path = os.path.join(MODEL_DIR, 'gesture_recognizer.task')
        try:
            urllib.request.urlretrieve(url, model_path)
            print("✓ モデルをダウンロードしました")
            print("もう一度実行してください")
        except Exception as e2:
            print(f"✗ ダウンロード失敗: {e2}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("ジェスチャー認識を終了しました")


def main():
    parser = argparse.ArgumentParser(description='MediaPipe リアルタイム検出 (姿勢 / 顔 / 手 / ジェスチャー)')
    parser.add_argument('--mode', type=str, choices=['pose', 'face', 'hand', 'gesture'], default='pose',
                        help='検出モード: pose(姿勢検出) / face(顔検出) / hand(手検出) / gesture(ジェスチャー認識) (デフォルト: pose)')
    args = parser.parse_args()
    
    if args.mode == 'pose':
        run_pose_detection(args)
    elif args.mode == 'face':
        run_face_detection(args)
    elif args.mode == 'hand':
        run_hand_detection(args)
    elif args.mode == 'gesture':
        run_gesture_detection(args)


if __name__ == "__main__":
    main()
