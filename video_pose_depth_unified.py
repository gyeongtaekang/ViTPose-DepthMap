# -*- coding: utf-8 -*-
import argparse
import sys
import os
import os.path as osp

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import transforms, Compose
from tqdm import tqdm

# -------------------------------------------------------------
# [1] 경로 설정 (visualization.py 위치)
# -------------------------------------------------------------
base_dir = os.path.abspath(os.path.dirname(__file__))

# 예: ViTPose 폴더
vit_pose_path = os.path.join(base_dir, "ViTPose_pytorch-main")
if vit_pose_path not in sys.path:
    sys.path.append(vit_pose_path)

# ViTPose_pytorch-main/utils 폴더
vit_pose_utils_path = os.path.join(vit_pose_path, "utils")
if vit_pose_utils_path not in sys.path:
    sys.path.append(vit_pose_utils_path)

# Depth-Anything 폴더
depth_anything_path = os.path.join(base_dir, "Depth-Anything-main")
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

# -------------------------------------------------------------
# visualization.py 임포트
# -------------------------------------------------------------
from visualization import (
    # 원본 17개 관절만 사용할 것이므로 draw_points_and_skeleton를 사용
    draw_points_and_skeleton,
    get_3d_coord
)

# ViTPose 관련
from models.model import ViTPose
from utils.top_down_eval import keypoints_from_heatmaps
from configs.ViTPose_base_coco_256x192 import model as model_cfg
from configs.ViTPose_base_coco_256x192 import data_cfg

# Depth-Anything 관련
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


# -------------------------------------------------------------
# (추가) 3D 각도 계산용 함수
# -------------------------------------------------------------
def compute_3d_angle(p1_3d, p2_3d, p3_3d):
    """
    p2_3d를 기준으로 p1_3d, p3_3d를 연결하는 두 벡터의 3차원 각도(도 단위)를 계산.
      - p1_3d, p2_3d, p3_3d: (x, y, z) 형태의 numpy 배열
    """
    v1 = p1_3d - p2_3d
    v2 = p3_3d - p2_3d

    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return None

    dot = np.dot(v1, v2)
    cos_angle = dot / (norm1 * norm2)
    cos_angle = max(min(cos_angle, 1.0), -1.0)

    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)
    return angle_deg


def project_3d_to_2d(X, Y, Z, fx, fy, cx, cy):
    """
    3D 좌표 -> 2D 좌표로 역투영
    """
    if Z < 1e-5:
        return None
    x_2d = (X / Z) * fx + cx
    y_2d = (Y / Z) * fy + cy
    return (y_2d, x_2d)


def calculate_3d_angle_for_hip(
    hip_idx,           # 왼엉덩이=11, 오른엉덩이=12
    depth_map_float,   # Depth-Anything 결과 (0~1 정규화)
    points_2d,         # (17,3) = [y, x, conf]
    fx, fy, cx, cy,
    offset_front,      # 앞쪽 offset
    offset_back,       # 뒤쪽 offset
    conf_threshold=0.4
):
    """
    원본 17개 관절 중 hip_idx(11 or 12)를 기준으로,
    Z를 ±offset 한 'front/back' 점을 2D 재투영 -> 다시 3D를 얻어
    compute_3d_angle(front, hip, back)을 계산.
    """
    # 1) confidence 체크
    if points_2d[hip_idx, 2] < conf_threshold:
        return None

    # 2) 해당 hip의 2D -> 3D
    y_hip, x_hip, _ = points_2d[hip_idx]
    hip_3d = get_3d_coord(x_hip, y_hip, depth_map_float, fx, fy, cx, cy)
    if hip_3d is None:
        return None

    # 3) hip_3d에서 Z를 ±offset
    X, Y, Z = hip_3d

    # front (Z - offset_front)
    front_3d_temp = (X, Y, Z - offset_front)
    front_2d = project_3d_to_2d(*front_3d_temp, fx, fy, cx, cy)
    if front_2d is None:
        return None
    y_front, x_front = front_2d
    front_3d = get_3d_coord(x_front, y_front, depth_map_float, fx, fy, cx, cy)
    if front_3d is None:
        return None

    # back (Z + offset_back)
    back_3d_temp = (X, Y, Z + offset_back)
    back_2d = project_3d_to_2d(*back_3d_temp, fx, fy, cx, cy)
    if back_2d is None:
        return None
    y_back, x_back = back_2d
    back_3d = get_3d_coord(x_back, y_back, depth_map_float, fx, fy, cx, cy)
    if back_3d is None:
        return None

    # 4) 3D 각도 계산
    angle = compute_3d_angle(np.array(front_3d), np.array(hip_3d), np.array(back_3d))
    return angle


# -------------------------------------------------------------
# (1) ViTPose 추론 함수 (원본 17개 관절)
# -------------------------------------------------------------
def run_pose_inference_on_frame(
    frame_bgr,
    vit_pose,
    device,
    frame_width,
    frame_height,
    img_size
):
    """
    ViTPose로 17개 관절 추론 → (x, y, conf)
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)
    pose_transform = transforms.Compose([
        transforms.Resize((img_size[1], img_size[0])),
        transforms.ToTensor()
    ])
    input_tensor = pose_transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        heatmaps = vit_pose(input_tensor).cpu().numpy()

    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[frame_width // 2, frame_height // 2]]),
        scale=np.array([[frame_width, frame_height]]),
        unbiased=True,
        use_udp=True
    )
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)  # shape: (1,17,3)
    return points


# -------------------------------------------------------------
# (2) Depth-Anything 추론 함수
# -------------------------------------------------------------
def run_depth_inference_on_frame(frame_bgr, depth_model, device):
    """
    Depth-Anything로 뎁스 추론 → (뎁스 float 배열, 컬러맵)
    """
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0

    transform = Compose([
        Resize(
            width=518, height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    depth_input = transform({"image": frame_rgb})["image"]
    depth_input = torch.from_numpy(depth_input).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_pred = depth_model(depth_input)  # (1,H,W)

    # 원본 해상도로 보간
    depth_pred = F.interpolate(
        depth_pred[None], (h, w),
        mode='bilinear', align_corners=False
    )[0, 0]

    # min-max 정규화 (0~1)
    depth_pred = depth_pred - depth_pred.min()
    max_val = depth_pred.max()
    if max_val > 1e-5:
        depth_pred = depth_pred / max_val

    depth_8u = (depth_pred * 255.0).cpu().numpy().astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_8u, cv2.COLORMAP_INFERNO)
    depth_float = depth_pred.cpu().numpy()

    return depth_float, depth_color


# -------------------------------------------------------------
# (3) 스켈레톤 + 관절점 시각화 (각도 텍스트 없음)
# -------------------------------------------------------------
def overlay_skeleton_no_angles(frame_bgr, points_list, skeleton, conf_threshold=0.4):
    """
    원본 17개 관절 + 스켈레톤을 그리되, 각도 표시 없이 출력
    """
    out_frame = frame_bgr.copy()
    for pid, pts in enumerate(points_list):
        out_frame = draw_points_and_skeleton(
            out_frame,
            pts,  # (N,3)
            skeleton,
            person_index=pid,
            confidence_threshold=conf_threshold
        )
    return out_frame


# -------------------------------------------------------------
# (4) 메인 함수
# -------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True)
    parser.add_argument('--outdir', type=str, default='./results')
    parser.add_argument('--pose-ckpt', type=str, default='runs/vitpose-b-multi-coco.pth')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])

    # 3D pinhole 파라미터
    parser.add_argument('--fx', type=float, default=1000.0)
    parser.add_argument('--fy', type=float, default=1000.0)
    parser.add_argument('--cx', type=float, default=640.0)
    parser.add_argument('--cy', type=float, default=360.0)

    # 각도 계산 시 offset (단위: m)
    parser.add_argument('--offset-front', type=float, default=0.05)
    parser.add_argument('--offset-back', type=float, default=0.05)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # (A) ViTPose 로드
    print("[INFO] Loading ViTPose...")
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(args.pose_ckpt, map_location=device)
    vit_pose.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    vit_pose.to(device).eval()
    img_size = data_cfg['image_size']  # (256,192) 예시

    # (B) Depth-Anything 로드
    print("[INFO] Loading DepthAnything...")
    depth_model = DepthAnything.from_pretrained(
        f"LiheYoung/depth_anything_{args.encoder}14"
    ).to(device).eval()

    # (C) 입력 비디오 열기
    if not osp.isfile(args.video_path):
        print(f"[ERROR] Invalid video path: {args.video_path}")
        return
    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {args.video_path}")
        return

    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # (D) 결과 비디오 설정
    base_name   = osp.basename(args.video_path)
    stem, ext   = osp.splitext(base_name)
    out_name    = f"{stem}_depth_pose_17{ext}"
    out_path    = osp.join(args.outdir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    print(f"[INFO] Output video: {out_path}")
    print(f"[INFO] Video size: {frame_width}x{frame_height}, FPS: {fps}, Frames: {total_frames}")

    # (E) COCO 원본 17개 관절 스켈레톤 정의
    skeleton_17 = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
        [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
        [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4],
        [0, 5], [0, 6]
    ]

    pbar = tqdm(total=total_frames, desc="Processing")
    frame_idx = 0

    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_idx += 1

        # 1) 포즈 추론 (17개 관절)
        points_17 = run_pose_inference_on_frame(
            frame_bgr,
            vit_pose,
            device,
            frame_width,
            frame_height,
            img_size
        )
        pts_17 = points_17[0]  # (17,3)

        # 2) 뎁스 맵 추론
        depth_float, depth_color_bgr = run_depth_inference_on_frame(
            frame_bgr, depth_model, device
        )

        # 3) 스켈레톤(17개) 그리기 (각도는 표시 안 함)
        depth_pose_vis = overlay_skeleton_no_angles(
            depth_color_bgr,
            points_17,   # 17개
            skeleton_17,
            conf_threshold=0.4
        )

        # ---------------------------------------------------------
        # (★) 오른쪽/왼쪽 엉덩이 각도만 계산해서 텍스트로 표시
        # 오른쪽 엉덩이(hip_idx=12) 각도
        angle_right = calculate_3d_angle_for_hip(
            hip_idx=12,
            depth_map_float=depth_float,
            points_2d=pts_17,
            fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
            offset_front=args.offset_front,
            offset_back=args.offset_back,
            conf_threshold=0.4
        )
        if angle_right is not None:
            y_r, x_r, _ = pts_17[12]
            cv2.putText(
                depth_pose_vis,
                f"{angle_right:.1f} deg",
                (int(x_r), int(y_r) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
            )

        # 왼쪽 엉덩이(hip_idx=11) 각도
        angle_left = calculate_3d_angle_for_hip(
            hip_idx=11,
            depth_map_float=depth_float,
            points_2d=pts_17,
            fx=args.fx, fy=args.fy, cx=args.cx, cy=args.cy,
            offset_front=args.offset_front,
            offset_back=args.offset_back,
            conf_threshold=0.4
        )
        if angle_left is not None:
            y_l, x_l, _ = pts_17[11]
            cv2.putText(
                depth_pose_vis,
                f"{angle_left:.1f} deg",
                (int(x_l), int(y_l) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2
            )
        # ---------------------------------------------------------

        # 4) 결과 프레임 저장
        out_writer.write(depth_pose_vis)
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()

    print("[INFO] Done! Saved to:", out_path)


if __name__ == "__main__":
    main()

# 실행명령어 예시:
# python video_pose_depth_unified.py --video-path .\ViTPose_pytorch-main\examples\walk.mp4 --outdir .\results --pose-ckpt .\ViTPose_pytorch-main\runs\vitpose-b-multi-coco.pth --encoder vitl



# 실행명령어 예시:
# python video_pose_depth_unified.py --video-path .\ViTPose_pytorch-main\examples\walk.mp4 --outdir .\results --pose-ckpt .\ViTPose_pytorch-main\runs\vitpose-b-multi-coco.pth --encoder vitl


# 실행명령어
# python video_pose_depth_unified.py --video-path .\ViTPose_pytorch-main\examples\walk.mp4 --outdir .\results --pose-ckpt .\ViTPose_pytorch-main\runs\vitpose-b-multi-coco.pth --encoder vitl
