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

# ViTPose 폴더 경로
vit_pose_path = os.path.join(base_dir, "ViTPose_pytorch-main")
if vit_pose_path not in sys.path:
    sys.path.append(vit_pose_path)

# ViTPose_pytorch-main\utils 폴더 경로 (visualization.py가 이 안에 존재)
vit_pose_utils_path = os.path.join(vit_pose_path, "utils")
if vit_pose_utils_path not in sys.path:
    sys.path.append(vit_pose_utils_path)

# Depth-Anything 폴더 경로
depth_anything_path = os.path.join(base_dir, "Depth-Anything-main")
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

# -------------------------------------------------------------
# 이제 visualization.py 등 임포트 가능
# -------------------------------------------------------------
from visualization import (
    extended_joints_dict,
    add_pelvis_keypoint,
    draw_points_and_skeleton_joint_angles,
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


def run_pose_inference_on_frame(
    frame_bgr: np.ndarray,
    vit_pose: ViTPose,
    device: torch.device,
    frame_width: int,
    frame_height: int,
    img_size: tuple[int, int]  # e.g., (256, 192)
):
    """
    단일 프레임(BGR)에 대해:
      1) ViTPose 전처리
      2) 포즈 추론 (히트맵→keypoints)
      3) (x, y, conf) 형태의 keypoints (원본 해상도 기준 좌표) 반환
    """
    # BGR -> RGB -> PIL 변환
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # ex) (256,192)에 맞게 리사이즈 후 Tensor 변환
    pose_transform = transforms.Compose([
        transforms.Resize((img_size[1], img_size[0])),
        transforms.ToTensor()
    ])
    input_tensor = pose_transform(pil_img).unsqueeze(0).to(device)

    # 모델 추론
    with torch.no_grad():
        heatmaps = vit_pose(input_tensor).cpu().numpy()  # shape: (1, num_joints, h', w')

    # 히트맵 -> 키포인트(x, y, conf)
    # center, scale 은 "원본 프레임 해상도" 기준
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[frame_width // 2, frame_height // 2]]),
        scale=np.array([[frame_width, frame_height]]),
        unbiased=True,
        use_udp=True
    )
    # (N, num_joints, 3) 로 합침: [x, y, conf]
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)
    return points  # shape: (1, 17, 3)  (COCO 기본)


def run_depth_inference_on_frame(
    frame_bgr: np.ndarray,
    depth_model: DepthAnything,
    device: torch.device
) -> np.ndarray:
    """
    Depth-Anything를 이용해 단일 프레임(BGR)에 대한 뎁스맵 추론 & 컬러맵(BGR) 반환
    """
    h, w = frame_bgr.shape[:2]

    # BGR -> RGB [0,1]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) / 255.0

    # Depth-Anything transform
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    depth_input = transform({"image": frame_rgb})["image"]
    depth_input = torch.from_numpy(depth_input).unsqueeze(0).to(device)

    with torch.no_grad():
        depth_pred = depth_model(depth_input)

    # 원본 해상도로 보간
    depth_pred = F.interpolate(depth_pred[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_pred = depth_pred - depth_pred.min()
    max_val = depth_pred.max()
    if max_val > 1e-5:
        depth_pred = depth_pred / max_val
    depth_pred = depth_pred * 255.0
    depth_pred = depth_pred.cpu().numpy().astype(np.uint8)

    # 컬러맵 적용 (BGR)
    depth_color = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)
    return depth_color


def overlay_skeleton(
    frame_bgr: np.ndarray,
    points: np.ndarray,
    skeleton: list,
    conf_threshold: float = 0.4
) -> np.ndarray:
    """
    draw_points_and_skeleton_joint_angles 함수를 사용하여
    BGR 프레임 위에 스켈레톤 + 관절 각도 표시
    """
    out_frame = frame_bgr.copy()
    for pid, pts in enumerate(points):
        out_frame = draw_points_and_skeleton_joint_angles(
            out_frame,
            pts,  # [x, y, conf]
            skeleton,
            person_index=pid,
            points_color_palette='gist_rainbow',
            skeleton_color_palette='jet',
            points_palette_samples=10,
            confidence_threshold=conf_threshold
        )
    return out_frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True, help='입력 비디오 경로')
    parser.add_argument('--outdir', type=str, default='./results', help='결과 비디오 저장 폴더')
    parser.add_argument('--pose-ckpt', type=str, default='runs/vitpose-b-multi-coco.pth', help='ViTPose 체크포인트 경로')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='DepthAnything encoder')

    # 3D 계산용 카메라 파라미터 예시
    parser.add_argument('--fx', type=float, default=1000.0, help='focal length x')
    parser.add_argument('--fy', type=float, default=1000.0, help='focal length y')
    parser.add_argument('--cx', type=float, default=640.0, help='principal point x')
    parser.add_argument('--cy', type=float, default=360.0, help='principal point y')

    args = parser.parse_args()

    # -------------------------------------------------------------
    # 1) Device 설정
    # -------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device = {device}")

    # -------------------------------------------------------------
    # 2) ViTPose 모델 로드
    # -------------------------------------------------------------
    print("[INFO] Loading ViTPose...")
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(args.pose_ckpt, map_location=device)
    # ckpt에 'state_dict' 키가 있으면 사용
    vit_pose.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    vit_pose.to(device).eval()

    img_size = data_cfg['image_size']  # 예: (256, 192)

    # -------------------------------------------------------------
    # 3) Depth-Anything 모델 로드
    # -------------------------------------------------------------
    print("[INFO] Loading DepthAnything...")
    depth_model = DepthAnything.from_pretrained(
        f"LiheYoung/depth_anything_{args.encoder}14"
    ).to(device).eval()

    # -------------------------------------------------------------
    # 4) 입력 비디오 열기
    # -------------------------------------------------------------
    if not osp.isfile(args.video_path):
        print(f"[ERROR] Invalid video path: {args.video_path}")
        return

    os.makedirs(args.outdir, exist_ok=True)

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"[ERROR] Failed to open video: {args.video_path}")
        return

    # 비디오 정보
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # -------------------------------------------------------------
    # 5) 결과 비디오 설정
    # -------------------------------------------------------------
    base_name = osp.basename(args.video_path)
    stem, ext = osp.splitext(base_name)
    out_filename = f"{stem}_depth_pose{ext}"  # 결과 파일명
    out_path = osp.join(args.outdir, out_filename)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))

    print(f"[INFO] Output video: {out_path}")
    print(f"[INFO] Video size: {frame_width}x{frame_height}, FPS: {fps:.2f}, Frames: {total_frames}")

    # 확장된 coco skeleton (visualization.py 내부 함수)
    extended_coco = extended_joints_dict()

    # -------------------------------------------------------------
    # Pelvis 3D 결과를 담아둘 리스트
    # -------------------------------------------------------------
    pelvis_3d_results = []

    # -------------------------------------------------------------
    # 6) 프레임 단위 처리 (뎁스맵+확장 스켈레톤)
    # -------------------------------------------------------------
    pbar = tqdm(total=total_frames, desc="Processing")
    frame_index = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        frame_index += 1

        # (A) 포즈 추정 (기본 COCO 17개 관절)
        points_17 = run_pose_inference_on_frame(
            frame_bgr=frame_bgr,
            vit_pose=vit_pose,
            device=device,
            frame_width=frame_width,
            frame_height=frame_height,
            img_size=img_size
        )

        # (B) pelvis 추가 → 18개 관절
        points_18 = add_pelvis_keypoint(points_17)  # shape: (1,18,3)

        # (C) 뎁스맵 계산
        depth_map_bgr = run_depth_inference_on_frame(frame_bgr, depth_model, device=device)

        # (D) 스켈레톤 오버레이 (18개 관절)
        depth_pose = overlay_skeleton(
            depth_map_bgr,    # 배경: 뎁스맵
            points_18,        # 18개 관절
            extended_coco["skeleton"],
            conf_threshold=0.4
        )

        # (E) 3D 좌표 예시: pelvis(17)의 3D 계산 (화면에는 즉시 출력하지 않음)
        pelvis_2d = points_18[0, 17]  # [y, x, conf]
        if pelvis_2d[2] > 0.4:
            x_2d, y_2d = pelvis_2d[1], pelvis_2d[0]
            # pseudo-depth 예시 (컬러맵 → 그레이스케일)
            pseudo_depth = cv2.cvtColor(depth_map_bgr, cv2.COLOR_BGR2GRAY).astype(float)

            pelvis_3d = get_3d_coord(x_2d, y_2d, pseudo_depth, args.fx, args.fy, args.cx, args.cy)
            if pelvis_3d is not None:
                # Pelvis 3D 결과 리스트에 저장만 하고, 지금은 print 안 함
                pelvis_3d_results.append((frame_index, pelvis_3d))
                # 예: pelvis_3d_results.append(f"[Frame {frame_index}] Pelvis 3D: {pelvis_3d}")

        # (F) 결과 비디오 프레임 저장
        out_writer.write(depth_pose)
        pbar.update(1)

    pbar.close()
    cap.release()
    out_writer.release()
    print("[INFO] Done! Saved to:", out_path)

    # -------------------------------------------------------------
    # 7) Pelvis 3D 결과 한 번에 출력
    # -------------------------------------------------------------
    if pelvis_3d_results:
        print("\n--- Pelvis 3D 결과 (총 {}개) ---".format(len(pelvis_3d_results)))
        for frame_idx, p3d in pelvis_3d_results:
            print(f"[Frame {frame_idx}] Pelvis 3D: {p3d}")
    else:
        print("\nPelvis 3D를 추출할 수 없었습니다. (conf threshold 미달 등)")
    

if __name__ == "__main__":
    main()


# 실행명령어
# python video_pose_depth_unified.py --video-path .\ViTPose_pytorch-main\examples\walk.mp4 --outdir .\results --pose-ckpt .\ViTPose_pytorch-main\runs\vitpose-b-multi-coco.pth --encoder vitl
