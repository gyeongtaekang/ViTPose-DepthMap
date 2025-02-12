import argparse
import sys
import os
import os.path as osp
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from time import time
from tqdm import tqdm
from torchvision.transforms import transforms, Compose

# 프로젝트 경로 추가
base_dir = os.path.abspath(os.path.dirname(__file__))

# Depth-Anything 추가
depth_anything_path = os.path.join(base_dir, "Depth-Anything-main")
if depth_anything_path not in sys.path:
    sys.path.append(depth_anything_path)

# ViTPose 추가
vit_pose_path = os.path.join(base_dir, "ViTPose_pytorch-main")
if vit_pose_path not in sys.path:
    sys.path.append(vit_pose_path)

# 이제 ViTPose 모듈을 불러올 수 있음
from models.model import ViTPose
from utils.visualization import draw_points_and_skeleton, joints_dict
from utils.top_down_eval import keypoints_from_heatmaps

# Depth-Anything 관련 import
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# ViTPose config 불러오기
from configs.ViTPose_base_coco_256x192 import model as model_cfg
from configs.ViTPose_base_coco_256x192 import data_cfg


# --------------------------------------------------------------------------------
# 포즈 추정 함수
# --------------------------------------------------------------------------------
@torch.no_grad()
def run_pose_inference(image_path, pose_model, device, img_size, save_result=True, outdir="./pose_results"):
    """
    - image_path: 원본 이미지 경로
    - pose_model: 로드된 ViTPose 모델
    - device: CPU 또는 GPU
    - img_size: (width, height) 모델 입력 리사이즈 크기
    - save_result: 결과 이미지를 저장할지 여부
    - outdir: 결과 저장 경로
    """
    img_pil = Image.open(image_path)
    org_w, org_h = img_pil.size

    transform_pipe = transforms.Compose([
        transforms.Resize((img_size[1], img_size[0])),
        transforms.ToTensor()
    ])
    img_tensor = transform_pipe(img_pil).unsqueeze(0).to(device)

    # 모델 추론
    heatmaps = pose_model(img_tensor).detach().cpu().numpy()

    # heatmaps -> keypoints
    points, prob = keypoints_from_heatmaps(
        heatmaps=heatmaps,
        center=np.array([[org_w // 2, org_h // 2]]),
        scale=np.array([[org_w, org_h]]),
        unbiased=True,
        use_udp=True
    )

    # [y, x] -> [x, y], 그리고 confidence까지 합침
    # 최종 shape: (N, num_joints, 3) where [x, y, conf]
    points = np.concatenate([points[:, :, ::-1], prob], axis=2)

    # 원본 이미지(PIL) -> BGR np.array
    pose_draw_img = np.array(img_pil)[:, :, ::-1].copy()

    # keypoints & skeleton 그리기
    for pid, pts in enumerate(points):
        pose_draw_img = draw_points_and_skeleton(
            pose_draw_img,
            pts,
            joints_dict()['coco']['skeleton'],
            person_index=pid,
            points_color_palette='gist_rainbow',
            skeleton_color_palette='jet',
            points_palette_samples=10,
            confidence_threshold=0.4
        )

    if save_result:
        os.makedirs(outdir, exist_ok=True)
        filename = osp.basename(image_path)
        save_path = osp.join(outdir, filename.replace(".jpg", "_pose.jpg"))
        cv2.imwrite(save_path, pose_draw_img)

    return pose_draw_img, points


# --------------------------------------------------------------------------------
# 뎁스 추정 함수
# --------------------------------------------------------------------------------
@torch.no_grad()
def run_depth_inference(raw_image, depth_model, device, grayscale=False):
    """
    - raw_image: BGR numpy array
    - depth_model: 로드된 DepthAnything 모델
    - device: CPU 또는 GPU
    - grayscale: True이면 흑백 depth, False이면 컬러맵 적용
    """
    image_rgb = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    h, w = raw_image.shape[:2]

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

    depth_input = transform({"image": image_rgb})["image"]
    depth_input = torch.from_numpy(depth_input).unsqueeze(0).to(device)

    # 추론 및 원본 크기로 보간
    depth_pred = depth_model(depth_input)
    depth_pred = F.interpolate(depth_pred[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
    depth_pred = depth_pred.cpu().numpy().astype(np.uint8)

    # 컬러맵 or 그레이스케일
    if grayscale:
        depth_color = np.repeat(depth_pred[..., np.newaxis], 3, axis=-1)
    else:
        depth_color = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)

    return depth_color


# --------------------------------------------------------------------------------
# 뎁스맵 위에 포즈 스켈레톤 그리기
# --------------------------------------------------------------------------------
def overlay_pose_on_depth(depth_img, points):
    """
    - depth_img: 뎁스맵(BGR np.array)
    - points: (N, num_joints, 3) 포즈 추정 keypoints [x, y, conf]
    """
    overlay_img = depth_img.copy()
    for pid, pts in enumerate(points):
        overlay_img = draw_points_and_skeleton(
            overlay_img,
            pts,
            joints_dict()['coco']['skeleton'],
            person_index=pid,
            points_color_palette='gist_rainbow',   # 원하는 팔레트로 변경 가능
            skeleton_color_palette='jet',          # 원하는 팔레트로 변경 가능
            points_palette_samples=10,
            confidence_threshold=0.4
        )
    return overlay_img


# --------------------------------------------------------------------------------
# 메인 실행부
# --------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-path', type=str, required=True, help="이미지 파일 또는 폴더 경로")
    parser.add_argument('--outdir', type=str, default='./results', help='결과 저장 폴더')
    parser.add_argument('--pose-ckpt', type=str, default='runs/vitpose-b-multi-coco.pth', help='ViTPose 체크포인트 경로')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'], help='DepthAnything encoder')
    parser.add_argument('--grayscale', action='store_true', help='컬러맵 대신 흑백 depth를 저장')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f">>> Using device: {device}")

    # -------------------------- #
    # Load ViTPose
    # -------------------------- #
    vit_pose = ViTPose(model_cfg)
    ckpt = torch.load(args.pose_ckpt, map_location=device)
    vit_pose.load_state_dict(ckpt['state_dict'] if 'state_dict' in ckpt else ckpt)
    vit_pose.to(device).eval()
    print(f">>> Pose model loaded from: {args.pose_ckpt}")

    img_size = data_cfg['image_size']

    # -------------------------- #
    # Load DepthAnything
    # -------------------------- #
    depth_anything = DepthAnything.from_pretrained(
        f'LiheYoung/depth_anything_{args.encoder}14'
    ).to(device).eval()
    print(f">>> Depth model (encoder={args.encoder}) loaded.")

    # -------------------------- #
    # Gather image filenames
    # -------------------------- #
    if osp.isfile(args.image_path):
        filenames = [args.image_path]
    else:
        filenames = [
            osp.join(args.image_path, fn)
            for fn in os.listdir(args.image_path)
            if not fn.startswith('.') and fn.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        filenames.sort()

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------- #
    # Inference loop
    # -------------------------- #
    for img_file in tqdm(filenames, desc="Processing"):
        # 1) Run pose inference (저장 포함)
        pose_img, points = run_pose_inference(
            img_file,
            vit_pose,
            device,
            img_size,
            save_result=True,  # 포즈 이미지를 저장
            outdir=args.outdir
        )

        # 2) Run depth inference on the pose-img
        depth_img = run_depth_inference(pose_img, depth_anything, device, grayscale=args.grayscale)

        # 3) Save depth map
        base_name = osp.splitext(osp.basename(img_file))[0]
        depth_save_path = osp.join(args.outdir, f"{base_name}_depth.png")
        cv2.imwrite(depth_save_path, depth_img)

        # 4) 뎁스맵 위에 포즈 스켈레톤 그리기
        depth_pose_img = overlay_pose_on_depth(depth_img, points)

        # 5) 뎁스 + 포즈 결과 저장
        depth_pose_save_path = osp.join(args.outdir, f"{base_name}_depth_pose.png")
        cv2.imwrite(depth_pose_save_path, depth_pose_img)

    print(">>> Done! Results saved in:", args.outdir)


if __name__ == "__main__":
    main()

# 실행방법
# python pose_depth_unified.py \
#   --image-path ./examples/img1.jpg \
#   --outdir ./results \
#   --encoder vitl \
#   --pose-ckpt ./runs/vitpose-b-multi-coco.pth


# python pose_depth_unified.py --image-path ./ViTPose_pytorch-main/examples/dan.jpg --outdir ./results --encoder vitl --pose-ckpt ./ViTPose_pytorch-main/runs/vitpose-b-multi-coco.pth
