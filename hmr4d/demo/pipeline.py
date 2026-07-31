"""GVHMR demo pipeline stages, factored out of ``tools/demo/demo.py``.

These were previously module-level functions inside the non-importable CLI script.
Moving them into the ``hmr4d`` package makes the demo pipeline importable (see
:mod:`hmr4d.demo`) while the CLI in ``tools/demo/demo.py`` stays a thin wrapper.

Logic is unchanged from the original script; the only differences are:
- ``build_cfg`` takes explicit arguments instead of ``argparse`` (and gains an
  optional ``ckpt_path`` override so callers can point at checkpoints anywhere);
- bundled body-model tensors load via ``PROJ_ROOT`` so inference works when the
  package is pip-installed (CWD need not be the repo root);
- the ``renderer`` import is deferred into the render functions.

Every ``torch.load`` here passes ``weights_only=False`` explicitly. torch 2.6 flipped that
default to ``True``, which rejects any pickle containing a non-tensor — and the SLAM cache is a
numpy array, so a bare load raises ``UnpicklingError`` on every run under a modern torch. These
all read files GVHMR itself wrote (the per-video cache) or that it downloaded into the checkpoint
root, so full unpickling is the intended behaviour, but it now has to be stated. ``packaging.yml``
fails the build if a bare ``torch.load`` reappears in this package.
"""

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from einops import einsum
from dataclasses import dataclass, field
from types import SimpleNamespace
from hmr4d.utils.rotation_conversions import quaternion_to_matrix

from hmr4d import PROJ_ROOT, get_checkpoint_root
from hmr4d.utils.pylogger import Log
from hmr4d.utils.video_io_utils import (
    get_video_lwh,
    read_video_np,
    save_video,
    merge_videos_horizontal,
    get_writer,
    get_video_reader,
)
from hmr4d.utils.vis.cv2_utils import draw_bbx_xyxy_on_image_batch, draw_coco17_skeleton_batch
from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO
from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy, estimate_K, convert_K_to_K4, create_camera_sensor
from hmr4d.utils.geo_transform import compute_cam_angvel, apply_T_on_points, compute_T_ayfz2ay
from hmr4d.utils.net_utils import to_cuda
from hmr4d.utils.smplx_utils import make_smplx

CRF = 23  # 17 is lossless, every +6 halves the mp4 size

SMPLX2SMPL_PT = PROJ_ROOT / "hmr4d/utils/body_model/smplx2smpl_sparse.pt"
SMPL_NEUTRAL_J_REGRESSOR_PT = PROJ_ROOT / "hmr4d/utils/body_model/smpl_neutral_J_regressor.pt"


#: The demo's model spec, mirroring ``hmr4d/configs/demo.yaml`` and the group configs it pulls
#: in (``model/gvhmr/gvhmr_pl_demo``, ``network/gvhmr/relative_transformer``,
#: ``endecoder/gvhmr/v1_amass_local_bedlam_cam``). Written out rather than composed with Hydra so
#: inference never touches the global Hydra state — see :mod:`hmr4d.utils.instantiate`. Every
#: value here matches the resolved Hydra config; ``tests`` pins that, so the two cannot drift.
DEMO_MODEL_CFG = {
    "_target_": "hmr4d.model.gvhmr.gvhmr_pl_demo.DemoPL",
    "pipeline": {
        "_target_": "hmr4d.model.gvhmr.pipeline.gvhmr_pipeline.Pipeline",
        "args_denoiser3d": {
            "_target_": "hmr4d.network.gvhmr.relative_transformer.NetworkEncoderRoPE",
            "output_dim": 151,
            "max_len": 120,
            "cliffcam_dim": 3,
            "cam_angvel_dim": 6,
            "imgseq_dim": 1024,
            "latent_dim": 512,
            "num_layers": 12,
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "pred_cam_dim": 3,
            "static_conf_dim": 6,
            "dropout": 0.1,
            "avgbeta": True,
        },
        # SimpleNamespace, not a dict: Pipeline reads these by attribute (`args.weights`,
        # `args.normalize_cam_angvel`), which the DictConfig Hydra used to pass supported.
        "args": SimpleNamespace(
            endecoder_opt={
                "_target_": "hmr4d.model.gvhmr.utils.endecoder.EnDecoder",
                "stats_name": "MM_V1_AMASS_LOCAL_BEDLAM_CAM",
                "noise_pose_k": 10,
            },
            normalize_cam_angvel=True,
            weights=None,
            static_conf=None,
        ),
    },
}


@dataclass
class DemoPaths:
    """Where each stage reads and writes. Derived from ``output_dir`` exactly as demo.yaml did."""

    bbx: str
    bbx_xyxy_video_overlay: str
    vit_features: str
    vitpose: str
    vitpose_video_overlay: str
    hmr4d_results: str
    incam_video: str
    global_video: str
    incam_global_horiz_video: str
    slam: str


@dataclass
class DemoCfg:
    """One demo run's settings — the plain-Python replacement for the composed Hydra config."""

    video_name: str
    output_root: str
    output_dir: str
    preprocess_dir: str
    video_path: str
    ckpt_path: str
    paths: DemoPaths
    static_cam: bool = False
    verbose: bool = False
    use_dpvo: bool = False
    f_mm: float | None = None
    model: dict = field(default_factory=lambda: DEMO_MODEL_CFG)


def build_cfg(video, static_cam=False, verbose=False, use_dpvo=False, f_mm=None, output_root=None, ckpt_path=None):
    """Build the demo config for one video and copy the input into the output dir.

    Plain Python: no Hydra, so this is safe to call from a host process that owns Hydra's
    global state. Field-for-field equivalent to composing ``hmr4d/configs/demo.yaml``.
    """
    video_path = Path(video)
    assert video_path.exists(), f"Video not found at {video_path}"
    length, width, height = get_video_lwh(video_path)
    Log.info(f"[Input]: {video_path}")
    Log.info(f"(L, W, H) = ({length}, {width}, {height})")

    video_name = video_path.stem
    output_root = str(output_root) if output_root is not None else "outputs/demo"
    output_dir = f"{output_root}/{video_name}"
    preprocess_dir = f"{output_dir}/preprocess"
    cfg = DemoCfg(
        video_name=video_name,
        output_root=output_root,
        output_dir=output_dir,
        preprocess_dir=preprocess_dir,
        video_path=f"{output_dir}/0_input_video.mp4",
        ckpt_path=str(ckpt_path) if ckpt_path is not None else str(get_checkpoint_root() / "gvhmr/gvhmr_siga24_release.ckpt"),
        paths=DemoPaths(
            bbx=f"{preprocess_dir}/bbx.pt",
            bbx_xyxy_video_overlay=f"{preprocess_dir}/bbx_xyxy_video_overlay.mp4",
            vit_features=f"{preprocess_dir}/vit_features.pt",
            vitpose=f"{preprocess_dir}/vitpose.pt",
            vitpose_video_overlay=f"{preprocess_dir}/vitpose_video_overlay.mp4",
            hmr4d_results=f"{output_dir}/hmr4d_results.pt",
            incam_video=f"{output_dir}/1_incam.mp4",
            global_video=f"{output_dir}/2_global.mp4",
            incam_global_horiz_video=f"{output_dir}/{video_name}_3_incam_global_horiz.mp4",
            slam=f"{preprocess_dir}/slam_results.pt",
        ),
        static_cam=static_cam,
        verbose=verbose,
        use_dpvo=use_dpvo,
        f_mm=f_mm,
    )

    # Output
    Log.info(f"[Output Dir]: {cfg.output_dir}")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.preprocess_dir).mkdir(parents=True, exist_ok=True)

    # Copy raw-input-video to video_path
    Log.info(f"[Copy Video] {video_path} -> {cfg.video_path}")
    if not Path(cfg.video_path).exists() or get_video_lwh(video_path)[0] != get_video_lwh(cfg.video_path)[0]:
        reader = get_video_reader(video_path)
        writer = get_writer(cfg.video_path, fps=30, crf=CRF)
        for img in tqdm(reader, total=get_video_lwh(video_path)[0], desc=f"Copy"):
            writer.write_frame(img)
        writer.close()
        reader.close()

    return cfg


@torch.no_grad()
def run_preprocess(cfg):
    Log.info(f"[Preprocess] Start!")
    tic = Log.time()
    video_path = cfg.video_path
    paths = cfg.paths
    static_cam = cfg.static_cam
    verbose = cfg.verbose

    # Get bbx tracking result
    if not Path(paths.bbx).exists():
        tracker = Tracker()
        bbx_xyxy = tracker.get_one_track(video_path).float()  # (L, 4)
        bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()  # (L, 3) apply aspect ratio and enlarge
        torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, paths.bbx)
        del tracker
    else:
        bbx_xys = torch.load(paths.bbx, weights_only=False)["bbx_xys"]
        Log.info(f"[Preprocess] bbx (xyxy, xys) from {paths.bbx}")
    if verbose:
        video = read_video_np(video_path)
        bbx_xyxy = torch.load(paths.bbx, weights_only=False)["bbx_xyxy"]
        video_overlay = draw_bbx_xyxy_on_image_batch(bbx_xyxy, video)
        save_video(video_overlay, cfg.paths.bbx_xyxy_video_overlay)

    # Get VitPose
    if not Path(paths.vitpose).exists():
        vitpose_extractor = VitPoseExtractor()
        vitpose = vitpose_extractor.extract(video_path, bbx_xys)
        torch.save(vitpose, paths.vitpose)
        del vitpose_extractor
    else:
        vitpose = torch.load(paths.vitpose, weights_only=False)
        Log.info(f"[Preprocess] vitpose from {paths.vitpose}")
    if verbose:
        video = read_video_np(video_path)
        video_overlay = draw_coco17_skeleton_batch(video, vitpose, 0.5)
        save_video(video_overlay, paths.vitpose_video_overlay)

    # Get vit features
    if not Path(paths.vit_features).exists():
        extractor = Extractor()
        vit_features = extractor.extract_video_features(video_path, bbx_xys)
        torch.save(vit_features, paths.vit_features)
        del extractor
    else:
        Log.info(f"[Preprocess] vit_features from {paths.vit_features}")

    # Get visual odometry results
    if not static_cam:  # use slam to get cam rotation
        if not Path(paths.slam).exists():
            if not cfg.use_dpvo:
                simple_vo = SimpleVO(cfg.video_path, scale=0.5, step=8, method="sift", f_mm=cfg.f_mm)
                vo_results = simple_vo.compute()  # (L, 4, 4), numpy
                torch.save(vo_results, paths.slam)
            else:  # DPVO
                from hmr4d.utils.preproc.slam import SLAMModel

                length, width, height = get_video_lwh(cfg.video_path)
                K_fullimg = estimate_K(width, height)
                intrinsics = convert_K_to_K4(K_fullimg)
                slam = SLAMModel(video_path, width, height, intrinsics, buffer=4000, resize=0.5)
                bar = tqdm(total=length, desc="DPVO")
                while True:
                    ret = slam.track()
                    if ret:
                        bar.update()
                    else:
                        break
                slam_results = slam.process()  # (L, 7), numpy
                torch.save(slam_results, paths.slam)
        else:
            Log.info(f"[Preprocess] slam results from {paths.slam}")

    Log.info(f"[Preprocess] End. Time elapsed: {Log.time()-tic:.2f}s")


def load_data_dict(cfg):
    paths = cfg.paths
    length, width, height = get_video_lwh(cfg.video_path)
    if cfg.static_cam:
        R_w2c = torch.eye(3).repeat(length, 1, 1)
    else:
        # numpy array, so this is the one load that genuinely fails under the torch >= 2.6
        # weights_only=True default rather than merely being implicit. See the module docstring.
        traj = torch.load(cfg.paths.slam, weights_only=False)
        if cfg.use_dpvo:  # DPVO
            traj_quat = torch.from_numpy(traj[:, [6, 3, 4, 5]])
            R_w2c = quaternion_to_matrix(traj_quat).mT
        else:  # SimpleVO
            R_w2c = torch.from_numpy(traj[:, :3, :3])
    if cfg.f_mm is not None:
        K_fullimg = create_camera_sensor(width, height, cfg.f_mm)[2].repeat(length, 1, 1)
    else:
        K_fullimg = estimate_K(width, height).repeat(length, 1, 1)

    data = {
        "length": torch.tensor(length),
        "bbx_xys": torch.load(paths.bbx, weights_only=False)["bbx_xys"],
        "kp2d": torch.load(paths.vitpose, weights_only=False),
        "K_fullimg": K_fullimg,
        "cam_angvel": compute_cam_angvel(R_w2c),
        "f_imgseq": torch.load(paths.vit_features, weights_only=False),
    }
    return data


def render_incam(cfg):
    from hmr4d.utils.vis.renderer import Renderer

    incam_video_path = Path(cfg.paths.incam_video)
    if incam_video_path.exists():
        Log.info(f"[Render Incam] Video already exists at {incam_video_path}")
        return

    pred = torch.load(cfg.paths.hmr4d_results, weights_only=False)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(SMPLX2SMPL_PT, weights_only=False).cuda()
    faces_smpl = make_smplx("smpl").faces

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_incam"]))
    pred_c_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    K = pred["K_fullimg"][0]

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    reader = get_video_reader(video_path)  # (F, H, W, 3), uint8, numpy
    bbx_xys_render = torch.load(cfg.paths.bbx, weights_only=False)["bbx_xys"]

    # -- render mesh -- #
    verts_incam = pred_c_verts
    writer = get_writer(incam_video_path, fps=30, crf=CRF)
    for i, img_raw in tqdm(enumerate(reader), total=get_video_lwh(video_path)[0], desc=f"Rendering Incam"):
        img = renderer.render_mesh(verts_incam[i].cuda(), img_raw, [0.8, 0.8, 0.8])

        # # bbx
        # bbx_xys_ = bbx_xys_render[i].cpu().numpy()
        # lu_point = (bbx_xys_[:2] - bbx_xys_[2:] / 2).astype(int)
        # rd_point = (bbx_xys_[:2] + bbx_xys_[2:] / 2).astype(int)
        # img = cv2.rectangle(img, lu_point, rd_point, (255, 178, 102), 2)

        writer.write_frame(img)
    writer.close()
    reader.close()


def render_global(cfg):
    from hmr4d.utils.vis.renderer import Renderer, get_global_cameras_static, get_ground_params_from_points

    global_video_path = Path(cfg.paths.global_video)
    if global_video_path.exists():
        Log.info(f"[Render Global] Video already exists at {global_video_path}")
        return

    debug_cam = False
    pred = torch.load(cfg.paths.hmr4d_results, weights_only=False)
    smplx = make_smplx("supermotion").cuda()
    smplx2smpl = torch.load(SMPLX2SMPL_PT, weights_only=False).cuda()
    faces_smpl = make_smplx("smpl").faces
    J_regressor = torch.load(SMPL_NEUTRAL_J_REGRESSOR_PT, weights_only=False).cuda()

    # smpl
    smplx_out = smplx(**to_cuda(pred["smpl_params_global"]))
    pred_ay_verts = torch.stack([torch.matmul(smplx2smpl, v_) for v_ in smplx_out.vertices])

    def move_to_start_point_face_z(verts):
        "XZ to origin, Start from the ground, Face-Z"
        # position
        verts = verts.clone()  # (L, V, 3)
        offset = einsum(J_regressor, verts[0], "j v, v i -> j i")[0]  # (3)
        offset[1] = verts[:, :, [1]].min()
        verts = verts - offset
        # face direction
        T_ay2ayfz = compute_T_ayfz2ay(einsum(J_regressor, verts[[0]], "j v, l v i -> l j i"), inverse=True)
        verts = apply_T_on_points(verts, T_ay2ayfz)
        return verts

    verts_glob = move_to_start_point_face_z(pred_ay_verts)
    joints_glob = einsum(J_regressor, verts_glob, "j v, l v i -> l j i")  # (L, J, 3)
    global_R, global_T, global_lights = get_global_cameras_static(
        verts_glob.cpu(),
        beta=2.0,
        cam_height_degree=20,
        target_center_height=1.0,
    )

    # -- rendering code -- #
    video_path = cfg.video_path
    length, width, height = get_video_lwh(video_path)
    _, _, K = create_camera_sensor(width, height, 24)  # render as 24mm lens

    # renderer
    renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K)
    # renderer = Renderer(width, height, device="cuda", faces=faces_smpl, K=K, bin_size=0)

    # -- render mesh -- #
    scale, cx, cz = get_ground_params_from_points(joints_glob[:, 0], verts_glob)
    renderer.set_ground(scale * 1.5, cx, cz)
    color = torch.ones(3).float().cuda() * 0.8

    render_length = length if not debug_cam else 8
    writer = get_writer(global_video_path, fps=30, crf=CRF)
    for i in tqdm(range(render_length), desc=f"Rendering Global"):
        cameras = renderer.create_camera(global_R[i], global_T[i])
        img = renderer.render_with_ground(verts_glob[[i]], color[None], cameras, global_lights)
        writer.write_frame(img)
    writer.close()
