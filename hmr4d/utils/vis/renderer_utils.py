from tqdm import tqdm
import numpy as np

# `Renderer` pulls in pytorch3d's rasterizer, which needs the compiled CUDA extension. It is
# imported inside the functions below so that merely *importing* this module — which the Hydra
# config store does transitively, via the training datasets — does not make a pytorch3d build a
# requirement of every inference run.


def simple_render_mesh(render_dict):
    """Render an camera-space mesh, blank background"""
    from hmr4d.utils.vis.renderer import Renderer

    width, height, focal_length = render_dict["whf"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]

    renderer = Renderer(width, height, focal_length, device="cuda", faces=faces)
    outputs = []
    for i in tqdm(range(len(verts)), desc=f"Rendering"):
        img = renderer.render_mesh(verts[i].cuda(), colors=[0.8, 0.8, 0.8])
        outputs.append(img)
    outputs = np.stack(outputs, axis=0)
    return outputs


def simple_render_mesh_background(render_dict, VI=50, colors=[0.8, 0.8, 0.8]):
    """Render an camera-space mesh, blank background"""
    from hmr4d.utils.vis.renderer import Renderer

    K = render_dict["K"]
    faces = render_dict["faces"]
    verts = render_dict["verts"]
    background = render_dict["background"]
    N_frames = len(verts)
    if len(background.shape) == 3:
        background = [background] * N_frames
    height, width = background[0].shape[:2]

    renderer = Renderer(width, height, device="cuda", faces=faces, K=K)
    outputs = []
    for i in tqdm(range(len(verts)), desc=f"Rendering"):
        img = renderer.render_mesh(verts[i].cuda(), colors=colors, background=background[i], VI=VI)
        outputs.append(img)
    outputs = np.stack(outputs, axis=0)
    return outputs
