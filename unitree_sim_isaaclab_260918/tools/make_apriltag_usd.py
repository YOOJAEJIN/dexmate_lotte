#!/usr/bin/env python3
"""Generate an AprilTag (tag36h11) texture and a textured-quad USD for Isaac Lab.

Run with the sim venv (needs OpenCV with the aruco module)::

    python tools/make_apriltag_usd.py --tag-id 0 --tag-size 0.24 \
        --out-dir assets/objects/apriltag

Outputs (in ``--out-dir``):

* ``tag36h11_<id>.png``      tag image, 8x8 cells (6 data + 1 black border per
                             side) plus a 1-cell white quiet zone -> 10x10 cells
* ``apriltag_36h11_<id>.usda`` quad mesh in the XY plane showing that texture.

Geometry / frame conventions (match OpenCV aruco + solvePnP IPPE_SQUARE):

* ``tag_size`` is the side of the **black square** (what pose estimation uses).
* The quad side is ``tag_size * 10 / 8`` because of the quiet zone.
* Quad centre = tag centre = prim origin; +X = image right, +Y = image up,
  +Z = out of the tag toward the viewer (normal). Rotate the prim with
  ``init_state.rot`` to aim +Z at the camera.
"""

from __future__ import annotations

import argparse
import os

import cv2

CELLS_TAG = 8       # tag36h11: 6 data cells + 1 black border cell per side
CELLS_QUIET = 1     # white quiet zone cells per side
PX_PER_CELL = 64

USDA_TEMPLATE = """#usda 1.0
(
    defaultPrim = "AprilTag"
    metersPerUnit = 1
    upAxis = "Z"
    doc = "AprilTag tag36h11 id {tag_id}; black-square side {tag_size:.4f} m; quad side {quad:.4f} m; frame: +X image right, +Y image up, +Z toward viewer"
)

def Xform "AprilTag"
{{
    def Mesh "Quad"
    {{
        uniform token subdivisionScheme = "none"
        bool doubleSided = 1
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [({mh}, {mh}, 0), ({h}, {mh}, 0), ({h}, {h}, 0), ({mh}, {h}, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
        float3[] extent = [({mh}, {mh}, 0), ({h}, {h}, 0)]
        rel material:binding = </AprilTag/Looks/TagMaterial>
    }}

    def Scope "Looks"
    {{
        def Material "TagMaterial"
        {{
            token outputs:surface.connect = </AprilTag/Looks/TagMaterial/Shader.outputs:surface>

            def Shader "Shader"
            {{
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor.connect = </AprilTag/Looks/TagMaterial/Texture.outputs:rgb>
                float inputs:roughness = 0.9
                float inputs:metallic = 0
                token outputs:surface
            }}

            def Shader "Primvar"
            {{
                uniform token info:id = "UsdPrimvarReader_float2"
                string inputs:varname = "st"
                float2 outputs:result
            }}

            def Shader "Texture"
            {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @./{png}@
                float2 inputs:st.connect = </AprilTag/Looks/TagMaterial/Primvar.outputs:result>
                token inputs:wrapS = "clamp"
                token inputs:wrapT = "clamp"
                float3 outputs:rgb
            }}
        }}
    }}
}}
"""


def make_png(tag_id: int, path: str) -> tuple[int, int]:
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = cv2.aruco.generateImageMarker(dictionary, tag_id, CELLS_TAG * PX_PER_CELL, borderBits=1)
    pad = CELLS_QUIET * PX_PER_CELL
    img = cv2.copyMakeBorder(tag, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    cv2.imwrite(path, img)
    return img.shape[1], img.shape[0]


def make_usda(tag_id: int, tag_size: float, png_name: str, path: str) -> float:
    quad = tag_size * (CELLS_TAG + 2 * CELLS_QUIET) / CELLS_TAG
    h = quad / 2.0
    text = USDA_TEMPLATE.format(
        tag_id=tag_id, tag_size=tag_size, quad=quad, h=f"{h:.5f}", mh=f"{-h:.5f}", png=png_name
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return quad


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag-id", type=int, default=0)
    ap.add_argument("--tag-size", type=float, default=0.24, help="black-square side in metres")
    ap.add_argument("--out-dir", default="assets/objects/apriltag")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    png_name = f"tag36h11_{args.tag_id:05d}.png"
    usda_name = f"apriltag_36h11_{args.tag_id:05d}.usda"
    w, h = make_png(args.tag_id, os.path.join(args.out_dir, png_name))
    quad = make_usda(args.tag_id, args.tag_size, png_name, os.path.join(args.out_dir, usda_name))
    print(f"wrote {png_name} ({w}x{h}px) and {usda_name}: tag {args.tag_size:.3f} m, quad {quad:.3f} m")


if __name__ == "__main__":
    main()
