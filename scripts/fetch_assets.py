#!/usr/bin/env python3
"""Fetch pinned Piper, wrist-camera and table assets; preserve existing files.

No Isaac Sim imports are needed. Existing valid files require no network access.
A conflicting file is reported, never overwritten. Sources and checksums are
recorded in assets/sources.lock.json. NVIDIA assets retain NVIDIA's asset terms;
robot_lab's Apache-2.0 license is retained alongside its Piper assets. The
separate piper_isaac_sim repository has no root license file; its actual package
declarations and Intel camera notices are retained without inventing a license.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
ROBOT_COMMIT = "b868140eeb1459acefef24a865587ec39a5278c3"
ROBOT_URL = f"https://raw.githubusercontent.com/agilexrobotics/robot_lab/{ROBOT_COMMIT}/"
PIPER_ISAAC_COMMIT = "8e1f88fdb7afca49c40e9a0c1c01cc588e86f0d2"
PIPER_ISAAC_URL = f"https://raw.githubusercontent.com/agilexrobotics/piper_isaac_sim/{PIPER_ISAAC_COMMIT}/"
ISAAC_URL = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/4.5/Isaac/"
SOURCES = {
    "robot_lab": ("vendor/robot_lab", ROBOT_URL),
    "piper_isaac_sim": ("vendor/piper_isaac_sim", PIPER_ISAAC_URL),
    "isaac_sim": ("assets/isaac", ISAAC_URL),
}
# SHA-256 values were checked against the pinned upstream checkout/downloads.
FILES = [
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/.asset_hash",
        "sha256": "512414a49a5b7b46bd6c1d9c32a2aa75a9bfa8a85fa87fc81a4b5a2c1b2fd484",
        "bytes": 32
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/config.yaml",
        "sha256": "fd6605a3fd617fd431e4cd59d29d8346ebba13f983bebd51487f35329f961a67",
        "bytes": 695
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/configuration/piper_base.usd",
        "sha256": "441ab91764629f9a1cf4ed31628623425175e6da30a0834754f769238d072fe0",
        "bytes": 13238960
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/configuration/piper_physics.usd",
        "sha256": "81ae471b38850d9f1f28e869dee30d86c73d38d2d82ba7d78b984ce5c7f3dff6",
        "bytes": 5546
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/configuration/piper_sensor.usd",
        "sha256": "04b0d38d0540afb8976f66e3d2b87d5560651e2e17f310f396b30d2fef9af983",
        "bytes": 645
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/base_link.STL",
        "sha256": "fd4a3d6d0266205fe3f725f048645be66592c23564db22e08b0807d5569b2c64",
        "bytes": 607384
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/gripper_base.STL",
        "sha256": "2895bfa6f1eef2d0ad2cb9dfdedd5f6c760ec07388bcde2b4d1e412e1189c9cb",
        "bytes": 649484
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link1.STL",
        "sha256": "9ee6e1d67c0241f42a0dbaf5c676d19230375ed078ee1943bb8c4c37ab303d14",
        "bytes": 447784
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link2.STL",
        "sha256": "0d37675b699b643f2e15348de99aed535fac10dc5f09a79b8e4b0acfaf4e0378",
        "bytes": 3658384
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link3.STL",
        "sha256": "e8ae51f17c0c7f6d67b753922e2cc269f5e8372a65734f859231639a64fe8fad",
        "bytes": 1845784
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link4.STL",
        "sha256": "efa33022976218f246f736a6133494b5d0e5486e8b50b52900c42f285ae85558",
        "bytes": 877984
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link5.STL",
        "sha256": "a1c69d1c7f6f8c60e2c82331c0f444725200ed4053d3e5ec23aa82cb5db5b29f",
        "bytes": 833684
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link6.STL",
        "sha256": "88374973bc3bd9e7389b5853cd6e555af9731c30b658b1ee6f85be34a126cb92",
        "bytes": 58884
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link7.STL",
        "sha256": "475a7d8db056524d7659fadf5a2c307d62791047b6214d733dfc7846ed1c613d",
        "bytes": 104384
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/meshes/link8.STL",
        "sha256": "3f88d534a85e28a33ef7d2061adacbd1d25214aa9325f9e464aad0f9e4b74606",
        "bytes": 104384
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/piper.usd",
        "sha256": "1c19fa993984bc476120d84dcfeb50356fac4f6551a34ffae344b3cc43c48623",
        "bytes": 1592
    },
    {
        "source": "robot_lab",
        "path": "source/robot_lab/data/Robots/Agilex/PIPER/piper_description.urdf",
        "sha256": "01471dc1766d7ecaea268a4bdc15615e6ed95b969a42b32b9302a2c77095c439",
        "bytes": 12889
    },
    {
        "source": "robot_lab",
        "path": "LICENSE",
        "sha256": "6c902f2125ac13341ec3bd7a8e2a06550926b58a056a7a223172e6501370b70b",
        "bytes": 11343
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/MI_Table.mdl",
        "sha256": "761114022d945ee6fae32174fe47d8fa100b33443d75488bf3458469b7d3c931",
        "bytes": 2329
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/Textures/DefaultMaterial_Base_Color.png",
        "sha256": "49a93f3124cc628d579ac7ad39708ef1692c642668fd9ae784264dd728069bfd",
        "bytes": 309606
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/Textures/DefaultMaterial_Mixed_AO.png",
        "sha256": "1a53593fa84590c8339714c247a7b5b71bd578ce72fc0168b1e335efb0a17cd5",
        "bytes": 1110839
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/Textures/DefaultMaterial_Normal_DirectX.png",
        "sha256": "12aec50406cf9ba8f5c32cb27d35dd1af367433a7cd91de5d191fe0ddec57009",
        "bytes": 2052839
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/Textures/DefaultMaterial_Roughness.png",
        "sha256": "0ab8364b5adfab4bd74ac4df359eb271595b3f0593eaf7e50ffd539b6155ad7c",
        "bytes": 87313
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/Textures/OmniUe4Base.mdl",
        "sha256": "505961c8fc04f949f6cd02d99b878d2eaed050873ec6af8987d2656167878373",
        "bytes": 8443
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Materials/Textures/OmniUe4Function.mdl",
        "sha256": "4afd5f8cbee83e8f9a4520409917b2b72fcb5cbc48ade66580b8453c4a2c1726",
        "bytes": 46370
    },
    {
        "source": "isaac_sim",
        "path": "Environments/Simple_Room/Props/table_low.usd",
        "sha256": "9843b47c308706a3e17543910285d75d841fa095571e1096ea5c1324e42702bc",
        "bytes": 105767
    },
    {
        "source": "isaac_sim",
        "path": "Props/Mounts/Stand/stand.usd",
        "sha256": "9788fa192ae3a28117daa6f99a38375d3d1c9ae844f58236aa0bfaae33b6b8ab",
        "bytes": 2679034
    },
    {
        "source": "piper_isaac_sim",
        "path": "piper_description/urdf/piper_description_v100_realsense_camera_v2/configuration/piper_description_v100_realsense_camera_v2_base.usd",
        "sha256": "7561a29c320017aab66ca8a067af48722e6644e47598a2315bde8eb8e9738ec4",
        "bytes": 41020584
    },
    {
        "source": "piper_isaac_sim",
        "path": "piper_description/urdf/piper_description_v100_realsense_camera_v2.urdf",
        "sha256": "d21046ed1614b8077ce00b5309beae4fa4fbc7d3b654c3f500c815deec90d287",
        "bytes": 12070
    },
    {
        "source": "piper_isaac_sim",
        "path": "piper_description/meshes/dae/realsense_mid_stand.dae",
        "sha256": "1b9253505836723353528b41c7008c4d8f539a774959da3ad9669c26ae62e45a",
        "bytes": 713245
    },
    {
        "source": "piper_isaac_sim",
        "path": "realsense2_description/meshes/d435.dae",
        "sha256": "42f3b66f47a1f8f425a2e4dc07c1d9c283183167d8441f520a15623d98f9bf78",
        "bytes": 15782439
    },
    {
        "source": "piper_isaac_sim",
        "path": "realsense2_description/urdf/_d435.urdf.xacro",
        "sha256": "db7ff24c900fd68d9841bd317d4b5077aa2fd3c1091fa2416f7a10048ba2132b",
        "bytes": 7550
    },
    {
        "source": "piper_isaac_sim",
        "path": "README.md",
        "sha256": "95680571b7a070936d01619870b4b492a8433aa60ceb6dfd5fe4e83e9aea66df",
        "bytes": 712
    },
    {
        "source": "piper_isaac_sim",
        "path": "piper_description/package.xml",
        "sha256": "307ddcda650974988726809436f3aeb63d19db55174ac5976d4869ddbd994177",
        "bytes": 901
    },
    {
        "source": "piper_isaac_sim",
        "path": "realsense2_description/package.xml",
        "sha256": "73c0fe593f574dd3cea0e57378c437c2671d3fdb2db2d32341d599799ee2be59",
        "bytes": 1048
    }
]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(record, root, offline=False):
    directory, source_url = SOURCES[record["source"]]
    relative = Path(directory) / record["path"]
    destination = root / relative
    url = source_url + record["path"]
    if destination.exists():
        if sha256(destination) != record["sha256"]:
            raise RuntimeError(f"Checksum mismatch; existing file preserved: {destination}")
        status = "verified_existing"
    else:
        if offline:
            raise RuntimeError(f"Missing asset in --check-only mode: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            request = Request(url, headers={"User-Agent": "PressB-assets/1.0"})
            with urlopen(request, timeout=120) as response, tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".download-", delete=False) as stream:
                temporary = Path(stream.name)
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            if sha256(temporary) != record["sha256"]:
                raise RuntimeError(f"Downloaded checksum differs from pinned asset: {url}")
            # Check a second time, in case another process completed the download.
            if destination.exists():
                if sha256(destination) != record["sha256"]:
                    raise RuntimeError(f"Concurrent change preserved: {destination}")
            else:
                temporary.rename(destination)
            status = "downloaded"
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
    return {
        **record, "local_path": relative.as_posix(), "source_url": url,
        "available": True, "verification": status,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Verify every checksum without using the network")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root for fetching or verifying assets")
    args = parser.parse_args()
    root = args.root.resolve()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda record: fetch(record, root, args.check_only), FILES))
    report = {
        "schema_version": 1,
        "robot_repository": "https://github.com/agilexrobotics/robot_lab",
        "robot_revision": ROBOT_COMMIT,
        "wrist_asset_repository": "https://github.com/agilexrobotics/piper_isaac_sim",
        "wrist_asset_revision": PIPER_ISAAC_COMMIT,
        "wrist_asset_licenses": {
            "repository_root_license_file": None,
            "piper_description_package_declaration": "TODO: License declaration",
            "realsense2_description_package_declaration": "Apache License 2.0",
            "intel_notice_file": "vendor/piper_isaac_sim/realsense2_description/urdf/_d435.urdf.xacro",
            "note": "The pinned upstream has no LICENSE, COPYING or NOTICE file. Its actual package.xml declarations and Intel source header are retained; no license is inferred for the Piper bracket geometry.",
        },
        "isaac_asset_version": "4.5",
        "catalog_inventory": "assets/isaac/asset_inventory.json",
        "selection": {
            "global_camera_stand": "Reuse the original NVIDIA Props/Mounts/Stand/stand.usd mesh with transform scaling to a configured tabletop footprint (currently 0.18 m square) and the required support height beside the robot on its left (+Y). The complete foot rests on the tabletop; the local source mesh remains unchanged. Retain the scaled static column collider and matching plate proxies, plus a small ball-head interface connected to the official D435 bottom screw. Camera housing uses the same source-locked, geometry-preserving compatible D435 mesh and nominal RGB optics as the wrist sensor.",
            "table": "Reuse Simple_Room/Props/table_low.usd and its official texture set as the stand. Normalize its measured mesh bounds to the specified footprint and height minus 40 mm, then add a level 40 mm tabletop flush with the exact collision surface and robot base.",
            "alternatives_inspected": ["Props/Mounts/SeattleLabTable", "Props/PackingTable", "Environments/Simple_Room"],
            "panel": "The inspected stock catalogs do not provide this 2-column floor-24-to-35 actuated panel. Author task-specific caps, spring joints, numeric legends and feedback materials.",
            "elevator_wall": "Task-specific collision-safe cladding and panel arrangement around the robot workspace.",
            "wrist_camera": "Reuse the unmodified /meshes/realsense_mid_stand and /meshes/d435 meshes from AgileX piper_isaac_sim's pinned piper_description_v100_realsense_camera_v2_base.usd. scripts/prepare_wrist_asset.py copies the exact mesh arrays to assets/generated/official_wrist/assembly.usda for Isaac 4.5 compatibility, omitting unsupported Chinese material names and rebinding local materials. Use its official URDF bracket/camera mounting transforms and Intel D435 nominal color extrinsics (+0.015 m camera-frame Y), with the old-to-current wrist mapping Rz(+90 degrees), translation (0,0,-0.004 m). The D435 geometry and bracket are upstream meshes, not a custom adapter. RGB intrinsics use D435 nominal 69x42-degree FOV with a centered square-pixel 640x480 crop (approximately 54.2x42 degrees, fx=fy approximately 625.22 px). RGB/depth use ideal rectified simulation cameras; depth is RTX geometric ground truth, not hardware stereo matching. Previously downloaded NVIDIA D455 geometry and optics are historical and no longer required.",
        },
        "files": [{k: v for k, v in record.items() if k != "verification"} for record in results],
    }
    output = root / "assets/sources.lock.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    downloaded = sum(record["verification"] == "downloaded" for record in results)
    print(f"Verified {len(results)} files ({downloaded} downloaded), {sum(record['bytes'] for record in results):,} bytes")
    print(f"Source lock: {output}")


if __name__ == "__main__":
    main()
