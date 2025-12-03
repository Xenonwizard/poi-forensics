import os
import subprocess
from pathlib import Path
import traceback

# ===================================================
# CONFIG
# ===================================================
VIDEOS_ROOT = Path("./videos")
OUTPUT_ROOT = Path("./newoutput0312")
GPU = "0"

# Folders to skip entirely
SKIP_FOLDERS = {"trump_side_front", "train_trump", "stashed_trump_videos"}

ERROR_LOG = Path("./error_log.txt")


# ===================================================
# HELPER FUNCTIONS
# ===================================================
def detect_poi(folder_name: str) -> str:
    """Return the correct POI directory based on folder name."""
    name = folder_name.lower()

    if "cage" in name or "nic" in name:
        return "./pois/nicolas-cage/app_poiforensics"

    if "trump" in name:
        return "./pois/trump-side-front/app_poiforensics"

    if "michelle" in name or "yeoh" in name:
        return "./pois/michelle-yeoh/app_poiforensics"

    # Default fallback
    return "./pois/nicolas-cage/app_poiforensics"


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def append_error_log(text: str):
    """Append text to error_log.txt"""
    with ERROR_LOG.open("a") as f:
        f.write(text + "\n")


def run_command(cmd, env):
    """Run a shell command with error skipping."""
    try:
        subprocess.run(cmd, shell=True, check=True, env=env)
        return True
    except subprocess.CalledProcessError:
        return False
    except Exception:
        return False


# ===================================================
# MAIN LOGIC
# ===================================================
def main():
    print("=== POI Forensics Batch Runner (Smart POI + Error Skip) ===")

    base_env = os.environ.copy()

    for root, dirs, files in os.walk(VIDEOS_ROOT):
        root_path = Path(root)

        # Folder relative to videos/
        try:
            rel_folder = root_path.relative_to(VIDEOS_ROOT)
        except ValueError:
            continue

        # Skip unwanted folders
        if rel_folder.parts and rel_folder.parts[0] in SKIP_FOLDERS:
            print(f"Skipping folder: {rel_folder}")
            continue

        # Detect target POI based on folder name
        first_folder = rel_folder.parts[0] if rel_folder.parts else ""
        poi_dir = detect_poi(first_folder)

        for file in files:
            # Accept only known video formats
            if not file.lower().endswith((".mp4", ".mov", ".mkv", ".avi")):
                continue

            video_path = root_path / file
            video_stem = video_path.stem

            # Output directory
            out_dir = OUTPUT_ROOT / rel_folder
            ensure_dir(out_dir)

            output_npz = out_dir / f"{video_stem}.npz"

            print(f"\n>>> Processing Video: {video_path}")
            print(f"    POI Model     : {poi_dir}")
            print(f"    Output Target : {output_npz}")

            # Prepare environment variables
            env = base_env.copy()
            env["INPUT_VIDEO"] = str(video_path)
            env["OUPUT_NPZ"]   = str(output_npz)  # yes spelled OUPUT on purpose
            env["PYTHONPATH"]  = env.get("PYTHONPATH", "") + ":" + str(Path("./pythonlib").resolve())

            # Build command
            cmd = (
                f'python main_test.py '
                f'--file_video_input "{env["INPUT_VIDEO"]}" '
                f'--file_output "{env["OUPUT_NPZ"]}" '
                f'--dir_poi "{poi_dir}" '
                f'--gpu {GPU} '
                f'--create_plot 1 '
                f'--create_videoout 1 '
                f'--dist_normalization 1 '
            )

            # RUN + ERROR-SAFE
            try:
                success = run_command(cmd, env)
                if not success:
                    raise RuntimeError("main_test.py execution failed")

            except Exception as e:
                print(f"❌ ERROR — Skipping video: {video_path}")

                # Write error to log
                error_entry = (
                    f"ERROR processing video:\n"
                    f"Video Path: {video_path}\n"
                    f"POI Used  : {poi_dir}\n"
                    f"Command   : {cmd}\n"
                    f"Exception : {e}\n"
                    f"Traceback:\n{traceback.format_exc()}\n"
                    f"{'-'*60}\n"
                )
                append_error_log(error_entry)

                continue  # skip to next video

    print("\n=== ALL VIDEOS PROCESSED (Errors logged in error_log.txt) ===")


if __name__ == "__main__":
    main()
