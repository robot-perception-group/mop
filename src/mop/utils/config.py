import os

SMPL_DATA_PATH = "data/smpl/"

SMPL_KINTREE_PATH = os.path.join(SMPL_DATA_PATH, "kintree_table.pkl")

dataset_dir = "/home/nsaini/Datasets"
nmg_repo_path = "/is/ps3/nsaini/projects/neural-motion-graph"

if os.uname()[1].lower() == "ps104" or os.uname()[1].lower() == "ps106":
    logdir = "/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs"
    dataset_dir = "/home/nsaini/Datasets"
    nmg_repo_path = "/is/ps3/nsaini/projects/neural-motion-graph"
elif os.path.exists("/is/ps3/nsaini"):
    dataset_dir = "/home/nsaini/Datasets"
    nmg_repo_path = "/is/ps3/nsaini/projects/neural-motion-graph"
    logdir = "/is/ps3/nsaini/projects/neural-motion-graph/nmg_logs"
else:
    dataset_dir = "/mnt/session_space/home/jsaito/Datasets"
    nmg_repo_path = "/mnt/session_space/home/jsaito/projects/neural-motion-graph"
    logdir = "/mnt/session_space/home/jsaito/projects/neural-motion-graph/nmg_logs"