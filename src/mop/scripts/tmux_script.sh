#!/bin/bash

# Session Name
session="tfmencdec_nmg_vae_2021_12_27_"$1

which tmux


# Start New Session with our name
tmux new-session -d -s $session
ps -ef | grep tmux
# tmux send-keys -t "$session" "ls -lR /tmp" C-m;
tmux send-keys -t "$session" "conda deactivate" C-m;
tmux send-keys -t "$session" "conda deactivate" C-m;
tmux send-keys -t "$session" "conda activate nmg" C-m;
tmux send-keys -t "$session" "python /is/ps3/nsaini/projects/neural-motion-graph/src/nmg/scripts/nmg_trainer.py $1; tmux wait-for -S $session" C-m;
tmux wait-for $session
tmux kill-window -t $session

