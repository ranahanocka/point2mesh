python main.py --input-pc ./data/noisy_ankylosaurus.ply \
--initial-mesh ./data/noisy_ankylosaurus_initmesh.obj \
--save-path ./checkpoints/noisy_ankylosaurus2_long \
--samples 60000 \
--begin-samples 40000 \
--max-faces 15000 \
--pools 0.1 0.2 0.3 0.4 \
--iterations 6000