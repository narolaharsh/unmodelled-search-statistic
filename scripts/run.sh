#Bash script to generate frames files for ET and analysing them using DeepExtractor
outdir="trial2"
label="model1024"
python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 1 \
--frame-duration 256 \
--n-signals 4 \
--n-glitches 20 \
--signal-catalog ./catalog/t1_catalog.json \
--padding 10  --seed 16
python reconstruct.py --outdir ${outdir} \
--label ${label} --data ${outdir}/${label}_frames.npz --delta-t 0.5