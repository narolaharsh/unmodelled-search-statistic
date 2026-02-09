#Bash script to generate frames files for ET and analysing them using DeepExtractor
outdir="trial1"
label="model1024"
python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 0 \
--frame-duration 32
python reconstruct.py --outdir ${outdir} \
--label ${label} --data ${outdir}/${label}_frames.npz