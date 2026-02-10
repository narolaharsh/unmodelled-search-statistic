#Bash script to generate frames files for ET and analysing them using DeepExtractor
outdir="trial2"
label="model1024"
python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 1 \
--frame-duration 512 \
--n-signals 10 \
--n-glitches 40 \
--signal-catalog ./catalog/t1_catalog.json \
--padding 10  --seed 16
python reconstruct.py --outdir ${outdir} \
--label ${label} --data ${outdir}/${label}_frames.npz --delta-t 0.5
python plot.py --outdir ${outdir} --label ${label} --snr-data ${outdir}/dex_snr.npz