#Bash script to generate frames files for ET and analysing them using DeepExtractor
outdir="trial20hz"
label="model1024"
python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 1 \
--frame-duration 120 \
--n-signals 1 \
--n-glitches 10 \
--minimum-frequency 20 \
--signal-catalog ./catalog/t1_catalog.json \
--padding 10  --seed 10
python reconstruct.py --outdir ${outdir} \
--label ${label} --data ${outdir}/${label}_frames.npz --delta-t 2
python plot.py --outdir ${outdir} --label ${label} --snr-data ${outdir}/dex_snr.npz