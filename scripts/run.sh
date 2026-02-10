#Bash script to generate frames files for ET and analysing them using DeepExtractor
outdir="trial20hz"
detnet='ETT'
label="model1024_network_${detnet}"

python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 0 \
--frame-duration 60 \
--n-signals 1 \
--n-glitches 3 \
--minimum-frequency 20 \
--signal-catalog ./catalog/t1_catalog.json \
--padding 10  --seed 2 --detector-network ${detnet}
python reconstruct.py --outdir ${outdir} \
--label ${label} --data ${outdir}/${label}_frames.npz --delta-t 0.5
python plot.py --outdir ${outdir} --label ${label} --snr-data ${outdir}/dex_snr.npz