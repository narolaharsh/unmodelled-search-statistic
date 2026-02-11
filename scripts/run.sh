#Bash script to generate frames files for ET and analysing them using DeepExtractor
outdir="trial20hz"



detnet='ET2L'
label="model1024_network_${detnet}"
python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 1 \
--frame-duration 600 \
--n-signals 10 \
--n-glitches 10 \
--minimum-frequency 20 \
--signal-catalog ./catalog/t1_catalog.json \
--padding 10  --seed 22 --detector-network ${detnet}
python reconstruct.py --outdir ${outdir} \
--label ${label} --frames ${outdir}/${label}_frames.npz --delta-t 0.5
python plot.py --outdir ${outdir} --label ${label} --snr-data ${outdir}/${label}_dex_snr.npz


detnet='ETT'
label="model1024_network_${detnet}"

python generate_frames.py --outdir ${outdir} \
--label ${label} --inject-signals 1 \
--inject-glitches 1 \
--frame-duration 600 \
--n-signals 10 \
--n-glitches 10 \
--minimum-frequency 20 \
--signal-catalog ./catalog/t1_catalog.json \
--padding 10  --seed 22 --detector-network ${detnet}
python reconstruct.py --outdir ${outdir} \
--label ${label} --frames ${outdir}/${label}_frames.npz --delta-t 0.5
python plot.py --outdir ${outdir} --label ${label} --snr-data ${outdir}/${label}_dex_snr.npz
