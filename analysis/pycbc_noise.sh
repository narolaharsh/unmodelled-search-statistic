outdir='pycbc_noise'
label='t1'

python ../src/generate_frames.py --outdir ${outdir} --label ${label} \
--detector-network ETT \
--signal-catalog ./catalog/t1_catalog.json \
--minimum-frequency 20 \
--seed 12 \
--n-glitches 5  \
--n-signals 1 \
--frame-duration 128 \
--sampling-frequency 4096


python ../src/reconstruct.py --frame-directory ${outdir} --detector-network ETT --outdir ${outdir} --label ${label}


python ../src/plot.py --outdir ${outdir} --label ${label}
