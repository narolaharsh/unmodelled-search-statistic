outdir='search'
label='trial1'
frames_location=./frames

python ../src/reconstruct.py --frame-directory ${frames_location} --detector-network ETT --outdir ${outdir} --label ${label}
python ../src/plot.py --outdir ${outdir} --label ${label}
