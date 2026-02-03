### Inputs needed 

1. Time domain strain of the supernovae signals
    a. LVK CCSN search from O1 and O2: https://arxiv.org/pdf/1908.03584
    b. Supernovae waveforms: The paper above and the following papers http://arxiv.org/abs/1106.6301 (Müller et al) http://arxiv.org/abs/1210.6674 (Ott et al), http://arxiv.org/abs/1505.05824 (Yakunin et al)
    c. > "During the prompt convection, in the initial stages post bounce, GWs are emitted in the frequency range from 100-300 Hz, while at later times, GWs up to around 2 kHz can be expected [105, 106]"
2. Verify if you can inject the SN signals in the noise the same way we inject CBC. 
3. A U-net model (DeepExtractor) trained in the high-frequency range (relevant for SN) with ET-D PSD. 
4. Do you want to fold in PSD variation statistic?


### Outputs 

1. One day of ET data which contains SN signals and glitches.
2. Ranking statistic for the SN signals and glitches.


[![PDF](https://img.shields.io/badge/PDF-Latest-blue)](https://nightly.link/github.com/narolaharsh/unmodelled-search-statistic/workflows/paper/main/unmodelled_search.zip)