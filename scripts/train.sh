# Training on our synthetic noisy dataset
# Note this would result in 0.1-0.3 (dB) lower performance than our released pretrained models
# as these models are trained offline with noise generated once, 
# in contrast to our online models with noise generated on-the-fly
python train_syn.py --name sid-ours-sonya7s2 --stage_in raw --stage_out raw --include 4
python train_syn.py --name sid-ours-nikond850 --stage_in raw --stage_out raw --include 3
python train_syn.py --name sid-ours-canoneos700d --stage_in raw --stage_out raw --include 2
python train_syn.py --name sid-ours-canoneos70d --stage_in raw --stage_out raw --include 1

# Training with paired real data
# python train_real.py --name sid-paired-new --stage_in raw --stage_out raw
# Raw to sRGB pipeline
python train_real.py --name sid-paired-raw2rgb --stage_in raw --stage_out srgb
# Raw to sRGB pipeline with camera response function (CRF calibrated on SonyA7S2)
python train_real.py --name sid-paired-raw2rgb --stage_in raw --stage_out srgb --crf
