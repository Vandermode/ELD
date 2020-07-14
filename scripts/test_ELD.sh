# sid-paired: the model trained with paired real data from SID training dataset
python test_ELD.py --name sid-paired -r -re 200 --no-verbose --chop
# sid-ours-*: the model trained only with the synthetic data from our noise model
# inc1-4 denote the noise parameters are calibrated using the CanonEOS70D, CanonEOS700D, NikonD850, SonyA7S2 respectively. 
python test_ELD.py --name sid-ours-inc4 -r -re 200 --no-verbose --include 4 --chop 
python test_ELD.py --name sid-ours-inc3 -r -re 200 --no-verbose --include 3 --chop 
python test_ELD.py --name sid-ours-inc1 -r -re 200 --no-verbose --include 1 --chop 
python test_ELD.py --name sid-ours-inc2 -r -re 200 --no-verbose --include 2 --chop 
