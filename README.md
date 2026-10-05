# Two-patch foraging identification

This repository contains the code for the article:
``Socio-cognitive models in a patch foraging setting: a case study for model selection and parameter identifiability methods''
Lisa Blum Moyse, Ahmed El Hady

python files (.py)

# Repository structure
## functions folder:
To simulate the dynamics:
* simu_fc.py generates output: accuracy distributions
* simu_fc_P.py generates output: departure time distribution
 
 
To define the functions used for model selection and parameter identifiability:
* Bay_fc.py for Fig 6AB, Fig 7, Fig 8ABC
* Bay_fc_P.py for Fig 6CD
* Bay_fc_sampling.py for Fig 8D
* Bay_fc_env.py for Fig 9
* Bay_fc_nlfc.py for SFig 4

## model_characterization folder: first part of the Results
Scripts for each figure of this part of the article (i.e. Fig3A.py = script for the Fig 3A)
* Fig2.py
* Fig3A.py
* Fig3B.py
* Fig3C.py
* Fig4A.py
* Fig4B.py
* Fig4C.py
* Fig5.py

Supplementary figures (SFig2 is generated inside the codes Fig3A.py - Fig4C.py)
* SFig1.py
* SFig3.py

## model_selection_identifiability folder: first part of the Results
To generate the datasets and the likelihoods:
* generate_likelihoods.py for Fig 6AB, Fig 7, Fig 8ABC
* generate_likelihoods_P.py for Fig 6CD
* generate_likelihoods_sample.py for Fig 8D
* generate_likelihoods_env.py for Fig 9
* generate_likelihoods_nlfc.py for SFig 4

To assess the model selection and parameter identifiability through the generation of confusion matrices:
* assess_id.py for Fig 6AB, Fig 7, Fig 8ABC
* assess_id_P.py for Fig 6CD
* assess_id_sample.py for Fig 8D
* assess_id_env.py for Fig 9
* assess_id_nlfc.py for SFig 4

To plot the figures of this part of the article (i.e. Fig8D.py = script for the Fig 8D)
* Fig6.py
* Fig7_8ABC.py
* Fig8D.py
* Fig9.py
* SFig4.py

# Citation
If you use this code in your own research, please cite:
Lisa Blum Moyse and Ahmed El Hady. Twopatchforagingidentification. 
Github. url: https://github.com/lblummoyse/twopatchforagingidentification.

# Contact
For questions about this repository, contact: Lisa Blum Moyse at lisa.blum-moyse@ens-lyon.fr






