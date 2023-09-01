# SEG.A - MW - 2023
Contribution to the SEG.A Challenge (MICCAI 2023) by Marek Wodzinski.

The challenge website: [Link](https://multicenteraorta.grand-challenge.org/multicenteraorta/)

Here you can see the full source code used to train / test the proposed solution.

Only the final experiment is left (the one used for the final Docker submission).

* In order to reproduce the experiment you should:
    * Download the SEG.A dataset [Link](https://multicenteraorta.grand-challenge.org/data/)
    * Update the [hpc_paths.py](./src/paths/hpc_paths.py) and [paths.py](./src/paths/paths.py) files.
    * Run the [parse_sega.py](./src/parsers/parse_sega.py)
    * Run the [run_aug_sega.py](./src/parsers/run_aug_sega.py)
    * Run the training using [run_segmentation_trainer.py](./src/runners/run_segmentation_trainer.py)
    * And finally use the trained model for inference using [inference.py](./src/inference/inference_sega.py)

The network was trained using HPC infrastructure (PLGRID). Therefore the .slurm scripts are omitted for clarity.

To run just the inference - the final model is available upon request. 
You can also apply for an access to the algorithm directly using the Grand-Challenge platform: [Link](https://grand-challenge.org/algorithms/sega_mw/)

* If you found the source code useful, please cite:
 * 1) The SEGA Challenge Paper (TODO) 
 * 2) The paper presenting the proposed solution (TODO)

Please find the method description: [Description](TODO).


The description will be extended soon.
