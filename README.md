# Vae_anomaly_detection

In this repository there is all the code necessary to run a complete analysis, the **scripts** are so divided:

**all the background and signals root file are supposed to be in the **"/root_data"** folder**

1) **010_root_to_numpy.py**: run this script to convert from root to numpy data your background/signal .root data.
   -) The root data are supposed to be in the **"/root_data"** folder
   -) before running the script, remember to set the variable "data_kind" in the script to 'background' or 'signal'
   -) If background is selected, all the ntuples are processed (with cutflow designed in utils.py) and the results are splitted in may files to save 
      computational resources. The results are stored in the **"/splitted_numpy_bkg"** folder
   -) If signal is selected, all the final ntuples are processed and stored in **"/numpy_data/signal"**. No need to merge 
      the results because their size is smaller than background ntuples
      
2) **011_merge_numpy.py**: Run this script to merge all the splitted background files. Final results stored in 
   **"/numpy_data/backgrdound"**
   
3) **"012_train_val_test_splitting.py"**: Split the background in train-val-test splitting or train-val with custom proporctions. 
    Only need to modify the script the variable proportion. The final data are stored in **"/numpy_data/train_val_test"**
    
4) **"020_plot_distribution.ipynb"**: Jupyter notebook to visualize the background and signal distributions

5) **"030_vae_model_train.py"**: launch the train of vae on train-val data. You can modify here the model names (used to 
   the model) and all the others parameters affecting the model train
   
 6) **"031_vae_model_multiple_train.py"**: the same as 031_vae_model_train.py but repeating the train several times
 
 7) **"040_test_results.ipynb"**: Jupyter notebook to visualize the reconstruction loss on validation backgrodund and signals
 
 7) utils.py, config.py, vae_utility are auxiliary custom module containing path and utility functions used in the above scripts
 
