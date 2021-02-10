# LTO-CMA
Code for the paper "Learning Step-Size Adaptation in CMA-ES"
## License
Our work is available under Apache-2.0. In order to learn step-size adaptation in CMA-ES we use guided policy search (GPS).
We built upon the GPS version as given by Li and Malik. The original GPS code of Li and Malik can be found at https://www.math.ias.edu/~ke.li/downloads/lto_code.tar.gz

In a nutshell, we modified the GPS code to be able to continuously sample from the starting teacher. To this end we introduce a sampling rate that determines how often we use new samples generated from the starting policy.

The original code falls under GPLv3. In *source/gps* we list files that we modifed (thereby fall under GPLv3) and those that are of our creation (i.e. under Apache-2.0)

## Experiment Setup
### Training
- Create experiment folder
- Create file with hyperparameters of the experiment *hyperparams.py* in the experiment folder
- Start learning step-size adaptation by executing the command:
```
python gps_main.py EXPERIMENT_FOLDER_NAME
```
- The output of training is the pickled version of the learned policy, saved in the path *EXPERIMENT_FOLDER_NAME/data_files*.
### Testing
- Add the path to the learned policy in the hyperparameter file *hyperparams.py*
- Start testing the performance of the learned policy on the test set by executing the command:
```
python gps_test.py EXPERIMENT_FOLDER_NAME
```
- The output of testing are the files *test_data_X.json* for each condition index X of the test set, saved in the experiment folder.
- The output file *test_data_X.json* contains:
  - The average objective values from 25 samples of running the learned policy on the test condition X,
  - The end objective values of the 25 samples,
  - The average step-size for each step of the optimization trajectory from 25 samples, and 
  - The standard deviation of the objective value and the step-size for each step of the optimization trajectory.
- To plot the results, run the *plot_performance.py* script in the *scripts* folder.
## Reference
```
@inproceedings{shala-ppsn20,
  author    = {G.Shala and A. Biedenkapp and N.Awad and S. Adriaensen and M.Lindauer and F. Hutter},
  title     = {Learning Step-size Adaptation in CMA-ES},
  booktitle = {Proceedings of the Sixteenth International Conference on Parallel Problem Solving from Nature ({PPSN}'20)},
  year = {2020},
  month     = sep,
}
```
