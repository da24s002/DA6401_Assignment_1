This is a modularized implementation of a feed forward neural network.

In order to run the model you can run the following command :

## runs with the best arguments found while experimenting with the FashionMNIST dataset.
python train.py 

arguments supported :

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `-wp`, `--wandb_project` | myprojectname | Project name used to track experiments in Weights & Biases dashboard |
| `-we`, `--wandb_entity` | myname  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `-d`, `--dataset` | fashion_mnist | choices:  ["mnist", "fashion_mnist"] |
| `-e`, `--epochs` | 1 |  Number of epochs to train neural network.|
| `-b`, `--batch_size` | 4 | Batch size used to train neural network. | 
| `-l`, `--loss` | cross_entropy | choices:  ["mean_squared_error", "cross_entropy"] |
| `-o`, `--optimizer` | sgd | choices:  ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"] | 
| `-lr`, `--learning_rate` | 0.1 | Learning rate used to optimize model parameters | 
| `-m`, `--momentum` | 0.5 | Momentum used by momentum and nag optimizers. |
| `-beta`, `--beta` | 0.5 | Beta used by rmsprop optimizer | 
| `-beta1`, `--beta1` | 0.5 | Beta1 used by adam and nadam optimizers. | 
| `-beta2`, `--beta2` | 0.5 | Beta2 used by adam and nadam optimizers. |
| `-eps`, `--epsilon` | 0.000001 | Epsilon used by optimizers. |
| `-w_d`, `--weight_decay` | .0 | Weight decay used by optimizers. |
| `-w_i`, `--weight_init` | random | choices:  ["random", "Xavier"] | 
| `-nhl`, `--num_layers` | 1 | Number of hidden layers used in feedforward neural network. | 
| `-sz`, `--hidden_size` | 4 | Number of hidden neurons in a feedforward layer. |
| `-a`, `--activation` | sigmoid | choices:  ["identity", "sigmoid", "tanh", "ReLU"] |

The final test accuracy is reported in console after all the epochs are run.

Please go through the wandb init workflow in your current directory before running the code, as the code is configured to automatically log the run into wandb.

Run the command:
## wandb init

<br>
then provide the options asked, (note that the default project name is provided in the key value pair : "wandb_project": "DA6401_Assignment_1", you can ## change it while providing command line argument or directly in the dictionary, if you use a different wandb project while initialization)

==========================================================================================

Running a wandb sweep,

eg:

## wandb sweep config.yaml
<br>
the link of the sweep created along with the id will be provided after you run the above command, just add the argument --count after the command provided. following is an example of such a command
<br>
## wandb agent da24s002-indian-institute-of-technology-madras/DA6401_Assignment_1/edmy42po --count 75
<br>
run the above to start a wandb sweep.
by default the sweep uses the list of hyperparams written in config.yaml


========================================================================================
program: "train.py"<br>
name: "DA6401_Assignment1_sweep"<br>
method: "bayes"  --options ['bayes', 'random', 'grid'], in the final version, we have used bayes search, as it gave best validation accuracy<br>
metric:<br>
  goal: maximize<br>
  name: validation_accuracy<br>
parameters:<br>
  epochs:<br>
    values: [5, 10]<br>
  batch_size:<br>
    values: [16, 32, 64]<br>
  num_layers:<br>
    values: [3,4,5]<br>
  hidden_size:<br>
    values: [32, 64, 128]<br>
  weight_decay:<br>
    max: 0.00001<br>
    min: 0.000001<br>
    distribution: log_uniform_values<br>
  learning_rate:<br>
    max: 0.001<br>
    min: 0.0001<br>
    distribution: log_uniform_values<br>
  optimizer:<br>
    values: ["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"]<br>
  weight_init:<br>
    values: ["random", "xavier"]<br>
  activation:<br>
    values: ["sigmoid", "tanh", "relu"]<br>
  momentum:<br>
    max: 0.999<br>
    min: 0.1<br>
    distribution: log_uniform_values<br>
  beta:<br>
    max: 0.999<br>
    min: 0.1<br>
    distribution: log_uniform_values<br>
  beta1:<br>
    max: 0.999<br>
    min: 0.1<br>
    distribution: log_uniform_values<br>
  beta2:<br>
    max: 0.999<br>
    min: 0.1<br>
    distribution: log_uniform_values<br>
  
  
  

