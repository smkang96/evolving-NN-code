# evolving-NN-code
Evolving neural network code based on existing human code. 

## Dependencies
Our implementation requires the following. 
* Python 2.7
* PyTorch 0.4.0

Errors have been reported when trying to run the code on Python 3 or on higher or lower versions of PyTorch. It also assumes there is a GPU in the system, which the author guesses is a dependency. 

## Execution 
The genetic algorithm is implemented in the ```GACode``` directory. 

### 1. Preparation
One must make the directories specified in ```code_path``` and ```img_path``` in the ```ga.py``` code. These are directories where the resulting architectures and Pareto front images will be saved, respectively. 

### 2. Execution
All the following regards the ```ga.py``` code. The baseline models to place in the initial population can be modified by changing the components in the ```INIT_POP``` variable. The amount of growth time can be specified using the ```growth_time``` variable. The population size can be specified using the ```pop_size``` variable. The number of generations can be specified by setting the number in the ```breed``` function to the desired number. 

After all parameters have been specified, execution is simple:
```
python ga.py
```

### 3. Result Collection
The best-performing networks' names and their measured performance is saved in the file name specified in the ```result_file``` variable. The code from each generation is saved in the directory specified in the ```code_path``` variable. A plot of best-performing networks can be collected from the ```img_path```. 

## Additional Information
This project was done as part of a term project in the AI Based Software Engineering course at KAIST in Autumn 2018.
