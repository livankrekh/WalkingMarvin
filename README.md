# Walking Marvin (bipedal walker problem)
Remember the sad Marvin from "Hitchhiker's guide to the galaxy"? In this project we train him to walk!

![gif_sad_marvin](https://media.giphy.com/media/SFkjp1R8iRIWc/giphy.gif)

##  Installation guide
To install Marvin environment and dependencies, please, run following shell script:
```shell
> ./install.sh
```
It will install `python3.7` with all dependencies inside the virtualenv named `venv`.

To run Marvin, please, use the following commands:
```shell
> ./marvin.py <-r>
```
or
```shell
> source venv/bin/activate
> python3 marvin.py <-r>
```
Flag `-r` is used for running trained Marvin (by default pretrained model is used)

## Results
Basically, Marvin starts from the folowing state:

![gif_marvin1](gifs/marvin1.gif)
![gif_marvin1](gifs/marvin2.gif)
![gif_marvin1](gifs/marvin3.gif)
