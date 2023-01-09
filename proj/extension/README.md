# README

## Requirements

pytorch with cuda, for installation, this can be installed with conda in the following way:

```
conda install pytorch pytorch-cuda=11.7 -c pytorch -c nvidia
``` 
For more installation instructions see https://pytorch.org/get-started/locally/

## Installation
In this folder, run python setup.py install. This will compile the cuda kernel and make it accessible within pytorch. 

## Checking validity

To check that the cuda kernel returns the correct output, run:

```
python test.py --dtype d 16 32 32 check
```

This checks the tree implementations for a batch size of 16, and two factors (32 and 32).
It performs a forward pass and a bakcwards pass, and checks that they align between versions.
(Note that backwards pass for the cuda kernel is not yet implemented)
It should output something like the following:
```
||v1_y|| = 124.41339422306116
||v1_g|| = 125.37834018338314
||v2_y|| = 124.41339422306116
||v2_g|| = 125.37834018338314
||v3_y|| = 124.41339422306116
||v3_g|| = 127.99595817207963
v1_y == v2_y: True (0.0e+00)
v1_g == v2_g: True (0.0e+00)
v1_y == v3_y: True (0.0e+00)
v1_g == v3_g: False (1.8e+02)
v2_y == v3_y: True (0.0e+00)
v2_g == v3_g: False (1.8e+02)
```

v1 corresponds to the naive version, v2 corresponds to the pytorch native improved version, and v3 corresponds to the cuda version. 

## Profiling
To profile a specific implemention, run the following: 


```
python.py test.py --dtype f 16 32 32 profile --implementation v1 #or v2, or v3
```

This profiles the specified implementation, prints out basic profiling information, and outputs a trace that can be viewed in chrome://tracing.

## Benchmarking

To benchmark a specific factorization, run:

```
python.py test.py --dtype f 16 32 32 bench
```

This will output a summary of all three implementations on the given factorization: 

```
python test.py --dtype f 32 128 128  bench
state size : 16384
AI         : 1.2e+02
[--------------------- benchmark ----------------------]
                                  |   v1  |   v2  |   v3
1 threads: ---------------------------------------------
      32 x 128 x 128 (1.342e+08)  |  241  |  220  |  150

Times are in microseconds (us).
```

