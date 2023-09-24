# Rationale

- This is an improvised neuroevolution algorithm. 
- The point is to prove that you don't need anyone else's research.
- You can make your own neuroevolution algorithm and it will achieve results similar to those of state-of-the-art algorithms. 


# How to write a neuroevolution algorithm?

This repo contains an experimental genetic algorithm which works on neural networks. 
The neural network is implemented using `List[int]` representing nodes and `Dict[Tuple[int, int], float]` to represent the edges.
genetic.py and graph.py only uses standard python.
The mutation steps can add new nodes, remove edges, change weights and choose new activation functions. This is pretty much all you need to implement neuroevolution.

This repo does NOT contain an implementation of NEAT (NeuroEvolution of Augmented Topologies).


# RUNNING TESTS

```
python -m unittest discover
```

