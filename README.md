# E(n) Equivariant Graph Neural Networks

Using the official implementation (Pytorch 1.7.1) of:  

**E(n) Equivariant Graph Neural Networks**  
Victor Garcia Satorras, Emiel Hogeboom, Max Welling  
https://arxiv.org/abs/2102.09844

<img src="models/egnn.png" width="400">

### Project:
Applying the EGNN model to the standard AIDS graphs benchmark dataset. The project works without node positions by setting all of them to 0. By doing so the Equivariant Graph Convolutional Layer works like a standard Graph Convolutional Layer. <br/>
To run the project just type: 

```
cd models/egnn_clean
python3 main.py
```


