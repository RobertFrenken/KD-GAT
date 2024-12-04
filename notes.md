Here is a list of notes for this project.

In this repo I would like to compile different datasets of CAN bus security data, 
and create graph representations of them. I will have a file that will do the bulk processing 
and can handle the slight differences between each file, and will turn said CAN frames
into a graph representation. I will have another file, likely a notebook that 
will visualize these graphs to help illistrate my idea, and maybe do some interesting
similariites across different datasets. I will have a model file that will store the GNN
that I would like to encode. For my first iteration I will do a simple comparision between:
- LSTM or time series
- Feature only like random forest
- GNN (with relevant features as inputs)

I would then like to make a compositie model that will essentially comvbine all of theses components
into a single model, where each model can derive interesting features. Similar to what Ryan suggested 
perhaps I can take these features and perform some sort of UMAP or similar to show in a latent space
how each of the models processes each of the datasets.

Also would like to make a teacher-student (knowledge distillation) model to see if that can
help put models on edge devices. Keep closing the gap between theory and implementation.