This program obtains 1D arrays as its numerical input and combines them with categorical input which is nucleotide bases of RNA. Then, 1D-CNN accepts mixed input to produce a single numerical outcome, which is conservation score for each nucleotide of interest. One example of these type of models are shown below. As it looks like a graph with 2 branches, one brach (longer) is associated with numerical input and the other brach (little) is designed for accepting categorical variables.


![model_1dcnn_680_concat_5_conv_plus_cat_3_new_3_5 10_hidden_4 (1)](https://github.com/gbulbul/1d-CNN-concatenated-model/assets/79763247/b3d0c57e-04da-4b19-a33f-f148099542d1)


![A_G_76 (3)](https://github.com/gbulbul/1d-CNN-concatenated-model/assets/79763247/a2b26f71-47b8-4ee1-b8e8-272a80636645)
This figure shows what we mean by "local context of a specific nucleotide" in 3D. It contains nucleotide itself with neighboring nts where each nt is made up of multiple atoms.
![image002 (1)](https://github.com/gbulbul/1d-CNN-concatenated-model/assets/79763247/64ffa72d-5a56-4357-aa35-f0e6fed5034d)
This figure shows what a machine could understand/see when it has the local context of a nucleotide. Now, in this format, we have 15X15 (2D) array which includes cells and numbers indicating the number of atoms falling into each cell.
Since we're in 3D, as you may guess that a unique 15X15 (2D) array would be sufficient to represent full information. Instead, we ended up with three 2D arrays which are shown below. Each 2D array refers a layer: bottom, base, top where a base layer is where the nucleotide of interest itself belongs.
![grid_675_base_A_layer_-1 (1)](https://github.com/gbulbul/1d-CNN-concatenated-model/assets/79763247/30a5e5ed-5395-4198-89f9-236c6db09769)
![grid_675_base_A_layer_0 (1)](https://github.com/gbulbul/1d-CNN-concatenated-model/assets/79763247/041168ad-8c52-48e3-ba34-44457caa76b6)
![grid_675_base_A_layer_1 (1)](https://github.com/gbulbul/1d-CNN-concatenated-model/assets/79763247/c98c8b56-feec-4e28-9d52-5902e6a3ec12)


