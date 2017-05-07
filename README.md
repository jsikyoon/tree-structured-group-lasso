Tree Structured Group LASSO
===========

Python Implementation of Tree Structured Group LASSO.

http://www.jmlr.org/papers/volume12/jenatton11a/jenatton11a.pdf

http://icml2010.haifa.il.ibm.com/papers/416.pdf

"Sparse coding consists in representing signals as sparse linear combinations of atoms selected from a dictionary. We consider an extension of this framework where the atoms are further assumed to be embedded in a tree. This is achieved using a recently introduced tree-structured sparse regularization norm, which has proven useful in several applications." from the JMLR paper

![alt tag](https://github.com/jaesik817/hsc/blob/master/figures/tsgl.PNG)

MNIST Dataset Representation
-----------

MNIST Dataset Representation with Tree Structured Group LASSO. 

### Setting
The number of atoms is 30, in which, 0-9 is root group, 10-19 and 20-29 are reaf groups. Each reaf group has hierarchical relationship with root group.

You can run the code as followed command.

`
python mnist_tsgl.py
`

### Results
You can check Tree Structured relationship in printed logs.
(If parent is zeros, then every children are also zeros. If one of children is zeros, then every nodes in upper groups are also zeros.)

![alt tag](https://github.com/jaesik817/hsc/blob/master/figures/tsgl_result.PNG)
