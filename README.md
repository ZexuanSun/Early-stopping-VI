	
Early stopping VI
==============================




## Overview
As opaque black-box predictive models become more prevalent, the need to develop interpretations for these models is of great interest. The concept of *variable importance* and *Shapley values* are interpretability measures that applies to any predictive model and assesses how much a variable or set of variables improves prediction performance. When the number of variables is large, estimating variable importance presents a significant computational challenge because re-training neural networks or other black-box algorithms requires significant additional computation. In this paper, we address this challenge for algorithms using gradient descent and gradient boosting (e.g.  neural networks, gradient-boosted decision trees). By using the ideas of early stopping of gradient-based methods in combination with warm-start using the *dropout* method, we develop a scalable method to estimate variable importance for any algorithm that can be expressed as an *iterative kernel update equation*. Importantly, we provide theoretical guarantees by using the  theory for early stopping of kernel-based methods for  neural networks with sufficiently large (but not necessarily infinite) width and gradient-boosting decision trees that use symmetric trees as a weaker learner. We also demonstrate the efficacy of our methods through simulations and a real data example which illustrates the computational benefit of early stopping rather than fully re-training the model as well as the increased accuracy of our approach.




## Code
This repo contains all the model implementation and experiments in the paper. 
A complete example can be found in `EarlyStopping_Demo.ipynb`. 

## License
```
MIT License

Copyright (c) 2024 Zexuan Sun

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
