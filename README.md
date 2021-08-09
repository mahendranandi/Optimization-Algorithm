# Optimization-Algorithm
For learning, visualizing and understanding the optimization techniques and algorithms.
# ____________________________________________________________________________________________________________


Visualizations and in depth concepts of the Machine Learning optimization algorithms are discussed and shown here with different functions as examples and understanding the differences by comparing  them.
## Over all types of Line and direction search algorithms in Gradient Descent discussed here:
- Line search & direction search in Gradient Descent:
  - >> Exact Methods:
    - > Using derivatives ( for differentiable functions only):
      - Newton's method
      - Secant Method
    - > Without using derivatives (require function evaluation only):
      - Golden Section
      - Fibonacci
      - Bisection
  - >> Inexact method  (step length found is not exactly optimal )  {these are parameter specific}:
      - Armijo
      - Wolf-Powell



<!-- * Line search & direction search in Gradient Descent:
+     Exact Methods:
*         Using derivatives ( for differentiable functions only):
*             Newton's method
*             Secant Method
*         Without using derivatives (require function evaluation only):
*             Golden Section
*             Fibonacci
*             Bisection
*     Inexact method  (step length found is not exactly optimal )  {these are parameter specific}:
*         Armijo
*         Wolf-Powell -->

Here All optimization techniques are explained through code. You can find the codes inside the ipynb files along with some explanation. The codes are very simple to understand, if the theory is clear to you. ( In future I will put some easy explanation and theory here )
First go through the file " complete_OPTAL.ipynb" then go through "GD_SGD_MNGD_momentum.ipynb" initially with the pdf. If you want to visualize the pots in other way checkout the "collection_of_plots.ipynb"


### Example of Some Bivariate Functions: :+1:
Here we go,
First some visualizations of functions will definitely make you curious to know more about the optimization, so, 
look at the functions think how to find the minimum starting from a arbitrary point,

$\frac{sin(10(x^2+y^2))}{10}$
![Image of function](Images/cool.png)

We will work with simple univariate and bivariate functions for understanding, One convex and one non-convex function is shown below,
![Image of function](Images/convex_function.png)      ![Image of function](Images/non_convex.png)

You can visualize some nice simple functions and other complex functions like Rosenbrock function in the ipynb.

### GD in Univariate function:

Next I will show some plots by which you will get some interest about Gradient descent,
Now see how zigzagging happens while reaching the minima starting from an arbitrary point, in the left side the value of the parameter is shown with iteration and in the right side function values are shown
![Image of function](Images/download8.png)

Here you can see how step size gradually decrease near optima, and the ideal approximation of step length is shown with iteration using Armijo rule,
![Image of function](Images/download.png)


To reduce this long zigzag path we need to adjust parameters or need to apply various methods, after using these, as you can see below the zigzagging reduced ..again ignore all other terms just follow the curves, later you can understand why and how this hes been happened.
![Image of function](Images/download12.png)


### GD in Bivariate function:

Now the same thing for a bivariate function is shown in a contour plot, latter you can visualize it in a 3d plot also:
![Image of function](Images/download3.png)


### SGD in Bivariate function ( perturbed GD):

Now come to SGD[ stochastic gradient descent]
As we are not dealing with dataset directly, we here take simulation of that by adding random noise from normal distribution, and then follow how the difficulty is getting increased by adding this noise, and this problem looks like a **Stochastic Gradient Descent** problem, latter we will reduce this noise by taking average of some random noise to simulate the **Mini BAtch Gradient Descent** ( OK, so you have to be good with random numbers in python, Here I attached one plot to get you idea how you can play with it to see differences) Follow the sampling, i.e, sample number and its distribution, When the sample variance is large this is a simulation of SGD and when the variance is low this is same as a  Mini Batch GD.
![Image of function](Images/random.png)


## Dynamic step size:
As we have added noise to perform a SGD, it becomes tough now to reach the minima within affordable iteration number. So, for that we have to control the step size wisely(in the plot below, the step size is polynomially controlled for other methods go to the first portion of the "GD_SGD_MBGD_momentum.ipynb"). In the figure below, The change of the parameters, function value, function value near minima, gradient norm value step size is shown with iteration. Also you can visualize how the minima is reached in a contour plot as well as in a 3d plot.

- ![Image of function](Images/pr_gd.png) - ![Image of function](Images/pr_gd_3d.png)

## Comparison of different methods of dynamic step size:
Here polynomially decreases step size is used but you can use exponential functions to handle eta, or you may keep it constant or step wise decreasing, a plot showing comparison among these different methods are shown here,,, these different methods behave differently in different function, so be careful. But in most cases Polynomially decreasing $\eta$ is doing better control
![Image of function](Images/compare.png)

## For the non-convex Surface:
Finally we will see how tough this is for a non-convex surface ( the function used here is shown above $\large f(x_1, x_2) =  x_1^2 - 2 x_2^2$ ),

![Image of function](Images/pr_gd_cncv.png)
![Image of function](Images/pr_gd_cncv_3d.png)


## Momentum updation in GD:
And finally We will use momentum updation to handle these difficulties. Momentum help the point keep going in a direction resultant of its momentum and gradient and helps not to stuck in local minima or saddle point. Here to show the change I purposefully kept the momentum parameter high, so you can see that though it know in which direction the minima is , but it will take its momentum in consideration. As a result it takes a long way, but if you reduce momentum controlling parameter it will help,

![Image of function](Images/momentum.png)
![Image of function](Images/momentum_3d.png)
Thats it. Go to the ipynb files now.

**
Also I think, it will be better if anyone want to help me by just making the ipynb files more understandable by separation the topics.
If you feel hard anywhere, contact me in mahendranandi.0608@gmail.com
**
