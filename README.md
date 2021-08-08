# Optimization-Algorithm
For learning OPTAL

Visualization of the Machine Learning optimization algorithms are shown here with example of different functions and with comparison among them.


* Line search & direction search in Gradient Descent:
+     Exact Methods:
*         Using derivatives ( for differentiable fnctons only):
*             Newton's method
*             Secant Method
*         Without using derivatives (requre function evaluation only):
*             Golden Section
*             Fibonacci
*             Bisection
*     Inexat method  (step length found is not exactly optimal )  {these are parameter specific}:
*         Armijo
*         Wolf-Powell

Here All optimization techniques are explained through code. You can find the codes inside the ipynb files along with some explaination. The codes are very simple to understand, if the theory is clear to you. ( In future I will put some easy explaination and theory here )
First go through the file " complete_OPTAL.ipynb" then go through "GD_SGD_MNGD_momentum.ipynb" initially with the pdf. If you want to visualize the pots in other way checkout the "collection_of_plots.ipynb"

Here we go,
First some visualization of functions will definitely make you curious to know more about the optimization, so, 
look at the function,

$\frac{sin(10(x^2+y^2))}{10}$
![Image of function](Images/cool.png)

We will work with simple univariat and bivariate function for understanding, One convex and one non-convex function is shown below,
![Image of function](Images/convex_function.png)
![Image of function](Images/non_convex.png)
You can visualize some nice simple functions and other complex functions and Rosenbrock function in the ipynb.
Next I will show some plots by which you will get some interest about Gradient descent,
Here you can see how step size gradually decrease near optima, ignore Armijo method for now just follow the curve
![Image of function](Images/download.png)

Now see how zagziging happens while reaching the minima, in the left side the value of the parameter is shown and in the right side function values are showen
![Image of function](Images/download8.png)

To reduce this long zigzag path we need various shar maehtods, as you can see below the zigzaging reduced ..again ignore all other terms just follow the curves, later you can understand why and how this hes been happened.
![Image of function](Images/download12.png)

Now the same thing for a bivarite function:
![Image of function](Images/download3.png)

Now come to SGD[ stocastic gradient descent]
As we are not dealing with dataset directly, we here take simulation of that by adding random noise, and then follow how the diffficulty is getting increased,( okey, so you have to be good with random numbers in python, Here I attached one plot to get you idea how you can play with it to see differents) Follow the sampling and its distribution,
![Image of function](Images/random.png)

and for that we have to control the step size wisely(in the pic the step size is polinomially decreased)

![Image of function](Images/pr_gd.png)
![Image of function](Images/pr_gd_3d.png)
Here polinomially decreases step size is use dbut you can use exponentiall function to handle eta, or you may keep it constant or step wise decreasung,, a plot showing comparison among these are shown here,,, these different methods behave differently in different function.
![Image of function](Images/compare.png)


Finally we will see how tough this is for a non-convex surface,

![Image of function](Images/pr_gd_cncv.png)
![Image of function](Images/pr_gd_cncv_3d.png)

And finally We will use momentum updation to haldel these dificulties. Here to show the change I purposefully kept the momentum parameter high

![Image of function](Images/momentum.png)
![Image of function](Images/momentum_3d.png)
Thats it. Go to the ipynb files now.

**
Also I think, it will be better if anyone want to help me by just making the ipynb files more understable by seperation the topics.
If you feel hard anywhere, contact me in mahendranandi.0608@gmail.com
**
