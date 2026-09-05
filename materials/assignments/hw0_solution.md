---
title: "Problem Set 0 Solution"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw0_solution/
published: false   # solution not released
---

## Problem 3: Gradients and Hessians

**(a)** $f(x) = \frac{1}{2}x^TAx + b^Tx$ where $A$ is an $n\times n$ matrix and $x,b$ are $n$-dimensional column vectors.  
To find the derivative of $f(x)$, I will take $n = 3$ and try finding a pattern to deduce the expression.

$$
f(x) = \frac{1}{2} \begin{pmatrix}
x_1 & x_2 & x_3
\end{pmatrix} 
\begin{pmatrix}
a & b & c\\
b & e & o\\
c & o & d
\end{pmatrix} 
\begin{pmatrix}
x_1\\ x_2\\ x_3
\end{pmatrix} 
+ \begin{pmatrix}
b_1 & b_2 & b_3
\end{pmatrix} 
\begin{pmatrix}
x_1\\ x_2\\ x_3
\end{pmatrix}
$$

$$
= \frac{1}{2} \left( ax_1^2 + ex_2^2+ d x_3^2 + 2bx_2x_1 + 2cx_3x_1 + 2ox_3x_2+ b_1 x_1 + b_2 x_2 + b_3x_3 \right)
$$

Differentiating this expression with respect to $(x_1, x_2, x_3)$, we get

$$
\nabla f(x) = \begin{pmatrix}
\frac{\partial}{\partial x_1} \\
\frac{\partial}{\partial x_2} \\
\frac{\partial}{\partial x_3} 
\end{pmatrix} f(x) 
= \begin{pmatrix}
a & b & c\\
b & e & o\\
c & o & d
\end{pmatrix}
\begin{pmatrix}
x_1\\ x_2\\ x_3
\end{pmatrix} 
+ \begin{pmatrix}
b_1 \\ b_2 \\ b_3
\end{pmatrix}
$$

We can hence deduce that  
$$
\nabla f(x) = A x + b
$$

**(b)** $f(x) = g(h(x))$ where $f: \mathbb{R^n} \rightarrow \mathbb{R}, \; g: \mathbb{R} \rightarrow \mathbb{R}$ and $h:\mathbb{R^n} \rightarrow \mathbb{R}$ are all differentiable.

$$
\nabla f(x) = \begin{pmatrix}
\frac{\partial}{\partial x_1} \\
\frac{\partial}{\partial x_2} \\
\frac{\partial}{\partial x_3} 
\end{pmatrix} g(h(x))
$$

By the chain rule, $\frac{\partial}{\partial x_1}g(h(x)) =  \frac{\partial g}{\partial h}\frac{\partial h}{\partial x_1}$. We deduce

$$
\nabla f(x) = g'(h(x)) \nabla h(x)
$$

**(c)** Applying the gradient operator on the derivative operator $\nabla^T$, noting that $\nabla^T f(x) = (\nabla f(x))^T$, we use the expression for $\nabla f(x)$ that we found in part a:

$$
\nabla^2 f(x) = \begin{pmatrix}
\frac{\partial^2}{\partial x_1^2} & \frac{\partial^2}{\partial x_1x_2} & \frac{\partial^2}{\partial x_1x_3}\\
\frac{\partial^2}{\partial x_2x_1} & \frac{\partial^2}{\partial x_2^2} & \frac{\partial^2}{\partial x_2x_3}\\
\frac{\partial^2}{\partial x_3x_1} & \frac{\partial^2}{\partial x_3x_2} & \frac{\partial^2}{\partial x_3^2}
\end{pmatrix} f(x)
= \nabla \left[ \nabla f(x) \right]^T = \nabla\left[ (Ax + b)\right]^T
$$

Again, taking $n = 3$:

$$
\begin{pmatrix}
\frac{\partial}{\partial x_1} \\
\frac{\partial}{\partial x_2} \\
\frac{\partial}{\partial x_3} 
\end{pmatrix}
\begin{pmatrix}
ax_1 + bx_2 + cx_3 + b_1 &
bx_1 + ex_2 + ox_3 + b_2 &
cx_1 + ox_2 + dx_3 + b_3
\end{pmatrix} 
= \begin{pmatrix}
a & b & c\\ b & e & o \\ c & o & d
\end{pmatrix}
$$

Hence $\nabla^2 f(x) = A$.

**(d)** $f(x) = g(a^Tx)$ with $a$ an $n$-dimensional column vector. Taking $h(x) = a^Tx$, as we found in part b, $\nabla f(x) = g'(h(x)) \nabla (h(x))$. Taking $n = 3$:

$$
\nabla (a^Tx) = \begin{pmatrix}
\frac{\partial}{\partial x_1} \\
\frac{\partial}{\partial x_2} \\
\frac{\partial}{\partial x_3} 
\end{pmatrix} 
\begin{pmatrix}
a_1 x_1 + a_2x_2 + a_3x_3
\end{pmatrix} 
= \begin{pmatrix}
a_1 \\ a_2 \\ a_3
\end{pmatrix}
$$

Hence, $\nabla f(x) = g'(a^T x) a$.  

Now, applying the gradient operator a second time on $f(x)$:

$$
\nabla^2 f(x) = \nabla \left[ \nabla f(x) \right]^T = \nabla \left[g'(a^T x)a\right]^T = \nabla \left[a^T g'(a^T x)\right] = \left( \nabla a^T\right) g'(a^T x) + \left(\nabla g'(a^T x)\right) a^T
$$

Since $\nabla a^T = 0$, we get

$$
\nabla^2 f(x) = g''(a^Tx) a a^T \;\; (n\times n)
$$

## Problem 4: Positive Definite Matrices

**(a)** $z$ is an $n$-dimensional column vector, say $n = 3$, $z = \begin{pmatrix} z_1\\ z_2 \\ z_3 \end{pmatrix}$, then  

$$
A = zz^T = \begin{pmatrix}
z_1 ^2 & z_1z_2 & z_1z_3\\
z_2z_1 & z_2^2 & z_2z_3\\
z_3z_1 & z_3z_2 & z_3^2
\end{pmatrix}
$$

We notice that $A$ is symmetric, so it satisfies the first condition for a positive semi-definite matrix.  
Now, we check the sign of $x^T A x$ where $x$ is a 3-dimensional column vector:

$$
x^T Ax = x^Tzz^Tx = (z^Tx)^T(z^Tx) = (z_1x_1 + z_2x_2 + z_3x_3)^2
$$

Which is bigger than zero only when $z \ne (0,0,0, .., 0)$, and since we are accounting for the zero $z$ vector case, we get the second condition  

$$
x^T A x \ge 0
$$

**(b)** The Nullspace/Kernel of $A \in \mathbb{R}^{n\times n}$ is defined as

$$
\text{null}(A) = \{ x \mid Ax = 0 \}, \quad x \in \mathbb{R}^n
$$

We need to solve $Ax = 0$:

$$
\begin{bmatrix}
z_1^2 & z_1z_2 &  \cdots & z_1z_n \\
\vdots & \vdots & \vdots & \vdots \\
z_1z_n & z_2z_n & \cdots & z_n^2
\end{bmatrix} 
\begin{bmatrix}
x_1 \\ x_2 \\ \vdots \\ x_n
\end{bmatrix} = 0
$$

Simplifying, we get  

$$
z(z^T x) = 0
$$

Since $z \ne 0$, this implies $z^T x = 0$. Hence the Nullspace of $A$ is  

$$
\{ x \in \mathbb{R}^n \mid z^T x = 0 \}
$$

The Nullspace of $A$ is all vectors $x$ orthogonal to $z$, hence $\dim(\text{Null}(A)) = n-1$. Using the Rank-Nullity theorem:

$$
\dim(\text{Rank}(A)) + \dim(\text{Null}(A)) = n \implies \text{Rank}(A) = 1
$$

**(c)** 

- Symmetry:

$$
(BAB^T)^T = (B^T)^T A^T B^T = BA^T B^T = BAB^T
$$

Hence $BAB^T$ is symmetric.

- Positive semidefinite:

$$
x^T BAB^T x = (B^T x)^T A (B^T x) = y^T A y \quad \forall y \in \mathbb{R}^n
$$

But $x^T A x \ge 0$ for all $x$, hence $(B^T x)^T A (B^T x) \ge 0$ for all $x$.  

$\implies BAB^T$ is positive semi-definite.

## Problem 5

**(a)** We know that $A = T \Lambda T^{-1}$ with 
$$
T = [t^{(1)} \cdots t^{(n)}], \quad \Lambda = \text{diag}(\lambda_1, \dots, \lambda_n).
$$

We want to show that $t^{(i)}$ are the eigenvectors of $A$ such that $A t^{(i)} = \lambda_i t^{(i)}$.

To do that, we know that $A = T \Lambda T^{-1}$. Multiplying both sides by $T$, we get

$$
\begin{aligned}
&AT = (T \Lambda T^{-1}) T = (T \Lambda)(T^{-1} T) = T \Lambda I = T \Lambda \\
\implies &A [t^{(1)} \ \cdots \ t^{(n)}] 
   = [t^{(1)} \ \cdots \ t^{(n)}]
   \begin{bmatrix}
      \lambda_{1} & & \\
      & \ddots & \\
      & & \lambda_{n}
   \end{bmatrix} \\
\implies &[A t^{(1)} \ \cdots \ A t^{(n)}] 
   = [\lambda_1 t^{(1)} \ \cdots \ \lambda_n t^{(n)}].
\end{aligned}
$$

Matching the columns of the above matrices, we get exactly $A t^{(i)} = \lambda_i t^{(i)}$, making $(t^{(i)}, \lambda_i)$ the eigenvector/eigenvalue pairs of $A$, as desired.

**(b)** Here we know that $A$ is symmetric, thus $A = A^\top$. And $U = [u^{(1)} \ \cdots \ u^{(n)}]$ satisfies $U^\top U = I$ and $A = U \Lambda U^\top$.

So, multiplying both sides of $A = U \Lambda U^\top$ by $U$, we get

$$
\begin{aligned}
&AU = (U \Lambda U^\top) U = (U \Lambda)(U^\top U) = U \Lambda \\
\implies &A [u^{(1)} \ \cdots \ u^{(n)}] 
   = [u^{(1)} \ \cdots \ u^{(n)}]
   \begin{bmatrix}
      \lambda_{1} & & \\
      & \ddots & \\
      & & \lambda_{n}
   \end{bmatrix} \\
\implies &[A u^{(1)} \ \cdots \ A u^{(n)}] 
   = [\lambda_1 u^{(1)} \ \cdots \ \lambda_n u^{(n)}].
\end{aligned}
$$

Again, matching the columns of the above matrices, we get exactly $A u^{(i)} = \lambda_i u^{(i)}$, making $(u^{(i)}, \lambda_i)$ the eigenvector/eigenvalue pairs of $A$.

**(c)** Suppose that $A \in \mathbb{R}^{n \times n}$ is positive semidefinite (PSD). 

So, it is symmetric ($A^\top = A$). By the spectral theorem, it is diagonalizable by a real orthogonal matrix. Thus, we have
$
U^\top A U = \text{diag}(\lambda_1, \dots, \lambda_n),
$
where $U^\top U = I$ and $U=[ u^{(1)} \cdots u^{(n)}]$ with $A u^{(i)} = \lambda_i u^{(i)}$.

Also, since it is PSD, all vectors $x \in \mathbb{R}^n$ satisfy
$$
x^\top A x \geq 0.
$$
In particular, when applied on the eigenvectors of $A$, we get
$$
(u^{(i)})^\top A u^{(i)} \geq 0.
$$

But $A u^{(i)} = \lambda_i u^{(i)}$. So,
$$
(u^{(i)})^\top A u^{(i)} 
   = (u^{(i)})^\top (\lambda_i u^{(i)}) 
   = \lambda_i \left( (u^{(i)})^\top u^{(i)} \right) \geq 0.
$$

And $(u^{(i)})^\top u^{(i)} \geq 0$.  
Thus, $\lambda_i \geq 0$ for each $i$.

So, we have proved that all eigenvalues $\lambda_i$ of a positive semidefinite matrix $A$ are nonnegative.