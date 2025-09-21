---
title: "Problem Set 0 Solution"
layout: note
category: "Assignment"
permalink: /materials/assignments/hw0_solution/
---

## Problem 5

### (a)

We know that $A = T \Lambda T^{-1}$ with 
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

---

### (b)

Here we know that $A$ is symmetric, thus $A = A^\top$. And $U = [u^{(1)} \ \cdots \ u^{(n)}]$ satisfies $U^\top U = I$ and $A = U \Lambda U^\top$.

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

---

### (c)

Suppose that $A \in \mathbb{R}^{n \times n}$ is positive semidefinite (PSD). 

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