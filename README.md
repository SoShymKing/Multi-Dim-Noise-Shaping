![feedback system](feedback_loop_filter.png)

$$
V(z) = U(z)\cdot\frac{H(z)}{1+H(z)}+Q(z)\cdot\frac{1}{1+H(z)}
$$

$$
V(w) = K(w)\cdot U(w) + \left(1-K(w)\right)\cdot Q(w)
$$

각 필터는 각 축에만 종속적이므로

$$
V(w_x, y) = K(w_x,y)\cdot U(w_x,y) + \left(1-K(w_x,y)\right)\cdot Q(w_x,y)
$$

$$
V(w_x, y) = K(w_x)\cdot U(w_x,y) + \left(1-K(w_x)\right)\cdot Q(w_x,y)
$$

$$
v_x(x,y) = k(x) *_x s(x,y) + n(x,y) -k(x) *_x n(x,y) = k(x) *_x s(x,y) + g(x) *_x n(x,y)
$$

비슷하게

$$
v_y(x,y) = k(y) *_y s(x,y) + n(x,y) -k(y) *_y n(x,y)
$$

곱하면

$$
o(x,y) = v_x(x,y) \cdot v_y(x,y) = \left(k(x) *_x s(x,y) + g(x) *_x n(x,y)\right)\cdot\left(k(y) *_y s(x,y) + g(y) *_y n(x,y)\right)
$$

$$
o =(k *_x s)\cdot (k *_y s) + (k *_x s)\cdot( g *_y n) + (k *_y s)\cdot( g *_y n) + ( g *_y n)\cdot(g *_y n)
$$

$$
(k(x) *_x s(x,y))\cdot(g(y) *_y n(x,y))
$$

$$
\left(K(w_x) \cdot S(w_x,w_y)\right) *_{w_x,w_y} \left(G(w_y) \cdot N(w_x,w_y)\right)
$$

$$
O =
(K(w_x) \cdot S(w_x,w_y))*_{w_x,w_y} (K(w_y) \cdot S(w_x,w_y)) + \left(K(w_x) \cdot S(w_x,w_y)\right) *_{w_x,w_y} \left(G(w_y) \cdot N(w_x,w_y)\right) +
\left(K(w_y) \cdot S(w_x,w_y)\right) *_{w_x,w_y} \left(G(w_x) \cdot N(w_x,w_y)\right) + (G(w_x) \cdot N(w_x,w_y))*_{w_x,w_y}(G(w_y) \cdot N(w_x,w_y))
$$

이때
low freq 에서 convolution과 mutiply는 근사하므로,

$$
\left(K(w_y) \cdot S(w_x,w_y)\right) *_{w_x,w_y} \left(G(w_x) \cdot N(w_x,w_y)\right)  \approx \left(K(w_y) \cdot S(w_x,w_y)\right) \cdot \left(G(w_x) \cdot N(w_x,w_y)\right) 
$$

$$
\left(K(w_y) \cdot S(w_x,w_y)\right) \cdot \left(G(w_x) \cdot N(w_x,w_y)\right) = K(w_y) \cdot S(w_x,w_y) \cdot G(w_x) \cdot N(w_x,w_y)\\
= K(w_y) \cdot S(w_x,w_y) \cdot \left(1-K(w_x)\right)\cdot N(w_x,w_y)
$$

1차 DSM에서
$H = 1/w $, then $K=\frac{1}{1+w}$, $1-K = \frac{w}{1+w}$

$$ 
\therefore \left(K(w_y) \cdot S(w_x,w_y)\right) *_{w_x,w_y} \left(G(w_x) \cdot N(w_x,w_y)\right)  \approx \frac{1}{1+w_y}\cdot \frac{w_x}{1+w_x} \cdot S(w_x,w_y)\cdot N(w_x,w_y)
$$
이때 공간에 대한 주파수를 고려해보면, 축적이 왜곡되지 않는 상황에서는 $w_x = w_y$가 성립한다.
$$ 
\frac{1}{1+w_y}\cdot \frac{w_x}{1+w_x} \cdot S(w_x,w_y)\cdot N(w_x,w_y) = \frac{1}{1+w}\cdot \frac{w}{1+w} \cdot S(w,w)\cdot N(w,w)
$$

$w$가 0에 수렴하면 $\frac{w}{(1+w)^2} $ 또한 0에 수렴한다.

따라서 $w$가 충분히 작은 상황에서 
$\left(K(w_y) \cdot S(w_x,w_y)\right) *_{w_x,w_y} \left(G(w_x) \cdot N(w_x,w_y)\right) $은 0 으로 수렴한다.

$$
\therefore O \approx
(K(w_x) \cdot S(w_x,w_y))*_{w_x,w_y} (K(w_y) \cdot S(w_x,w_y)) 
$$
