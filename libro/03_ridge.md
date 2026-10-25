# 3. Regresión Ridge: dos verificaciones

## Enunciado

```{figure} imagenes/enunciado_ridge.jpg
:width: 85%
:alt: Enunciado de las verificaciones de ridge

Enunciado original. Las dos verificaciones pedidas están resaltadas en naranja:
(1) “Nótese que $Z^T\mathbf 1=\mathbf 1^TZ=0$ (verifíquelo)” y
(2) las igualdades (3.11) para los datos estandarizados (“verifíquelas”).
```

## 3.1 Contexto: ¿para qué sirven estas verificaciones?

El modelo lineal inicial (Ecuación 3.3) es

$$
y_i=\beta_0+\beta_1x_{i1}+\dots+\beta_px_{ip}+\varepsilon_i,\qquad i=1,\dots,n .
$$

La regresión ridge penaliza el tamaño de los coeficientes. Como esa penalización depende de
las **unidades** de cada variable (no es lo mismo medir en metros que en centímetros), antes
se **estandarizan** los predictores. Sean

$$
\bar x_j=\frac1n\sum_{i=1}^nx_{ij},\qquad s_j^2=\frac1n\sum_{i=1}^n(x_{ij}-\bar x_j)^2,\qquad
z_{ij}=\frac{x_{ij}-\bar x_j}{s_j}.
$$

Llamamos $Z$ a la matriz $n\times p$ con entradas $z_{ij}$. El modelo se reescribe como

$$
y=\beta_0^*\mathbf 1+Z\beta_s+\varepsilon,\qquad\text{donde }\
\beta_{s,j}=s_j\beta_j\ \ \text{y}\ \ \beta_0^*=\beta_0+\sum_{j}\beta_j\bar x_j .
$$

(Basta sustituir $x_{ij}=\bar x_j+s_jz_{ij}$ en el modelo original para comprobarlo.)
El criterio ridge es

$$
S_\lambda(\beta_0^*,\beta_s)=(y-\beta_0^*\mathbf 1-Z\beta_s)^T(y-\beta_0^*\mathbf 1-Z\beta_s)+\lambda\,\beta_s^T\beta_s .
$$

Las dos verificaciones sirven para **separar el intercepto del resto** y concluir que, con
datos estandarizados, $\hat\beta_0^*=\bar y=0$. Así el problema se reduce a una fórmula
limpia para $\beta_s$.

---

## 3.2 Verificación 1: $Z^T\mathbf 1=\mathbf 1^TZ=0$

**Qué dice.** $\mathbf 1^TZ$ es un vector fila de tamaño $1\times p$ y $Z^T\mathbf 1$ es un vector
columna de tamaño $p\times1$. La afirmación es que **ambos son el vector de ceros**: cada
columna de $Z$ suma cero.

**Demostración.**

*Paso 1 — qué calcula el producto.* La componente $j$ de $Z^T\mathbf 1$ es el producto de la
fila $j$ de $Z^T$ (que es la columna $j$ de $Z$) por el vector de unos:

$$
\big(Z^T\mathbf 1\big)_j=\sum_{i=1}^n z_{ij}\cdot1=\sum_{i=1}^n z_{ij}.
$$

*Paso 2 — sustituir la definición de $z_{ij}$.* Como $s_j$ no depende de $i$, sale de la
suma:

$$
\sum_{i=1}^n z_{ij}=\sum_{i=1}^n\frac{x_{ij}-\bar x_j}{s_j}
=\frac{1}{s_j}\Big(\sum_{i=1}^n x_{ij}-\sum_{i=1}^n\bar x_j\Big).
$$

*Paso 3 — usar la definición de la media.* $\sum_{i=1}^nx_{ij}=n\bar x_j$ (porque
$\bar x_j=\frac1n\sum_ix_{ij}$), y $\sum_{i=1}^n\bar x_j=n\bar x_j$ (sumar $n$ veces la misma
constante). Entonces

$$
\sum_{i=1}^n z_{ij}=\frac{1}{s_j}\big(n\bar x_j-n\bar x_j\big)=0 .
$$

*Paso 4 — la otra igualdad.* Como esto vale para todo $j=1,\dots,p$, se tiene $Z^T\mathbf 1=0_{p}$.
Y como $\mathbf 1^TZ=(Z^T\mathbf 1)^T$ (propiedad $(AB)^T=B^TA^T$), también $\mathbf 1^TZ=0_p^T$. $\blacksquare$

```{note}
Solo se usó que las columnas están **centradas** (se les restó la media). La división por
$s_j$ no influye (siempre que $s_j\ne0$, es decir, que la variable no sea constante).
```

### Por qué es útil: la ecuación (3.8)

Derivemos $S_\lambda$ respecto de $\beta_0^*$. Llamemos $u=y-\beta_0^*\mathbf 1-Z\beta_s$, de
modo que $S_\lambda=u^Tu+\lambda\beta_s^T\beta_s$ y $\partial u/\partial\beta_0^*=-\mathbf 1$.
Por la regla de la cadena:

$$
\begin{aligned}
\frac{\partial S_\lambda}{\partial\beta_0^*}
&=-2\,\mathbf 1^T\big(y-\beta_0^*\mathbf 1-Z\beta_s\big)\\
&=-2\,\mathbf 1^Ty+2\beta_0^*\,\mathbf 1^T\mathbf 1+2\,\underbrace{\mathbf 1^TZ}_{=0}\,\beta_s\\
&=-2n\bar y+2n\beta_0^* ,
\end{aligned}
$$

usando $\mathbf 1^Ty=\sum_iy_i=n\bar y$ y $\mathbf 1^T\mathbf 1=n$. Esto es exactamente la
ecuación (3.8). **Sin la verificación 1 aparecería el término $2\,\mathbf 1^TZ\beta_s$** y el
intercepto quedaría mezclado con los demás coeficientes. Igualando a cero:
$\hat\beta_0^*=\bar y$.

### Complemento: derivada de una forma cuadrática y el estimador $\hat\beta_s$

El enunciado menciona que para derivar respecto de $\beta_s$ hace falta la derivada de una
forma cuadrática $x^TAx$. Escribiendo $x^TAx=\sum_k\sum_l x_kA_{kl}x_l$ y derivando respecto de
$x_r$ (aparece cuando $k=r$ o cuando $l=r$):

$$
\frac{\partial}{\partial x_r}x^TAx=\sum_lA_{rl}x_l+\sum_kx_kA_{kr}=(Ax)_r+(A^Tx)_r
\ \Longrightarrow\ \frac{\partial}{\partial x}x^TAx=(A+A^T)x,
$$

que vale $2Ax$ si $A$ es simétrica. Con esto (y $\partial(a^Tx)/\partial x=a$):

$$
\frac{\partial S_\lambda}{\partial\beta_s}=-2Z^Ty+2\beta_0^*\,\underbrace{Z^T\mathbf 1}_{=0}+2Z^TZ\beta_s+2\lambda\beta_s=0
\ \Longrightarrow\
\hat\beta_s=(Z^TZ+\lambda I_p)^{-1}Z^Ty .
$$

De nuevo la verificación 1 elimina el término cruzado con el intercepto.

---

## 3.3 Verificación 2: las igualdades (3.11)

Ahora se estandarizan **también** la respuesta y se usan los mismos nombres para los datos
transformados:

$$
y_i\ \leftarrow\ y_i-\bar y,\qquad x_{ij}\ \leftarrow\ \frac{x_{ij}-\bar x_j}{s_j}.
$$

Hay que verificar

$$
\sum_{i=1}^n y_i=0,\qquad \sum_{i=1}^n x_{ij}=0\ \ (j=1,\dots,p),\qquad \sum_{i=1}^n x_{ij}^2=n. \tag{3.11}
$$

Para no confundir, escribimos $y_i^{\text{orig}}$ y $x_{ij}^{\text{orig}}$ para los datos
originales.

### (a) $\sum_i y_i=0$

$$
\sum_{i=1}^n\big(y_i^{\text{orig}}-\bar y\big)=\sum_{i=1}^ny_i^{\text{orig}}-n\bar y=n\bar y-n\bar y=0 .
$$

### (b) $\sum_i x_{ij}=0$

Es exactamente el cálculo de la verificación 1 (Pasos 2 y 3):

$$
\sum_{i=1}^n\frac{x^{\text{orig}}_{ij}-\bar x_j}{s_j}=\frac{1}{s_j}\big(n\bar x_j-n\bar x_j\big)=0 .
$$

### (c) $\sum_i x_{ij}^2=n$

Elevamos al cuadrado y sacamos la constante $1/s_j^2$ de la suma:

$$
\sum_{i=1}^n\Big(\frac{x^{\text{orig}}_{ij}-\bar x_j}{s_j}\Big)^2
=\frac{1}{s_j^2}\sum_{i=1}^n\big(x^{\text{orig}}_{ij}-\bar x_j\big)^2 .
$$

Por la definición de la varianza, $\sum_{i=1}^n(x^{\text{orig}}_{ij}-\bar x_j)^2=n\,s_j^2$. Luego

$$
\sum_{i=1}^n x_{ij}^2=\frac{1}{s_j^2}\cdot n\,s_j^2=n . \qquad\blacksquare
$$

```{important}
La igualdad $\sum_ix_{ij}^2=n$ **depende de cómo se defina $s_j$**. Es cierta cuando
$s_j^2=\frac1n\sum_i(x_{ij}-\bar x_j)^2$ (varianza “poblacional”, divisor $n$). Si se usa la
varianza muestral con divisor $n-1$, se obtiene $\sum_ix_{ij}^2=n-1$. En `numpy`,
`x.std()` usa divisor $n$ por defecto (`ddof=0`); en `pandas`, `Series.std()` usa $n-1$.
```

**Interpretación matricial.** (b) y (c) juntos dicen que la diagonal de $Z^TZ$ vale $n$:
$Z^TZ/n$ es la **matriz de correlaciones** de los predictores. Por eso, después de
estandarizar, todas las variables “compiten en igualdad de condiciones” en la penalización.

### Conclusión: $\beta_0^*=0$

De (3.8), $\hat\beta_0^*=\bar y$, y por (a) la nueva respuesta tiene media cero:

$$
\hat\beta_0^*=\bar y=\frac1n\sum_{i=1}^ny_i=\frac1n\cdot0=0 .
$$

Por eso, con datos estandarizados, el modelo ridge se puede ajustar **sin intercepto** y la
solución es simplemente $\hat\beta_R=(X^TX+\lambda I_p)^{-1}X^Ty$.

### Ejemplo numérico

Tomemos $n=4$ y una sola variable $x^{\text{orig}}=(2,4,6,8)$, con $y^{\text{orig}}=(1,3,2,6)$.

- $\bar x=5$, $s^2=\frac14\big((-3)^2+(-1)^2+1^2+3^2\big)=\frac{20}{4}=5$, $s=\sqrt5$.
- $x=\frac{1}{\sqrt5}(-3,-1,1,3)$: suma $=0$ ✓ y suma de cuadrados $=\frac{9+1+1+9}{5}=4=n$ ✓.
- $\bar y=3$, $y=(-2,0,-1,3)$: suma $=0$ ✓.

---

## 3.4 Complemento: las propiedades (3.12) del estimador ridge

El recuadro superior del enunciado lista las propiedades (3.12). Aunque no se pide
verificarlas, se deducen en pocas líneas con lo anterior. Supuestos: $y=X\beta+\varepsilon$,
$\mathbb E[\varepsilon]=0$, $\operatorname{Cov}(\varepsilon)=\sigma^2I_n$, y $X$ fija. Sea
$W=(X^TX+\lambda I_p)^{-1}$ (simétrica), de modo que $\hat\beta_R=WX^Ty$.

1. **Esperanza.** $\mathbb E[\hat\beta_R]=WX^T\mathbb E[y]=WX^TX\beta$.
2. **Covarianza.** Como $\operatorname{Cov}(Ay)=A\operatorname{Cov}(y)A^T$:
   $\operatorname{Cov}(\hat\beta_R)=WX^T(\sigma^2I)XW=\sigma^2WX^TXW$.
3. **Sesgo.** Escribiendo $X^TX=(X^TX+\lambda I)-\lambda I=W^{-1}-\lambda I$:
   $\mathbb E[\hat\beta_R]-\beta=W(W^{-1}-\lambda I)\beta-\beta=-\lambda W\beta$.
4. **Error cuadrático medio.** Descomponiendo $\hat\beta_R-\beta=(\hat\beta_R-\mathbb E\hat\beta_R)+(\mathbb E\hat\beta_R-\beta)$,
   el término cruzado tiene esperanza cero, y
   $\mathbb E[(\hat\beta_R-\beta)^T(\hat\beta_R-\beta)]=\operatorname{tr}\operatorname{Cov}(\hat\beta_R)+\lambda^2\beta^TW^2\beta$.
   Con la descomposición espectral $X^TX=V\operatorname{diag}(l_1,\dots,l_p)V^T$, se tiene
   $W=V\operatorname{diag}\big(\tfrac{1}{l_j+\lambda}\big)V^T$ y
   $\operatorname{tr}(\sigma^2WX^TXW)=\sigma^2\sum_{j=1}^p\frac{l_j}{(l_j+\lambda)^2}$.

Resultado: **varianza total** $\sigma^2\sum_j\frac{l_j}{(l_j+\lambda)^2}$ (decrece con
$\lambda$) más **sesgo al cuadrado** $\lambda^2\beta^T(X^TX+\lambda I)^{-2}\beta$ (crece con
$\lambda$): el clásico compromiso sesgo-varianza.

---

## Resumen

- $Z^T\mathbf 1=\mathbf 1^TZ=0$ porque cada columna centrada suma $n\bar x_j-n\bar x_j=0$.
  Gracias a esto el intercepto se desacopla y $\partial S_\lambda/\partial\beta_0^*=-2n\bar y+2n\beta_0^*$.
- Tras estandarizar: $\sum y_i=0$, $\sum x_{ij}=0$ y $\sum x_{ij}^2=n$ (esta última con
  $s_j$ calculada con divisor $n$).
- En consecuencia $\hat\beta_0^*=\bar y=0$ y ridge se ajusta sin intercepto:
  $\hat\beta_R=(X^TX+\lambda I)^{-1}X^Ty$.
