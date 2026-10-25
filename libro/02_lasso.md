# 2. Estimación de parámetros en la regresión Lasso

## Enunciado

```{figure} imagenes/enunciado_lasso.jpg
:width: 85%
:alt: Enunciado del ejercicio de Lasso

Enunciado original: “Queda como ejercicio para el lector encontrar parámetros de estimación
lasso, tal como se realizó en el caso de la regresión ridge. Se sugiere investigar sobre el
algoritmo LARS de Efron et al. (2004)”.
```

Se pide minimizar

$$
S_\lambda(\beta)=(y-X\beta)^T(y-X\beta)+\lambda\sum_{j=1}^{p}|\beta_j|,\qquad \lambda\ge0,
$$

con los datos normalizados como en la Ecuación (3.11) (ver capítulo {doc}`03_ridge`):

$$
\sum_{i=1}^n y_i=0,\qquad \sum_{i=1}^n x_{ij}=0,\qquad \sum_{i=1}^n x_{ij}^2=n,\qquad j=1,\dots,p.
$$

## Hoja de ruta

1. Por qué en lasso **no** se puede repetir el procedimiento de ridge “tal cual”.
2. La herramienta que reemplaza a la derivada: el **subgradiente**.
3. Las **condiciones de optimalidad** del lasso (análogas a “derivada = 0”).
4. **Caso de una variable / predictores ortogonales**: fórmula cerrada (umbral suave).
5. **Caso general**: fórmula en el conjunto activo y algoritmo de **descenso por coordenadas**.
6. **Algoritmo LARS** (Efron et al., 2004) y su modificación para lasso.
7. Ejemplo numérico verificado.

---

## 2.1 Recordatorio: ¿qué se hizo en ridge?

En ridge la penalización es $\lambda\sum_j\beta_j^2=\lambda\beta^T\beta$. Toda la función es
**derivable**, así que basta con derivar e igualar a cero:

$$
\frac{\partial}{\partial\beta}\Big[(y-X\beta)^T(y-X\beta)+\lambda\beta^T\beta\Big]
=-2X^Ty+2X^TX\beta+2\lambda\beta=0
\ \Longrightarrow\
\hat\beta_R=(X^TX+\lambda I_p)^{-1}X^Ty .
$$

En lasso la penalización es $\lambda\sum_j|\beta_j|$, y **la función $|t|$ no es derivable en
$t=0$** (tiene un “pico”). Justamente ese pico es lo que permite que algunos coeficientes
queden **exactamente en cero**, pero obliga a usar otra herramienta.

## 2.2 El intercepto

Igual que en ridge, si el modelo incluye un intercepto $\beta_0$ y los datos están
centrados, la penalización **no** involucra a $\beta_0$; derivando respecto de $\beta_0$
(que sí es derivable):

$$
\frac{\partial S_\lambda}{\partial\beta_0}=-2\sum_{i=1}^n\Big(y_i-\beta_0-\sum_j x_{ij}\beta_j\Big)
=-2n\bar y+2n\beta_0+2\sum_j\beta_j\underbrace{\sum_i x_{ij}}_{=0}=0
\ \Longrightarrow\ \hat\beta_0=\bar y=0 .
$$

Por eso, a partir de aquí, se trabaja sin intercepto.

## 2.3 La herramienta: subgradiente del valor absoluto

Para una función convexa $f$, un número $g$ es un **subgradiente** de $f$ en $t_0$ si la recta
de pendiente $g$ que pasa por $(t_0,f(t_0))$ queda **por debajo** de la gráfica:
$f(t)\ge f(t_0)+g\,(t-t_0)$ para todo $t$. El conjunto de subgradientes se llama
**subdiferencial**, $\partial f(t_0)$.

Para $f(t)=|t|$:

$$
\partial|t|=\begin{cases}\{+1\} & t>0,\\ \{-1\} & t<0,\\ [-1,\,1] & t=0.\end{cases}
$$

Es decir: donde $|t|$ es derivable, el subgradiente es la derivada usual; en el pico $t=0$,
cualquier pendiente entre $-1$ y $1$ “cabe por debajo” de la V.

**Regla de optimalidad (convexa).** $\hat\beta$ minimiza una función convexa $S$ si y solo si
$0\in\partial S(\hat\beta)$. Es la generalización de “derivada igual a cero”.

```{note}
$S_\lambda$ es convexa porque es la suma de una función cuadrática convexa ($\|y-X\beta\|^2$,
cuya matriz hessiana $2X^TX$ es semidefinida positiva) y de valores absolutos (convexos).
Por eso la condición $0\in\partial S_\lambda(\hat\beta)$ es **necesaria y suficiente**.
```

## 2.4 Condiciones de optimalidad (KKT) del lasso

Derivamos la parte cuadrática, como en ridge, y usamos el subgradiente para la parte absoluta.
Llamamos $x_j$ a la columna $j$ de $X$:

$$
0\in -2X^T(y-X\beta)+\lambda\, s,\qquad s_j\in\partial|\beta_j| .
$$

Componente a componente, esto dice:

$$
\boxed{\;
x_j^T(y-X\hat\beta)=\tfrac{\lambda}{2}\operatorname{sign}(\hat\beta_j)\ \ \text{si }\hat\beta_j\neq0,
\qquad
\big|x_j^T(y-X\hat\beta)\big|\le\tfrac{\lambda}{2}\ \ \text{si }\hat\beta_j=0.
\;}
\tag{L.1}
$$

**Interpretación.** $x_j^T(y-X\hat\beta)$ es (proporcional a) la **correlación** entre la
variable $j$ y el residuo. Lasso deja “activas” solo las variables cuya correlación con el
residuo alcanza el nivel $\lambda/2$, y **todas las activas tienen la misma correlación en
valor absoluto**. Las demás quedan en cero. Esta observación es la base de LARS.

Consecuencia inmediata: si $\lambda\ge\lambda_{\max}:=2\max_j|x_j^Ty|$, entonces
$\hat\beta=0$ cumple (L.1), así que **todos los coeficientes son cero**.

## 2.5 Caso sencillo: predictores ortogonales (fórmula cerrada)

Supongamos que las columnas son ortogonales. Con la normalización (3.11),
$x_j^Tx_j=n$ y $x_j^Tx_k=0$ si $j\ne k$, es decir $X^TX=nI_p$.

**Paso 1 — mínimos cuadrados.** Sin penalización, $\hat\beta^{MC}=(X^TX)^{-1}X^Ty=\frac1n X^Ty$,
o sea $\hat\beta^{MC}_j=\frac1n x_j^Ty$.

**Paso 2 — el problema se separa.** Como $X^TX=nI$,

$$
S_\lambda(\beta)=y^Ty-2\beta^TX^Ty+n\beta^T\beta+\lambda\sum_j|\beta_j|
=y^Ty+\sum_{j=1}^p\Big(n\beta_j^2-2n\hat\beta_j^{MC}\beta_j+\lambda|\beta_j|\Big),
$$

así que cada $\beta_j$ se puede minimizar **por separado**. Dividiendo por $n$, hay que
minimizar en $b\in\mathbb R$:

$$
g(b)=b^2-2\hat\beta^{MC}_j\,b+\tfrac{\lambda}{n}|b| .
$$

**Paso 3 — condición $0\in\partial g(b)$:** $\;2b-2\hat\beta^{MC}_j+\frac{\lambda}{n}s=0$ con
$s\in\partial|b|$. Sea $\gamma=\frac{\lambda}{2n}$. Analizamos los tres casos:

- **Si $b>0$** ($s=1$): $b=\hat\beta^{MC}_j-\gamma$. Esto es consistente ($b>0$) solo si
  $\hat\beta^{MC}_j>\gamma$.
- **Si $b<0$** ($s=-1$): $b=\hat\beta^{MC}_j+\gamma$. Consistente solo si
  $\hat\beta^{MC}_j<-\gamma$.
- **Si $b=0$** ($s\in[-1,1]$): necesitamos $\hat\beta^{MC}_j=\gamma s$ para algún
  $s\in[-1,1]$, es decir $|\hat\beta^{MC}_j|\le\gamma$.

Los tres casos cubren todos los valores posibles de $\hat\beta^{MC}_j$ sin superponerse, y se
resumen en el **operador de umbral suave** (*soft-thresholding*):

$$
\boxed{\;\hat\beta^{L}_j=\operatorname{sign}\big(\hat\beta^{MC}_j\big)\Big(|\hat\beta^{MC}_j|-\frac{\lambda}{2n}\Big)_+\;}
\qquad (t)_+=\max\{t,0\}.
\tag{L.2}
$$

**Comparación con ridge en el mismo caso.** Con $X^TX=nI$,
$\hat\beta_R=(nI+\lambda I)^{-1}X^Ty=\frac{n}{n+\lambda}\hat\beta^{MC}$.

| | Ridge | Lasso |
|---|---|---|
| Efecto | **Multiplica** por $\frac{n}{n+\lambda}<1$ | **Resta** $\frac{\lambda}{2n}$ al valor absoluto |
| ¿Llega a cero? | Nunca (si $\hat\beta^{MC}_j\neq0$) | Sí, si $|\hat\beta^{MC}_j|\le\frac{\lambda}{2n}$ |

**Ejemplo.** Con $n=10$, $\lambda=20$ (así $\frac{\lambda}{2n}=1$ y $\frac{n}{n+\lambda}=\frac13$) y
$\hat\beta^{MC}=(3,\,-1.2,\,0.4)$:

- Lasso: $(3-1,\ -(1.2-1),\ 0)=(2,\ -0.2,\ 0)$ → **el tercer coeficiente se anula**.
- Ridge: $(1,\ -0.4,\ 0.133)$ → todos se encogen, ninguno se anula.

## 2.6 Caso general: fórmula en el conjunto activo

Cuando $X^TX$ no es diagonal no hay una fórmula de una sola línea, pero sí una **fórmula
explícita una vez se conoce qué coeficientes son distintos de cero**. Sea
$\mathcal A=\{j:\hat\beta_j\ne0\}$ (conjunto activo) y $s_{\mathcal A}$ el vector de signos.
Por (L.1), y como $X\hat\beta=X_{\mathcal A}\hat\beta_{\mathcal A}$:

$$
X_{\mathcal A}^T\big(y-X_{\mathcal A}\hat\beta_{\mathcal A}\big)=\tfrac{\lambda}{2}s_{\mathcal A}
\ \Longrightarrow\
\boxed{\;\hat\beta_{\mathcal A}=\big(X_{\mathcal A}^TX_{\mathcal A}\big)^{-1}\Big(X_{\mathcal A}^Ty-\tfrac{\lambda}{2}s_{\mathcal A}\Big),\qquad \hat\beta_{\mathcal A^c}=0.\;}
\tag{L.3}
$$

Compárese con ridge, $\hat\beta_R=(X^TX+\lambda I)^{-1}X^Ty$: ridge modifica la **matriz**,
lasso modifica el **vector** $X^Ty$ restándole $\frac{\lambda}{2}s$ y además elimina las
columnas inactivas.

Dos consecuencias importantes de (L.3):

1. Mientras $\mathcal A$ y $s_{\mathcal A}$ no cambien, $\hat\beta_{\mathcal A}(\lambda)$ es
   una **función lineal de $\lambda$**. Por eso la *trayectoria* del lasso
   ($\lambda\mapsto\hat\beta(\lambda)$) es **lineal a trozos**.
2. La trayectoria solo “se quiebra” cuando (a) una variable inactiva alcanza
   $|x_j^T(y-X\hat\beta)|=\lambda/2$ y **entra**, o (b) un coeficiente activo llega a $0$ y
   **sale**. Calcular esos quiebres es exactamente lo que hace LARS.

### Algoritmo de descenso por coordenadas

Una forma práctica y muy usada (es la de `scikit-learn` y `glmnet`) es aplicar (L.2) a una
variable a la vez, manteniendo las demás fijas:

1. Inicializar $\beta=0$.
2. Para $j=1,\dots,p$:
   - residuo parcial $r^{(j)}=y-\sum_{k\ne j}x_k\beta_k$;
   - actualizar $\displaystyle\beta_j\leftarrow\frac{1}{x_j^Tx_j}\,\operatorname{sign}\big(x_j^Tr^{(j)}\big)\Big(|x_j^Tr^{(j)}|-\frac{\lambda}{2}\Big)_+$
     (con datos normalizados, $x_j^Tx_j=n$).
3. Repetir el paso 2 hasta que $\beta$ deje de cambiar.

La actualización del paso 2 es simplemente el problema de una sola variable de la sección
2.5 aplicado al residuo parcial. Como $S_\lambda$ es convexa y la parte no derivable es
separable por coordenadas, este método converge al mínimo global.

## 2.7 El algoritmo LARS (Least Angle Regression)

**Idea.** LARS es una versión “prudente” de la selección hacia adelante. En lugar de meter
una variable completa al modelo, avanza **poco a poco** en la dirección que mantiene a todas
las variables activas **igualmente correlacionadas** con el residuo (la dirección que forma
el mismo ángulo con todas ellas, de ahí “ángulo mínimo”).

**Algoritmo (Efron, Hastie, Johnstone y Tibshirani, 2004):**

1. Estandarizar los predictores y centrar $y$ (Ecuación 3.11). Iniciar $\beta=0$, residuo
   $r=y$.
2. Encontrar la variable más correlacionada con el residuo: $j_1=\arg\max_j|x_j^Tr|$.
   $\mathcal A=\{j_1\}$.
3. Mover $\beta_{\mathcal A}$ en la dirección de mínimos cuadrados de $r$ sobre
   $X_{\mathcal A}$: $\;\delta=(X_{\mathcal A}^TX_{\mathcal A})^{-1}X_{\mathcal A}^Tr$.
   A lo largo de este movimiento, las correlaciones de las variables activas con el residuo
   **bajan todas al mismo ritmo** y siguen siendo iguales entre sí.
4. Detenerse en cuanto otra variable $k\notin\mathcal A$ alcance la misma correlación
   absoluta que las activas; añadirla a $\mathcal A$.
5. Repetir 3–4 hasta incluir todas las variables (al final se llega a mínimos cuadrados).

**Modificación lasso (LARS-lasso).** Añadir una regla: *si un coeficiente activo llega a
cero en el paso 3, sacarlo de $\mathcal A$ y recalcular la dirección*. Con esa regla, cada
punto de la trayectoria cumple (L.1), es decir, **LARS-lasso calcula la trayectoria completa
del lasso para todos los valores de $\lambda$**, con un costo similar a un solo ajuste de
mínimos cuadrados.

**¿Por qué coincide con el lasso?** Por (L.1), en la solución del lasso las variables activas
tienen correlación con el residuo exactamente igual a $\lambda/2$ en valor absoluto, y las
inactivas una correlación menor o igual. LARS mantiene precisamente esa propiedad (“todas las
activas empatadas en correlación máxima”), y el nivel común de correlación hace el papel de
$\lambda/2$, que va bajando desde $\lambda_{\max}/2$ hasta $0$. Dentro de cada tramo la
solución viene dada por (L.3), que es lineal en $\lambda$.

## 2.8 Ejemplo numérico verificado

Se simularon $n=50$ observaciones con $p=4$ predictores (los verdaderos coeficientes son
$(3,0,-2,0)$), se normalizaron según (3.11) y se calculó el lasso con $\lambda=100$ de tres
maneras distintas.

```python
import numpy as np
from sklearn.linear_model import Lasso, lars_path

rng = np.random.default_rng(0)
n, p = 50, 4
X = rng.normal(size=(n, p)); X[:, 1] += 0.5 * X[:, 0]
y = X @ np.array([3.0, 0.0, -2.0, 0.0]) + rng.normal(size=n)
X = (X - X.mean(0)) / X.std(0)          # sum x_ij = 0, sum x_ij^2 = n
y = y - y.mean()                        # sum y_i = 0

def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0)

def lasso_cd(X, y, lam, iters=1000):    # descenso por coordenadas (sección 2.6)
    b = np.zeros(X.shape[1])
    for _ in range(iters):
        for j in range(X.shape[1]):
            r = y - X @ b + X[:, j] * b[j]
            b[j] = soft(X[:, j] @ r, lam / 2) / (X[:, j] @ X[:, j])
    return b

lam = 100.0
b_cd = lasso_cd(X, y, lam)
# scikit-learn minimiza (1/2n)||y-Xb||^2 + alpha||b||_1  =>  alpha = lambda / (2n)
b_sk = Lasso(alpha=lam / (2 * n), fit_intercept=False).fit(X, y).coef_
A = b_cd != 0
b_A = np.linalg.solve(X[:, A].T @ X[:, A], X[:, A].T @ y - lam / 2 * np.sign(b_cd[A]))  # (L.3)
alphas, active, coefs = lars_path(X, y, method="lasso")  # LARS-lasso
```

Resultados:

| Método | $\hat\beta_1$ | $\hat\beta_2$ | $\hat\beta_3$ | $\hat\beta_4$ |
|---|---|---|---|---|
| Mínimos cuadrados | 2.7983 | 0.0018 | −2.0343 | 0.2678 |
| Lasso, descenso por coordenadas | 1.6277 | **0** | −0.8534 | **0** |
| Lasso, `scikit-learn` | 1.6277 | **0** | −0.8534 | **0** |
| Lasso, fórmula (L.3) con $\mathcal A=\{1,3\}$ | 1.6277 | — | −0.8534 | — |

Comprobación de las condiciones (L.1) con $\lambda/2=50$:
$X^T(y-X\hat\beta)=(50,\ 27.52,\ -50,\ -11.17)$. Las variables activas (1 y 3) tienen
correlación exactamente $\pm50$, con el signo de su coeficiente, y las inactivas (2 y 4)
tienen $|\cdot|\le 50$. ✓

Trayectoria LARS-lasso: las variables entran en el orden $1\to3\to4\to2$, en los valores
$\lambda=254.05$ ($=\lambda_{\max}$), $176.61$, $16.51$ y $0.50$. Para $\lambda=100$ estamos en
el tramo $176.61>\lambda>16.51$, donde el conjunto activo es $\{1,3\}$, que coincide con lo
obtenido arriba.

---

## Resumen

- Lasso no tiene la forma cerrada de ridge porque $|\beta_j|$ no es derivable en 0; se usa el
  **subgradiente** y la condición $0\in\partial S_\lambda(\hat\beta)$.
- Condiciones de optimalidad: $x_j^T(y-X\hat\beta)=\frac{\lambda}{2}\operatorname{sign}(\hat\beta_j)$ si
  $\hat\beta_j\ne0$ y $|x_j^T(y-X\hat\beta)|\le\frac{\lambda}{2}$ si $\hat\beta_j=0$.
- Predictores ortogonales: $\hat\beta^L_j=\operatorname{sign}(\hat\beta^{MC}_j)(|\hat\beta^{MC}_j|-\frac{\lambda}{2n})_+$
  (umbral suave), lo que explica los **ceros exactos**.
- Caso general: $\hat\beta_{\mathcal A}=(X_{\mathcal A}^TX_{\mathcal A})^{-1}(X_{\mathcal A}^Ty-\frac{\lambda}{2}s_{\mathcal A})$,
  lineal a trozos en $\lambda$; se calcula con descenso por coordenadas o con **LARS-lasso**.

**Referencias:**
Tibshirani, R. (1996). *Regression shrinkage and selection via the lasso*. JRSS-B.
Efron, B., Hastie, T., Johnstone, I. y Tibshirani, R. (2004). *Least angle regression*.
Annals of Statistics, 32(2).
Hastie, T., Tibshirani, R. y Friedman, J. (2009). *The Elements of Statistical Learning*, §3.4.
