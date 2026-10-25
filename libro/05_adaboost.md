# 5. Demostración del algoritmo AdaBoost

## Enunciado

```{figure} imagenes/enunciado_adaboost.jpg
:width: 85%
:alt: Enunciado de AdaBoost

Enunciado original: demostrar el Algoritmo 5.1 (AdaBoost).
```

Se tienen $N$ observaciones $(y_n,x_n)$ con $y_n\in\{-1,1\}$ y se busca un clasificador

$$
f(x)=\sgn\{F(x)\},\qquad F(x)=\sum_{k=1}^K a_k\,\phi(x;\theta_k), \tag{5.2–5.3}
$$

donde cada $\phi(x;\theta_k)\in\{-1,1\}$ es un **clasificador base** (o “débil”), por ejemplo un
árbol de un solo corte.

**Qué hay que demostrar.** Que cada paso del Algoritmo 5.1 no es arbitrario, sino que se
**deduce** de minimizar, paso a paso, la **pérdida exponencial**

$$
J=\sum_{n=1}^N\exp\big(-y_nF(x_n)\big).
$$

En concreto demostraremos:

| Paso del algoritmo | Qué se demuestra |
|---|---|
| Inicialización $\omega^{(1)}_n=1/N$ | Corresponde a $F_0\equiv0$. |
| Pasos 1–2: elegir $\theta_i$ minimizando $P_i$ | Minimizar $J$ en $\theta$ equivale a minimizar el **error ponderado** $P_i$. |
| Paso 3: $a_i=\tfrac12\ln\frac{1-P_i}{P_i}$ | Es el valor de $a$ que minimiza $J$. |
| Pasos 4–8: actualización y normalización de $\omega$ | Los pesos son la pérdida exponencial acumulada de cada punto, normalizada. |
| Paso 12: $f=\sgn\big(\sum a_k\phi_k\big)$ | Es la regla (5.2). |

Como complemento, se demuestra que **el error de entrenamiento cae exponencialmente**.

---

## 5.1 Intuición para no expertos

AdaBoost entrena muchos clasificadores sencillos **uno tras otro**. Después de cada uno,
**aumenta el peso** de los ejemplos que se clasificaron mal, para que el siguiente se
concentre en ellos. Al final, todos “votan”, y el voto de cada uno pesa $a_k$: más peso
cuanto más preciso fue.

¿Por qué la pérdida exponencial? Fíjese en el **margen** $y_nF(x_n)$:

- si es positivo, el punto está bien clasificado ($\sgn F(x_n)=y_n$);
- si es negativo, está mal clasificado.

La función $e^{-y_nF(x_n)}$ es pequeña cuando el margen es grande y positivo, y **crece muy
rápido** cuando el punto está mal clasificado. Además, siempre es **mayor o igual** que el
error 0-1: $\mathbf 1[y_n\ne\sgn F(x_n)]\le e^{-y_nF(x_n)}$. Minimizarla, entonces, empuja a
reducir los errores.

---

## 5.2 Estrategia: minimización por etapas (“stagewise”)

Minimizar $J$ en todos los $a_k,\theta_k$ a la vez es muy difícil. AdaBoost lo hace **de a
uno**: en la iteración $i$, los términos anteriores quedan **fijos** y solo se añade uno nuevo,

$$
F_i(x)=F_{i-1}(x)+a\,\phi(x;\theta),\qquad F_0(x)\equiv0 ,
$$

y se escogen $(a_i,\theta_i)$ como

$$
(a_i,\theta_i)=\arg\min_{a,\theta}\ J(a,\theta),\qquad
J(a,\theta)=\sum_{n=1}^N\exp\Big(-y_n\big[F_{i-1}(x_n)+a\,\phi(x_n;\theta)\big]\Big).
$$

---

## 5.3 Paso A: aparición de los pesos $\omega_n^{(i)}$

Como $e^{u+v}=e^ue^v$:

$$
J(a,\theta)=\sum_{n=1}^N\underbrace{\exp\big(-y_nF_{i-1}(x_n)\big)}_{\text{no depende de }a,\theta}\ \exp\big(-y_n\,a\,\phi(x_n;\theta)\big).
$$

Definimos los **pesos** como esa parte fija, normalizada para que sume 1:

$$
\omega^{(i)}_n=\frac{\exp\big(-y_nF_{i-1}(x_n)\big)}{\sum_{m=1}^N\exp\big(-y_mF_{i-1}(x_m)\big)} .
\tag{A.1}
$$

Dividir $J$ por una constante positiva no cambia dónde está su mínimo, así que basta minimizar

$$
\tilde J(a,\theta)=\sum_{n=1}^N\omega^{(i)}_n\exp\big(-y_n\,a\,\phi(x_n;\theta)\big). \tag{A.2}
$$

**Inicialización.** Para $i=1$, $F_0\equiv0$, así que $\exp(0)=1$ para todos y
$\omega^{(1)}_n=1/N$. Esto es exactamente el paso “Inicializa $\omega^{(1)}_n=1/N$”.

**Actualización de pesos (pasos 4–8).** Como $F_i=F_{i-1}+a_i\phi(\cdot;\theta_i)$,

$$
\exp\big(-y_nF_i(x_n)\big)=\exp\big(-y_nF_{i-1}(x_n)\big)\cdot\exp\big(-y_n a_i\phi(x_n;\theta_i)\big).
$$

Es decir, **el peso nuevo es el peso viejo multiplicado por $\exp(-y_na_i\phi(x_n;\theta_i))$**,
y luego se normaliza para que sume 1:

$$
\omega^{(i+1)}_n=\frac{\omega^{(i)}_n\exp\big(-y_na_i\phi(x_n;\theta_i)\big)}{Z_i},\qquad
Z_i=\sum_{n=1}^N\omega^{(i)}_n\exp\big(-y_na_i\phi(x_n;\theta_i)\big).
$$

Eso es lo que hacen los pasos 4–6 (acumular el numerador y la suma $Z_i$) y 7–8 (dividir por
$Z_i$). (Que dividir por $Z_i$ da la misma normalización que en (A.1) se sigue por inducción:
ambas expresiones son proporcionales a $\exp(-y_nF_i(x_n))$ y suman 1.)

---

## 5.4 Paso B: para $a$ fijo, elegir $\theta$ = minimizar el error ponderado (pasos 1–2)

Como $y_n\in\{-1,1\}$ y $\phi\in\{-1,1\}$, el producto $y_n\phi(x_n;\theta)$ solo puede valer:

- $+1$ si el clasificador **acierta** en $x_n$, y entonces el término es $\omega_n e^{-a}$;
- $-1$ si **se equivoca**, y entonces el término es $\omega_n e^{a}$.

Definimos el **error ponderado** del clasificador base:

$$
P(\theta)=\sum_{n=1}^N\omega^{(i)}_n\,\mathbf 1\big[y_n\ne\phi(x_n;\theta)\big]
=\sum_{n:\ \text{error}}\omega^{(i)}_n .
$$

Como los pesos suman 1, los aciertos pesan $1-P(\theta)$. Separando la suma (A.2):

$$
\tilde J(a,\theta)=e^{-a}\big(1-P(\theta)\big)+e^{a}P(\theta)
=e^{-a}+\big(e^{a}-e^{-a}\big)P(\theta). \tag{B.1}
$$

Para $a>0$, el factor $e^{a}-e^{-a}$ es **positivo**, y $e^{-a}$ no depende de $\theta$. Por
tanto,

$$
\arg\min_\theta\tilde J(a,\theta)=\arg\min_\theta P(\theta),
$$

y **el óptimo en $\theta$ no depende de $a$**. Esto justifica los pasos 1–2: elegir
$\theta_i$ como el clasificador base de **menor error ponderado** y llamar $P_i=P(\theta_i)$.

## 5.5 Paso C: el coeficiente óptimo $a_i$ (paso 3)

Con $\theta_i$ ya elegido, $\tilde J$ es función solo de $a$:

$$
g(a)=(1-P_i)e^{-a}+P_ie^{a}.
$$

**Derivada igual a cero:**

$$
g'(a)=-(1-P_i)e^{-a}+P_ie^{a}=0
\iff P_ie^{a}=(1-P_i)e^{-a}
\iff e^{2a}=\frac{1-P_i}{P_i}
\iff \boxed{\,a_i=\frac12\ln\frac{1-P_i}{P_i}\,}.
$$

**Es un mínimo:** $g''(a)=(1-P_i)e^{-a}+P_ie^{a}>0$ para todo $a$ (si $0<P_i<1$), así que $g$
es estrictamente convexa y el punto crítico es su único mínimo global.

**Observaciones.**

- $a_i>0\iff\frac{1-P_i}{P_i}>1\iff P_i<\frac12$: el clasificador base debe ser **mejor que
  lanzar una moneda** con los pesos actuales. Esto también garantiza que el argumento de
  la sección 5.4 (que usó $a>0$) es coherente.
- Cuanto menor es $P_i$, mayor es $a_i$: los clasificadores más precisos votan con más peso.
- Si $P_i=0$, $a_i=+\infty$ (un solo clasificador perfecto basta); en la práctica se detiene
  el algoritmo o se acota $P_i$.

## 5.6 Paso D: el clasificador final (pasos 9–12)

Tras $K$ iteraciones, $F_K(x)=\sum_{k=1}^Ka_k\phi(x;\theta_k)$ y la predicción es
$f(x)=\sgn\big(F_K(x)\big)$, que es la regla (5.2). Con esto queda demostrado que **cada paso
del Algoritmo 5.1 es el resultado de minimizar por etapas la pérdida exponencial**. $\blacksquare$

---

## 5.7 Complemento: valor de $Z_i$ y caída exponencial del error de entrenamiento

### $Z_i$ en forma cerrada

Sustituyendo $a_i$ en $g$ (es decir, $e^{a_i}=\sqrt{(1-P_i)/P_i}$):

$$
Z_i=(1-P_i)\sqrt{\frac{P_i}{1-P_i}}+P_i\sqrt{\frac{1-P_i}{P_i}}
=\sqrt{P_i(1-P_i)}+\sqrt{P_i(1-P_i)}=2\sqrt{P_i(1-P_i)} .
$$

Como $P_i(1-P_i)\le\frac14$, se tiene $Z_i\le1$, con igualdad solo si $P_i=\frac12$.

**Dato curioso.** Con los pesos nuevos, los puntos que $\phi_i$ clasificó mal pesan en total
$\frac{P_ie^{a_i}}{Z_i}=\frac{\sqrt{P_i(1-P_i)}}{2\sqrt{P_i(1-P_i)}}=\frac12$. Es decir, el
clasificador recién añadido queda **exactamente como una moneda** respecto de los nuevos pesos,
lo que obliga a que el siguiente clasificador sea diferente.

### Teorema: el error de entrenamiento decae exponencialmente

$$
\frac1N\sum_{n=1}^N\mathbf 1\big[f(x_n)\ne y_n\big]\ \le\ \prod_{i=1}^K Z_i
=\prod_{i=1}^K2\sqrt{P_i(1-P_i)}\ \le\ \exp\Big(-2\sum_{i=1}^K\gamma_i^2\Big),
\qquad\gamma_i=\tfrac12-P_i .
$$

**Demostración.**

1. **Desenrollar los pesos.** Aplicando la actualización $K$ veces desde $\omega^{(1)}_n=1/N$:
   $$
   \omega^{(K+1)}_n=\frac{1}{N}\,\frac{\exp\big(-y_n\sum_{k=1}^Ka_k\phi(x_n;\theta_k)\big)}{\prod_{k=1}^KZ_k}
   =\frac{\exp\big(-y_nF_K(x_n)\big)}{N\prod_kZ_k}.
   $$
2. **Los pesos suman 1.** Sumando en $n$:
   $\dfrac1N\sum_n\exp\big(-y_nF_K(x_n)\big)=\prod_kZ_k$.
3. **El error 0-1 está por debajo de la exponencial.** Si $f(x_n)\ne y_n$, entonces
   $y_nF_K(x_n)\le0$ y $e^{-y_nF_K(x_n)}\ge1$; si acierta, la indicadora es 0 y la exponencial
   es positiva. En ambos casos $\mathbf 1[f(x_n)\ne y_n]\le e^{-y_nF_K(x_n)}$. Promediando y
   usando el paso 2 se obtiene la primera desigualdad.
4. **Cota exponencial.** Con $P_i=\frac12-\gamma_i$:
   $2\sqrt{P_i(1-P_i)}=2\sqrt{\big(\tfrac12-\gamma_i\big)\big(\tfrac12+\gamma_i\big)}=\sqrt{1-4\gamma_i^2}$.
   Usando $1+t\le e^{t}$ con $t=-4\gamma_i^2$:
   $\sqrt{1-4\gamma_i^2}\le\sqrt{e^{-4\gamma_i^2}}=e^{-2\gamma_i^2}$. Multiplicando sobre
   $i$ se obtiene la segunda desigualdad. $\blacksquare$

**Consecuencia.** Si cada clasificador base es al menos un poco mejor que el azar,
$P_i\le\frac12-\gamma$ con $\gamma>0$, entonces el error de entrenamiento es
$\le e^{-2K\gamma^2}$, que **tiende a cero exponencialmente** con el número de iteraciones $K$.

### Ejemplo numérico

Si todos los clasificadores base tienen $P_i=0.3$ ($\gamma=0.2$):

- $a_i=\frac12\ln\frac{0.7}{0.3}\approx0.424$;
- los pesos de los errores se multiplican por $e^{0.424}\approx1.53$ y los de los aciertos
  por $e^{-0.424}\approx0.65$ (antes de normalizar);
- $Z_i=2\sqrt{0.3\cdot0.7}\approx0.917$;
- tras $K=50$ iteraciones, el error de entrenamiento es $\le0.917^{50}\approx0.013$, es decir,
  menor que $1.3\%$.

---

## Resumen

- AdaBoost minimiza la pérdida exponencial $\sum_ne^{-y_nF(x_n)}$ **añadiendo un término a la
  vez**.
- Los pesos $\omega_n^{(i)}$ son la pérdida exponencial acumulada de cada punto, normalizada;
  por eso se actualizan multiplicando por $e^{-y_na_i\phi_i(x_n)}$ y dividiendo por $Z_i$.
- Para $a>0$, minimizar la pérdida en $\theta$ es lo mismo que minimizar el **error
  ponderado** $P_i$.
- El coeficiente óptimo es $a_i=\frac12\ln\frac{1-P_i}{P_i}$, positivo si $P_i<\frac12$.
- El error de entrenamiento está acotado por $\prod_i2\sqrt{P_i(1-P_i)}\le e^{-2\sum_i\gamma_i^2}$.

**Referencias:** Freund, Y. y Schapire, R. (1997). *A decision-theoretic generalization of
on-line learning and an application to boosting*. Theodoridis, S. (2020). *Machine Learning:
A Bayesian and Optimization Perspective*, cap. 7. Hastie, Tibshirani y Friedman (2009),
*The Elements of Statistical Learning*, §10.4.
