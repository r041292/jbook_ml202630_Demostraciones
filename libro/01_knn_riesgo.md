# 1. Error verdadero, error empírico y la cota del 1-NN

## Enunciado

```{figure} imagenes/enunciado_knn.jpg
:width: 85%
:alt: Enunciado de la tarea de K-NN

Enunciado original. Se pide (a) demostrar lo que afirma la **Definición 2.3** y (b) en la
cota del vecino más cercano “ya se demostró el caso $\eta(x)$, hay que hacer el caso
$1-\eta(x)$”.
```

El capítulo tiene dos partes:

1. **Parte A** — Demostrar las propiedades que hacen que la Definición 2.3 tenga sentido:
   que el error verdadero es una probabilidad/esperanza, que el error de entrenamiento es un
   estimador insesgado y consistente del error verdadero, y cuál es el mejor error posible.
2. **Parte B** — Demostrar la cota
   $\displaystyle\mathbb E_{S\sim\mathcal D^m}[L_{\mathcal D}(h_S)]\le 2L_{\mathcal D}(h^\star)+c\,\mathbb E\|x-x'\|$
   para el clasificador 1-NN, completando el caso $\min\{\eta(x),1-\eta(x)\}=1-\eta(x)$.

---

## Intuición para no expertos

Imagine que quiere un programa $h$ que mire una radiografía $x$ y diga si hay fractura
($y=1$) o no ($y=0$).

- El **error verdadero** $L_{\mathcal D}(h)$ es *la proporción de pacientes del mundo real*
  a los que $h$ se equivocaría. Es lo que realmente nos importa, pero **no se puede
  calcular**, porque no conocemos a todos los pacientes posibles (no conocemos $\mathcal D$).
- El **error empírico** $L_S(h)$ es *la proporción de errores en las $m$ radiografías que sí
  tenemos* (la muestra de entrenamiento $S$). Este **sí se puede calcular**.

La pregunta natural es: ¿qué relación hay entre ambos? La Parte A responde: *en promedio son
iguales*, y cuando $m$ crece, el segundo se acerca al primero.

---

## Parte A. Demostraciones asociadas a la Definición 2.3

### A.0 Definiciones

Sea $\mathcal D$ una distribución sobre $\mathcal X\times\mathcal Y$ y $h:\mathcal X\to\mathcal Y$.

$$
L_{\mathcal D}(h) := \mathbb P_{(x,y)\sim\mathcal D}\big[h(x)\neq y\big]
:= \mathcal D\big(\{(x,y): h(x)\neq y\}\big). \tag{2.2}
$$

$$
L_S(h) := \frac{\big|\{i\in[m]: h(x_i)\neq y_i\}\big|}{m},\qquad [m]=\{1,\dots,m\}.
$$

### A.1 El error verdadero es la esperanza de la pérdida 0-1

**Proposición A.1.** Definiendo la pérdida 0-1 $\ell(h,(x,y)) = \mathbf 1[h(x)\neq y]$,

$$
L_{\mathcal D}(h)=\mathbb E_{(x,y)\sim\mathcal D}\big[\mathbf 1[h(x)\neq y]\big].
$$

**Demostración.**

1. Llamemos $A=\{(x,y): h(x)\neq y\}$ al conjunto de pares “mal clasificados”.
2. La variable $\mathbf 1[h(x)\neq y]=\mathbf 1_A(x,y)$ solo toma los valores $0$ y $1$.
3. La esperanza de una variable que toma valores $0$ y $1$ es
   $0\cdot\mathbb P[\text{vale }0]+1\cdot\mathbb P[\text{vale }1]=\mathbb P[(x,y)\in A]$.
4. Por definición, $\mathbb P[(x,y)\in A]=\mathcal D(A)=L_{\mathcal D}(h)$. $\blacksquare$

De la misma forma, el error empírico es un **promedio** de pérdidas 0-1:

$$
L_S(h)=\frac1m\sum_{i=1}^m \mathbf 1[h(x_i)\neq y_i],
$$

porque la suma de indicadoras **cuenta** cuántos índices $i$ cumplen $h(x_i)\neq y_i$, que es
exactamente $|\{i\in[m]:h(x_i)\ne y_i\}|$.

### A.2 El error empírico es un estimador insesgado del error verdadero

**Proposición A.2.** Si $h$ es **fija** (no depende de $S$) y $S\sim\mathcal D^m$ (las $m$
observaciones son independientes y con distribución $\mathcal D$), entonces

$$
\mathbb E_{S\sim\mathcal D^m}\big[L_S(h)\big]=L_{\mathcal D}(h).
$$

**Demostración.**

$$
\begin{aligned}
\mathbb E_{S}\big[L_S(h)\big]
&=\mathbb E_{S}\Big[\frac1m\sum_{i=1}^m \mathbf 1[h(x_i)\neq y_i]\Big]
&&\text{(A.1, forma de promedio)}\\
&=\frac1m\sum_{i=1}^m \mathbb E_{(x_i,y_i)\sim\mathcal D}\big[\mathbf 1[h(x_i)\neq y_i]\big]
&&\text{(linealidad de la esperanza)}\\
&=\frac1m\sum_{i=1}^m L_{\mathcal D}(h)
&&\text{(cada }(x_i,y_i)\sim\mathcal D\text{ y A.1)}\\
&=\frac1m\cdot m\,L_{\mathcal D}(h)=L_{\mathcal D}(h). && \blacksquare
\end{aligned}
$$

```{note}
“Insesgado” significa que **en promedio** sobre muchas muestras posibles, $L_S(h)$ acierta
el valor de $L_{\mathcal D}(h)$. Ojo: esto vale para $h$ **fija**. Si $h$ se elige mirando
$S$ (por ejemplo, $h=\mathrm{ERM}_{\mathcal H}(S)$), el error de entrenamiento suele ser
**optimista** ($L_S(h_S)$ tiende a ser menor que $L_{\mathcal D}(h_S)$). Esa es justamente
la razón por la que se necesitan cotas como la de la Parte B.
```

### A.3 Varianza y consistencia: más datos, mejor estimación

**Proposición A.3.** Con $h$ fija y $p:=L_{\mathcal D}(h)$,

$$
\operatorname{Var}\big(L_S(h)\big)=\frac{p(1-p)}{m}\le\frac{1}{4m},
$$

y por tanto $L_S(h)\to L_{\mathcal D}(h)$ en probabilidad cuando $m\to\infty$.

**Demostración.**

1. Las variables $Z_i=\mathbf 1[h(x_i)\ne y_i]$ son independientes y cada una es Bernoulli
   con parámetro $p$ (vale 1 con probabilidad $p$). Su varianza es $p(1-p)$.
2. Para variables independientes, la varianza de la suma es la suma de las varianzas:
   $\operatorname{Var}\big(\sum_i Z_i\big)=m\,p(1-p)$.
3. Como $L_S(h)=\frac1m\sum_i Z_i$ y $\operatorname{Var}(aW)=a^2\operatorname{Var}(W)$:
   $\operatorname{Var}(L_S(h))=\frac{1}{m^2}\,m\,p(1-p)=\frac{p(1-p)}{m}$.
4. La función $p(1-p)$ alcanza su máximo $1/4$ en $p=1/2$, de ahí la cota $1/(4m)$.
5. Por la desigualdad de Chebyshev, para cualquier $\varepsilon>0$:
   $\mathbb P\big[|L_S(h)-L_{\mathcal D}(h)|>\varepsilon\big]\le\dfrac{1}{4m\varepsilon^2}\xrightarrow[m\to\infty]{}0.$ $\blacksquare$

(Una cota más fuerte, la de **Hoeffding**, da
$\mathbb P[|L_S(h)-L_{\mathcal D}(h)|>\varepsilon]\le 2e^{-2m\varepsilon^2}$.)

### A.4 El mejor clasificador posible (Bayes) y su error

Para la Parte B necesitamos saber **cuál es el menor error verdadero alcanzable**.
Tomemos $\mathcal Y=\{0,1\}$ y definamos la **probabilidad condicional de la clase 1**:

$$
\eta(x):=\mathbb P[y=1\mid x].
$$

El **clasificador de Bayes** es $h^\star(x)=\mathbf 1[\eta(x)\ge 1/2]$: “predecir 1 si la
clase 1 es la más probable”.

**Proposición A.4.** Para cualquier $h:\mathcal X\to\{0,1\}$,

$$
L_{\mathcal D}(h)\ \ge\ L_{\mathcal D}(h^\star)=\mathbb E_{x\sim\mathcal D_x}\big[\min\{\eta(x),1-\eta(x)\}\big].
$$

**Demostración.**

1. Fijemos un $x$. Si $h(x)=1$, nos equivocamos cuando $y=0$, lo que ocurre con
   probabilidad $1-\eta(x)$. Si $h(x)=0$, nos equivocamos cuando $y=1$, con probabilidad
   $\eta(x)$. Es decir,
   $\mathbb P[h(x)\ne y\mid x]=h(x)\,(1-\eta(x))+(1-h(x))\,\eta(x)$.
2. Para ese $x$, el menor error posible es $\min\{\eta(x),1-\eta(x)\}$, y lo consigue
   justamente $h^\star$ (elige la clase con mayor probabilidad).
3. Promediando sobre $x$ (ley de la esperanza total):
   $L_{\mathcal D}(h)=\mathbb E_x\big[\mathbb P[h(x)\ne y\mid x]\big]\ge\mathbb E_x\big[\min\{\eta(x),1-\eta(x)\}\big]=L_{\mathcal D}(h^\star).$ $\blacksquare$

---

## Parte B. La cota del clasificador 1-NN

### B.1 Planteamiento

- $\mathcal X\subseteq\mathbb R^d$, $\mathcal Y=\{0,1\}$.
- **Hipótesis de suavidad (Lipschitz):** existe $c>0$ tal que
  $|\eta(x)-\eta(x')|\le c\,\|x-x'\|$ para todo $x,x'$. En palabras: *puntos cercanos tienen
  probabilidades de clase parecidas*. Sin algo así, mirar al vecino no serviría de nada.
- **Regla 1-NN:** dada $S$, para un punto nuevo $x$ se busca el punto de entrenamiento más
  cercano, $x'=x_{\pi_1(x)}$, y se predice su etiqueta: $h_S(x)=y'$.

**Teorema.**

$$
\mathbb E_{S\sim\mathcal D^m}\big[L_{\mathcal D}(h_S)\big]\ \le\ 2\,L_{\mathcal D}(h^\star)
\;+\;c\ \mathbb E_{x\sim\mathcal D_x,\;S_x\sim\mathcal D_x^m}\big[\|x-x'\|\big].
$$

Interpretación: **el 1-NN se equivoca, como mucho, el doble que el mejor clasificador
posible**, más un término que mide qué tan lejos queda el vecino más cercano (y que se hace
pequeño cuando hay muchos datos).

### B.2 Paso 1: probabilidad de error en un punto

Fijemos $x$ y su vecino $x'$. La etiqueta verdadera $y$ de $x$ y la etiqueta $y'$ de $x'$ son
independientes, con $\mathbb P[y=1]=\eta(x)$ y $\mathbb P[y'=1]=\eta(x')$. El 1-NN se equivoca
si $y\ne y'$:

$$
\mathbb P[y\ne y']=\underbrace{\eta(x)\,(1-\eta(x'))}_{y=1,\ y'=0}+\underbrace{\eta(x')\,(1-\eta(x))}_{y=0,\ y'=1}.
$$

### B.3 Paso 2: una identidad algebraica

**Afirmación.**
$\eta(x)(1-\eta(x'))+\eta(x')(1-\eta(x)) = 2\eta(x)(1-\eta(x)) + (\eta(x)-\eta(x'))(2\eta(x)-1).$

Escribamos $a=\eta(x)$, $b=\eta(x')$ para abreviar.

- Lado izquierdo: $a(1-b)+b(1-a)=a-ab+b-ab=a+b-2ab$.
- Lado derecho: $2a(1-a)+(a-b)(2a-1)=2a-2a^2+(2a^2-a-2ab+b)=a+b-2ab$.

Ambos lados coinciden. $\checkmark$

### B.4 Paso 3: acotar el segundo término usando Lipschitz

- Como $a\in[0,1]$, se tiene $2a-1\in[-1,1]$, es decir $|2\eta(x)-1|\le 1$.
- Por la hipótesis Lipschitz, $|\eta(x)-\eta(x')|\le c\|x-x'\|$.

Por lo tanto $(\eta(x)-\eta(x'))(2\eta(x)-1)\le|\eta(x)-\eta(x')|\cdot|2\eta(x)-1|\le c\|x-x'\|$, y

$$
\mathbb P[y\ne y']\ \le\ 2\eta(x)(1-\eta(x))+c\,\|x-x'\|. \tag{B.1}
$$

### B.5 Paso 4: la desigualdad con el mínimo (el punto pedido)

**Lema.** Para todo $\eta\in[0,1]$:
$\qquad 2\eta(1-\eta)\ \le\ 2\min\{\eta,1-\eta\}.$

La demostración se divide en dos casos, según cuál de los dos números es el mínimo.

**Caso 1 (ya hecho en clase): $\min\{\eta,1-\eta\}=\eta$.** Entonces $\eta\le 1-\eta$.
Como $0\le 1-\eta\le 1$ y $\eta\ge0$:

$$
2\eta(1-\eta)\ \le\ 2\eta\cdot 1 \;=\; 2\eta \;=\; 2\min\{\eta,1-\eta\}.
$$

**Caso 2 (el que se pide): $\min\{\eta,1-\eta\}=1-\eta$.**

1. Que el mínimo sea $1-\eta$ significa que $1-\eta\le\eta$, o sea $\eta\ge\tfrac12$.
2. Sabemos que $\eta\in[0,1]$, así que $\eta\le 1$.
3. Además $1-\eta\ge 0$, de modo que multiplicar la desigualdad $\eta\le1$ por $2(1-\eta)$
   **no cambia su sentido**:

$$
2\eta(1-\eta)\ \le\ 2\cdot 1\cdot(1-\eta)\;=\;2(1-\eta)\;=\;2\min\{\eta,1-\eta\}. \qquad\blacksquare
$$

Una forma equivalente, “espejo” del caso 1: si se define $\tilde\eta=1-\eta$, el caso 2 se
convierte en el caso 1 para $\tilde\eta$, porque la expresión
$2\eta(1-\eta)=2\tilde\eta(1-\tilde\eta)$ es **simétrica** al cambiar $\eta$ por $1-\eta$, y
$\min\{\eta,1-\eta\}=\min\{\tilde\eta,1-\tilde\eta\}=\tilde\eta$.

```{warning}
**Sobre la cadena escrita en la diapositiva.** El texto escribe, para el caso 1,
$2\eta(1-\eta)\le 2(1-\eta)^2\le 2\min\{\eta,1-\eta\}$. La **primera** desigualdad es correcta
(si $\eta\le1-\eta$ entonces $\eta(1-\eta)\le(1-\eta)(1-\eta)$), pero la **segunda es falsa**
en general: con $\eta=0.1$ se obtiene $2(0.9)^2=1.62$, que **no** es menor o igual que
$2\min\{0.1,0.9\}=0.2$. La cadena correcta es la de arriba,
$2\eta(1-\eta)\le 2\eta=2\min\{\eta,1-\eta\}$.
Lo mismo pasaría si se “copiara” la cadena al caso 2:
$2\eta(1-\eta)\le2\eta^2$ es cierto, pero $2\eta^2\le 2(1-\eta)$ falla (con $\eta=0.9$:
$1.62>0.2$). Por eso aquí se usa la cota $\eta\le1$ y no un cuadrado.
```

Verificación numérica rápida del lema en el caso 2:

| $\eta$ | $2\eta(1-\eta)$ | $2\min\{\eta,1-\eta\}=2(1-\eta)$ | ¿Se cumple? |
|---|---|---|---|
| 0.5 | 0.50 | 1.00 | sí |
| 0.7 | 0.42 | 0.60 | sí |
| 0.9 | 0.18 | 0.20 | sí |
| 1.0 | 0.00 | 0.00 | sí (igualdad) |

### B.6 Paso 5: juntar todo

Combinando (B.1) con el lema (válido en **ambos** casos, luego para todo $x$):

$$
\mathbb P[h_S(x)\ne y]\ \le\ 2\min\{\eta(x),1-\eta(x)\}+c\,\|x-x'\|.
$$

Tomamos esperanza sobre $x\sim\mathcal D_x$ y sobre la muestra $S$ (que determina $x'$):

$$
\begin{aligned}
\mathbb E_S\big[L_{\mathcal D}(h_S)\big]
&=\mathbb E_{S,\,x}\big[\mathbb P[y\ne y']\big]\\
&\le 2\,\mathbb E_x\big[\min\{\eta(x),1-\eta(x)\}\big]+c\,\mathbb E_{S,\,x}\big[\|x-x'\|\big]
&&\text{(linealidad)}\\
&=2\,L_{\mathcal D}(h^\star)+c\,\mathbb E_{x\sim\mathcal D_x,\,S_x\sim\mathcal D_x^m}\big[\|x-x'\|\big].
&&\text{(Proposición A.4)}
\end{aligned}
$$

Esto es exactamente la cota pedida. $\blacksquare$

---

## Resumen

- $L_{\mathcal D}(h)$ es la probabilidad de error (equivalentemente, la esperanza de la
  pérdida 0-1); $L_S(h)$ es su versión muestral.
- Para $h$ fija, $L_S(h)$ es **insesgado** y su varianza $p(1-p)/m$ tiende a 0.
- El mejor error alcanzable es $L_{\mathcal D}(h^\star)=\mathbb E[\min\{\eta,1-\eta\}]$.
- El caso $\min=1-\eta$ se demuestra con $2\eta(1-\eta)\le 2(1-\eta)$, usando solo que
  $\eta\le 1$ y $1-\eta\ge0$.
- Resultado final: el 1-NN tiene, en esperanza, a lo sumo el **doble** del error de Bayes
  más un término de “distancia al vecino” que se reduce con más datos.

**Referencia:** Shalev-Shwartz, S. y Ben-David, S. (2014). *Understanding Machine Learning*,
cap. 2 y cap. 19 (Lema 19.1 y Teorema 19.3).
