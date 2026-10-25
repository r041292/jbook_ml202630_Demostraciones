# 4. ¿Dónde está el error en el árbol? (Ejercicio 6)

## Enunciado

```{figure} imagenes/enunciado_arbol.jpg
:width: 85%
:alt: Enunciado del ejercicio del árbol

Enunciado original: en la solución publicada del Ejercicio 6 (sección *Random Forest y
XGBoost*, <https://lihkir.github.io/MachineLearning/exercises_sols.html>) hay que encontrar
dónde está el error en el árbol. “No se sabe si el error está aquí o en los otros árboles.”
```

**Respuesta corta.** La raíz del árbol (*Like Gravity*) está bien elegida, pero hay
**errores de conteo** en varios diagramas. Los dos decisivos están en el **segundo nivel**:
el umbral $49$ de *Age* y la variable *Like Dogs*, ambos calculados sobre el subconjunto
*Like Gravity = SÍ*. Con los conteos correctos, el umbral **Age < 49 separa perfectamente**
(Gini $=0$), así que el árbol correcto es *Like Gravity → Age < 49* y **no** lleva el nodo
*Like Dogs*. El árbol publicado clasifica mal a una persona del propio conjunto de
entrenamiento.

---

## 4.1 Conceptos necesarios

Un **árbol de decisión** clasifica haciendo preguntas de sí/no sobre una variable a la vez
(“¿le gusta la gravedad?”, “¿edad < 49?”). Para elegir qué pregunta hacer en cada nodo se
mide qué tan “mezcladas” quedan las clases en los grupos resultantes.

**Índice de Gini de un nodo** con $N$ observaciones, de las cuales $n_{\text{Sí}}$ son SÍ y
$n_{\text{No}}$ son NO:

$$
\operatorname{Gini}(\text{nodo})=1-\Big(\frac{n_{\text{Sí}}}{N}\Big)^2-\Big(\frac{n_{\text{No}}}{N}\Big)^2 .
$$

Vale $0$ si el nodo es **puro** (todas las observaciones de la misma clase) y vale $0.5$ (su
máximo con dos clases) si está mitad y mitad.

**Gini de una partición** en dos grupos $D_1$ (con $N_1$ observaciones) y $D_2$ (con $N_2$):

$$
\operatorname{Gini}_{\text{split}}=\frac{N_1}{N}\operatorname{Gini}(D_1)+\frac{N_2}{N}\operatorname{Gini}(D_2).
$$

Se elige la pregunta con **menor** $\operatorname{Gini}_{\text{split}}$.

**Umbrales para una variable numérica.** Se ordenan los valores y se prueban los puntos
medios entre valores consecutivos, $\frac{x_{(i)}+x_{(i+1)}}{2}$.

```{tip}
**Dos controles de consistencia** que permiten detectar errores sin recalcular nada:
1. En cada diagrama, $\text{Sí}(D_1)+\text{Sí}(D_2)$ debe ser igual al total de SÍ del nodo
   padre (lo mismo para NO).
2. Los conteos deben poder obtenerse de la tabla de datos: cada persona cae en un solo grupo
   y conserva su etiqueta.
```

## 4.2 Los datos

Tabla 5.1 del curso (10 observaciones, respuesta *Gonna Be an Astronaut*; 1 = SÍ, 0 = NO):

| # | Age | Like Dogs | Like Gravity | Astronaut |
|---|---|---|---|---|
| 1 | 24 | 0 | 0 | 0 |
| 2 | 30 | 1 | 1 | 1 |
| 3 | 36 | 0 | 1 | 1 |
| 4 | 36 | 0 | 0 | 0 |
| 5 | 42 | 0 | 0 | 0 |
| 6 | 44 | 1 | 1 | 1 |
| 7 | 46 | 1 | 0 | 0 |
| 8 | 47 | 1 | 1 | 1 |
| 9 | 47 | 0 | 1 | 1 |
| 10 | 51 | 1 | 1 | **0** |

Totales: **5 SÍ y 5 NO**. Obsérvese la fila 10: la persona de 51 años **no** será astronauta
aunque le gustan los perros y la gravedad. Esa fila es la clave de los errores.

---

## 4.3 Nivel 1 (nodo raíz, 10 observaciones)

### Diagramas publicados

```{figure} imagenes/arbol_original_1.png
:width: 70%

Árboles publicados para la variable *Age* en la raíz.
```

```{figure} imagenes/arbol_original_2.png
:width: 80%

Árbol publicado para *Like Dogs* en la raíz.
```

```{figure} imagenes/arbol_original_3.png
:width: 80%

Árbol publicado para *Like Gravity* en la raíz.
```

### Recálculo completo

| Pregunta | $D_1$ (Sí / No) | $\operatorname{Gini}(D_1)$ | $D_2$ (Sí / No) | $\operatorname{Gini}(D_2)$ | $\operatorname{Gini}_{\text{split}}$ | Publicado |
|---|---|---|---|---|---|---|
| Age < 27 | 1 (0 / 1) | 0 | 9 (5 / 4) | 0.494 | 0.444 | 0.44 ✓ |
| Age < 33 | 2 (1 / 1) | 0.5 | 8 (4 / 4) | 0.5 | 0.500 | 0.5 ✓ |
| Age < 36 | 2 (1 / 1) | 0.5 | 8 (4 / 4) | 0.5 | 0.500 | 0.5 ✓ (*) |
| Age < 39 | 4 (2 / 2) | 0.5 | 6 (3 / 3) | 0.5 | 0.500 | 0.5 ✓ |
| Age < 43 | 5 (2 / 3) | 0.48 | 5 (3 / 2) | 0.48 | 0.480 | 0.48 ✓ |
| Age < 45 | 6 (3 / 3) | 0.5 | 4 (2 / 2) | 0.5 | 0.500 | 0.5 ✓ |
| Age < 46.5 | 7 (3 / 4) | 0.490 | 3 (2 / 1) | 0.444 | 0.476 | 0.47 (≈) |
| Age < 47 | 7 (3 / 4) | 0.490 | 3 (2 / 1) | 0.444 | 0.476 | 0.47 (≈) (*) |
| **Age < 49** | 9 (**5 / 4**) | 0.494 | 1 (**0 / 1**) | 0 | 0.444 | 0.44, pero **conteos invertidos** ✗ |
| **Like Dogs** | 5 (**3 / 2**) | **0.48** | 5 (**2 / 3**) | **0.48** | **0.480** | **0.32** ✗ |
| Like Gravity | 6 (5 / 1) | 0.278 | 4 (0 / 4) | 0 | **0.167** | 0.17 ✓ |

(*) Los umbrales 36 y 47 salen de promediar dos edades **repetidas** (36 y 36, 47 y 47).
Producen la misma partición que 33 y 46.5 respectivamente; son redundantes, pero no
erróneos. (≈) $0.4762$ se redondea a $0.48$, no a $0.47$; es un detalle menor.

### Error 1 — *Age < 49* en la raíz: etiquetas intercambiadas

El diagrama publicado dice $D_1$: Sí $=4$, No $=5$ y $D_2$ (la persona de 51 años): Sí $=1$,
No $=0$. Pero la persona de 51 años es **NO** (fila 10). Lo correcto es

- $D_1$ (edad < 49, filas 1–9): SÍ = filas 2, 3, 6, 8, 9 → **5 SÍ, 4 NO**;
- $D_2$ (edad ≥ 49, fila 10): **0 SÍ, 1 NO**.

Como el Gini es simétrico entre clases, el valor numérico ($0.44$) no cambia y **no afecta
la decisión** en este nivel, pero el diagrama está mal y anticipa el error grave del nivel 2.

### Error 2 — *Like Dogs* en la raíz: conteos imposibles

El diagrama publicado dice $D_1$(Sí) $=5$ con Sí $=4$, No $=1$ y $D_2$(No) $=5$ con Sí $=0$,
No $=5$. Hay **tres** inconsistencias:

1. **No cuadra con los totales**: daría $4+0=4$ SÍ y $1+5=6$ NO, pero hay 5 y 5.
2. **No cuadra consigo mismo**: con Sí $=0$, No $=5$ el nodo $D_2$ sería puro
   ($\operatorname{Gini}=0$), pero se calcula $1-(1/5)^2-(4/5)^2=0.32$.
3. **No cuadra con la tabla.** Contando:
   - *Like Dogs* = 1: filas 2, 6, 7, 8, 10 → astronauta 1, 1, 0, 1, 0 → **3 SÍ, 2 NO**.
   - *Like Dogs* = 0: filas 1, 3, 4, 5, 9 → astronauta 0, 1, 0, 0, 1 → **2 SÍ, 3 NO**.

Valor correcto: $\operatorname{Gini}(D_1)=\operatorname{Gini}(D_2)=1-0.6^2-0.4^2=0.48$, y
$\operatorname{Gini}_{\text{split}}=0.48$ (no $0.32$).

**¿Cambia la raíz?** No: *Like Gravity* ($0.167$) sigue siendo la mejor. La rama
*Like Gravity = NO* es pura (4 NO) y queda como hoja NO, tal como en la solución publicada.

---

## 4.4 Nivel 2 (subconjunto *Like Gravity* = SÍ, 6 observaciones)

| # | Age | Like Dogs | Astronaut |
|---|---|---|---|
| 2 | 30 | 1 | 1 |
| 3 | 36 | 0 | 1 |
| 6 | 44 | 1 | 1 |
| 8 | 47 | 1 | 1 |
| 9 | 47 | 0 | 1 |
| 10 | 51 | 1 | **0** |

Esta tabla **sí** aparece correctamente en la solución publicada. Totales: 5 SÍ, 1 NO.

```{figure} imagenes/arbol_original_4.png
:width: 70%

Árboles publicados para *Age* en el segundo nivel.
```

```{figure} imagenes/arbol_original_5.png
:width: 80%

Árbol publicado para *Like Dogs* en el segundo nivel.
```

### Recálculo completo

| Pregunta | $D_1$ (Sí / No) | $\operatorname{Gini}(D_1)$ | $D_2$ (Sí / No) | $\operatorname{Gini}(D_2)$ | $\operatorname{Gini}_{\text{split}}$ | Publicado |
|---|---|---|---|---|---|---|
| Age < 33 | 1 (1 / 0) | 0 | 5 (4 / 1) | 0.32 | 0.267 | 0.27 ✓ |
| Age < 40 | 2 (2 / 0) | 0 | 4 (3 / 1) | 0.375 | 0.250 | 0.25 ✓ |
| Age < 45.5 | 3 (3 / 0) | 0 | 3 (2 / 1) | 0.444 | 0.222 | 0.22 ✓ |
| Age < 47 | 3 (3 / 0) | 0 | 3 (2 / 1) | 0.444 | 0.222 | 0.22 ✓ |
| **Age < 49** | 5 (**5 / 0**) | **0** | 1 (**0 / 1**) | **0** | **0.000** | **0.26** ✗ |
| **Like Dogs** | 4 (**3 / 1**) | **0.375** | 2 (**2 / 0**) | **0** | **0.250** | **0.17** ✗ |

### Error 3 (el decisivo) — *Age < 49* en el nivel 2

El diagrama publicado dice $D_1$(edad < 49) $=5$ con Sí $=4$, No $=1$ y $D_2$ $=1$ con
Sí $=1$, No $=0$. Pero:

- las edades menores que 49 son 30, 36, 44, 47, 47, y **todas** son astronautas → **5 SÍ, 0 NO**;
- la única edad ≥ 49 es 51, que **no** es astronauta → **0 SÍ, 1 NO**.

Ambos grupos son **puros**, así que

$$
\operatorname{Gini}_{\text{split}}=\tfrac56\cdot0+\tfrac16\cdot0=0 ,
$$

el **mínimo posible**. Es el mismo intercambio de etiquetas del Error 1, pero aquí **sí
cambia la decisión**: en lugar de $0.26$ el valor correcto es $0$.

### Error 4 — *Like Dogs* en el nivel 2

El diagrama publicado dice $D_1$(Sí) $=4$ con Sí $=4$, No $=0$ y $D_2$(No) $=2$ con Sí $=1$,
No $=1$. Contando en la tabla de 6 filas (que la misma solución muestra):

- *Like Dogs* = 1: edades 30, 44, 47, 51 → astronauta 1, 1, 1, 0 → **3 SÍ, 1 NO**;
- *Like Dogs* = 0: edades 36, 47 → astronauta 1, 1 → **2 SÍ, 0 NO**.

Se ve que se “movió” a la persona de 51 años (NO) al grupo equivocado. Valor correcto:

$$
\operatorname{Gini}_{\text{split}}=\tfrac46\big(1-0.75^2-0.25^2\big)+\tfrac26\cdot0=\tfrac46\cdot0.375=0.25 .
$$

### Error 5 — inconsistencia en el texto

El texto dice que en *Age* “el umbral con el índice de Gini más bajo es 45.5 (0.22)” y a
continuación elige *Like Dogs* (0.17). Con los números correctos, ni 0.22 ni 0.17 son el
mínimo: el mínimo es **Age < 49 con Gini 0**.

---

## 4.5 Consecuencia: el árbol final está mal

```{figure} imagenes/arbol_original_6.png
:width: 75%

Árbol final publicado: Like Gravity → Like Dogs → Age < 45.5.
```

**Prueba de que es incorrecto:** clasifique la fila 10 (51 años, le gustan los perros y la
gravedad). El árbol publicado responde *Like Gravity = SÍ → Like Dogs = SÍ → astronauta SÍ*,
pero la etiqueta verdadera es **NO**. El árbol se equivoca en el **10 %** de su propio
conjunto de entrenamiento, algo que no debería ocurrir con hojas que el algoritmo declaró
puras.

**Árbol correcto:**

```text
Like Gravity?
├── NO  → Astronauta: NO     (4 NO, nodo puro)
└── SÍ  → Age < 49?
          ├── SÍ → Astronauta: SÍ   (5 SÍ, nodo puro)
          └── NO → Astronauta: NO   (1 NO, nodo puro)
```

Tiene **3 hojas** (no 4), todas puras, y **0 %** de error de entrenamiento.

**Verificación independiente con `scikit-learn`:**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text

d = np.array([(24,0,0,0),(30,1,1,1),(36,0,1,1),(36,0,0,0),(42,0,0,0),
              (44,1,1,1),(46,1,0,0),(47,1,1,1),(47,0,1,1),(51,1,1,0)])
arbol = DecisionTreeClassifier(criterion="gini", random_state=0).fit(d[:, :3], d[:, 3])
print(export_text(arbol, feature_names=["Age", "LikeDogs", "LikeGravity"]))
```

```text
|--- LikeGravity <= 0.50
|   |--- class: 0
|--- LikeGravity >  0.50
|   |--- Age <= 49.00
|   |   |--- class: 1
|   |--- Age >  49.00
|   |   |--- class: 0
```

(`scikit-learn` usa “$\le$” en lugar de “$<$” y los puntos medios entre valores **distintos**,
por eso aparece $\text{Age}\le49$; la partición es la misma.)

---

## Resumen

| # | Dónde | Qué está mal | ¿Cambia el árbol? |
|---|---|---|---|
| 1 | Raíz, Age < 49 | Etiquetas Sí/No intercambiadas (la persona de 51 años es NO) | No (Gini simétrico) |
| 2 | Raíz, Like Dogs | Conteos imposibles (4/1 y 0/5); correcto 3/2 y 2/3, Gini 0.48 | No (la raíz sigue siendo Like Gravity) |
| 3 | Nivel 2, Age < 49 | Debería ser 5/0 y 0/1, **Gini 0** (se reportó 0.26) | **Sí** |
| 4 | Nivel 2, Like Dogs | Debería ser 3/1 y 2/0, Gini 0.25 (se reportó 0.17) | **Sí** |
| 5 | Texto del nivel 2 | Dice que 0.22 es el mínimo y luego elige 0.17 | — |

El origen común de los errores es la **fila 10** (51 años, NO): en varios diagramas se la
contó como SÍ o se la puso en el grupo equivocado. Con los conteos correctos el árbol es
**Like Gravity → Age < 49**, con error de entrenamiento cero.
