# Introducción

**Autor:** Rubiel Velasquez

Este libro reúne, en un solo lugar, la solución detallada de cinco tareas de un curso de
*Machine Learning*. La mayoría son **demostraciones matemáticas**; una de ellas es una
**auditoría numérica** de un árbol de decisión construido a mano.

La idea es que cada capítulo pueda leerse por separado y que una persona **no experta**
pueda seguir el razonamiento. Por eso:

- cada capítulo empieza con el **enunciado original** (la fotografía de la tarea);
- antes de demostrar algo se explica **qué significa** y **por qué importa**;
- las demostraciones van **paso a paso**, justificando cada igualdad o desigualdad;
- al final de cada capítulo hay un **resumen** con las ideas clave.

## Contenido

| Capítulo | Tema | Qué se hace |
|---|---|---|
| {doc}`01_knn_riesgo` | K-NN / teoría del aprendizaje | Se demuestran las propiedades del error verdadero y empírico (Definición 2.3) y se completa el caso $1-\eta(x)$ de la cota del clasificador 1-NN. |
| {doc}`02_lasso` | Regresión Lasso | Ejercicio para el lector: obtener los estimadores lasso “como se hizo con ridge”, incluyendo el umbral suave y el algoritmo LARS. |
| {doc}`03_ridge` | Regresión Ridge | Las dos verificaciones pedidas: $Z^T\mathbf 1 = \mathbf 1^T Z = 0$ y las igualdades (3.11) de los datos estandarizados. Como complemento se deducen las propiedades (3.12). |
| {doc}`04_arbol_decision` | Árboles de decisión (Gini) | Se localizan y corrigen los errores del árbol del Ejercicio 6 de la sección *Random Forest y XGBoost*. |
| {doc}`05_adaboost` | AdaBoost | Se demuestra que el Algoritmo 5.1 se obtiene minimizando la pérdida exponencial, y que su error de entrenamiento cae exponencialmente. |

## Notación común

A lo largo del libro se usa la siguiente notación:

| Símbolo | Significado |
|---|---|
| $\mathcal X$, $\mathcal Y$ | Espacio de entradas (características) y de etiquetas (respuestas). |
| $\mathcal D$ | Distribución de probabilidad (desconocida) que genera los pares $(x,y)$. |
| $S=\{(x_1,y_1),\dots,(x_m,y_m)\}$ | Muestra de entrenamiento, extraída de forma independiente de $\mathcal D$. |
| $h$ | Hipótesis, predictor o clasificador: una función $h:\mathcal X\to\mathcal Y$. |
| $\mathbb E[\cdot]$, $\mathbb P[\cdot]$ | Esperanza (promedio teórico) y probabilidad. |
| $\mathbf 1[A]$ | Función indicadora: vale $1$ si $A$ es cierto y $0$ si no. |
| $X$ ($n\times p$) | Matriz de diseño: $n$ observaciones (filas) y $p$ variables (columnas). |
| $\mathbf 1$ | Vector columna de unos, $\mathbf 1=(1,\dots,1)^T$. |
| $\|v\|$ | Norma euclidiana de un vector, $\sqrt{v^Tv}$. |

## Herramientas matemáticas que se usan

No se necesita más que lo siguiente (se recuerda en el capítulo donde aparece):

1. **Linealidad de la esperanza**: $\mathbb E[aX+bY]=a\,\mathbb E[X]+b\,\mathbb E[Y]$.
2. **Esperanza de una indicadora**: $\mathbb E[\mathbf 1[A]]=\mathbb P[A]$.
3. **Derivadas de vectores**: $\dfrac{\partial}{\partial \beta}(a^T\beta)=a$ y
   $\dfrac{\partial}{\partial \beta}(\beta^T A\beta)=(A+A^T)\beta$.
4. **Subgradiente** del valor absoluto (para lasso), que se explica en su capítulo.
5. Desigualdad elemental $1+t\le e^{t}$ (para AdaBoost).

## Cómo compilar este libro

El libro está hecho con [Jupyter Book](https://jupyterbook.org). Desde la carpeta del
proyecto:

```bash
pip install "jupyter-book<2"
```

```bash
jupyter-book build libro
```

El resultado queda en `libro/_build/html/index.html`, que se abre con cualquier navegador.

## Declaratoria de uso de inteligencia artificial

Este libro fue realizado **con la asistencia de inteligencia artificial**. Se utilizó
Claude (Anthropic), a través de Claude Code, como herramienta de apoyo para:

- Organizar las demostraciones y las explicaciones de cada capítulo.
- Verificar cálculos numéricos (índices de Gini, ejemplos de lasso y AdaBoost) mediante
  código en Python y `scikit-learn`.
- Estructurar y compilar el libro con Jupyter Book.

El autor, Rubiel Velasquez, revisó el contenido y es responsable de su versión final.
