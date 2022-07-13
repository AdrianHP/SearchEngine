# Search Engine

El proyecto provee de una infraestructura flexible con el objetivo de crear un motor de búsqueda. Además expone una API con la cual se pueden hacer las consultas a este.

## Corpus

El corpus sobre el cual se hace el análisis se encuentra en *search_logic/corpus*. Los archivos serán leídos como texto plano.

## Representación

Los documentos y la query son representados como un diccionario. Las llaves básicas son:

- tokens: Los tokens del texto
- text: El texto crudo del documento
- dir: La dirección del documento

## Pipeline

El flujo de los modelos se maneja mediante Pipes los cuales pasan un diccionario que va siendo anotado
en cada fase pudiendo ser utilizados los datos de fases anteriores.

### Tokenización

Fase general de procesamiento de texto para todos los modelos. Permite:

- Tokenización del texto
- Eliminación de las stopwords
- Stemming

Todos estos procesos modifican la llave *tokens* del documento asociado

### Vectorial

Fase de procesamiento de los documentos para su funcionamiento en el modelo vectorial. Permite:

- Calcular idf.
- Convertir documentos a vectores calculando los pesos por ntf * idf.
- Convertir query a vector con factor de suavizado.
- Calcular la similitud entre los documentos y a query y devolver los resultados rankeados fitrados por un umbral para la similitud.

### Rank SVM

Fase de procesamiento que en la cual se hace el entrenamiento del clasificador SVM. Su procesamiento depende en parte del modelo vectorial.

- Entrenamiento del clasificador SVM
- Cálculo de ranking de documentos dado una query

### Retroalimentación

La retroalimentación está dada explícita por el usuario. Se puede crear el modelo realizando un sembrado automático de
las relaciones query-documento del corpus las cuales sirven para, dada la consulta, encontrar consultas similares a
estas y devolver los documentos relevantes y no relevantes para utilizar en el algoritmo de Rocchio. Se utiliza un
valor preestablecido de 0.25 para separar las consultas similares más significativas y además solo se escogen las 5
primeras consultas con mayor similitud, incluyendo a la misma consulta para poder tomar en cuenta la retroalimentación
dada por el usuario.

El algoritmo de Rocchio utiliza toda la información acerca de los documentos relevantes y los no relevantes para una
determinada consulta(y la de consultas similares a esta en nuestro caso) para acercar más a dicha consulta a la zona
de los documentos relevantes y alejarla de los documentos no relevantes. Se utilizan los valores usuales de alpha,
beta y ro los cuales son 1, 0.75 y 0.1 respectivamente.

### Expansión de consulta

La expansión de consulta se hace mediante la construcción de una matriz de correlación de bigramas en el corpus, observando la última palabra de la consulta y devolviendo un rankking de las que más se repiten.

## API

Expone una API a la cual se le pueden hacer las siguientes conusltas:

- *query/?query=QUERY*: Devuelve una lista ordenada por similitud con la query. Vea el modelo **QueryResult**
- *document/?document\_dir=DOCUMENTDIR*: Devuelve el contenido del documento
- *expand/?query=QUERY*: Devuelve una lista de las posibles expansiones de query a realizar.
- *feedback/*: Marca como relevantes o no relevantes a la query los documentos. Vea el modelo **FeedbackModel**

## Visual

Para el visual se creó un proyecto de streamlit. Correr  `streamlit run visual.py`

## Test

Correr `pytest` en la consola

## Evaluación

Para evaluar el modelo se creó el script `eval_model.py`, con el cual se prueba el F1, Precisión y Recobrado de los modelos creados.

El script es modificable para que use las estadísticas de previas corridas. Ver comentarios en script.

## Cómo expandir el proyecto?

Para expandir el proyecto para agregarle diferentes features se debe realizar usando la misma asquitectura de Pipelines. El proyecto posee clases que simplifican el montaje de nuevos modelos.

- InformationRetreivalModel: Clase base que debe implementar los modelos de recuperación nuevos.
- FeedbackManager: Clase base para la implementación de algoritmos de feedback
- QueryExpansionManager: Clase base para la implementación de algoritmos de expansión de consultas

Se puede formar un nuevo modelo realizando los pipes adecuados e inyectándole al contexto implementaciones de las clases anteriores, por ejemplo.

## Consideraciones

- Se puede hacer los modelos Booleano y Probabilístico sobre la misma infraestructura e incluso combinarlos.
- Los corpus deben estar en los formatos dados en test (Cranfield y Med).
