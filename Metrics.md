# Метрика качества

## Подзадача 1 - Code2code Translation

**CodeBLEU** - метрика, предложенная в ```Ren et al., 2020``` и призванная приспособить метрику BLEU, которая используется для оценки качества перевода на естественном языке, к оценке перевода исходного кода; это достигается благодаря тому, что сравнение n-граммов дополняется сравнением соответствующих деревьев абстрактного синтаксиса эталонного кода и переведенного (таким образом, учитывается информация на уровне синтаксиса кода), а также сопоставляются "потоки данных" (data-flow, информация на уровне семантики кода).

CodeBLEU представляет собой взвешенную комбинацию четырех компонент:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{CodeBLEU}&space;=&space;\alpha&space;\cdot&space;\textrm{BLEU}&space;&plus;&space;\beta&space;\cdot&space;\textrm{BLEU}_{weight}&space;&plus;&space;\gamma&space;\cdot&space;\textrm{Match}_{ast}&space;&plus;&space;\delta&space;\cdot&space;\textrm{Match}_{df},)

где BLEU - стандартная метрика BLEU ```Papineni et al., 2002```, BLEU<sub>weight</sub> - взвешенное сопоставление n-грамм (токены различаются по важности - и совпадение определённых токенов переведенных и "золотых" функций имеет больший вес), Match<sub>ast</sub> – метрика соответствия деревьев абстрактного синтаксиса переведенного кода и эталонного, Match<sub>df</sub> отражает сходство "потоков данных" функций-гипотез и верных функций.

Остановимся подробнее на каждой из компонент метрики:

* BLEU основана на подсчете n-грамм, которые встретились и в переводе, и в референсной последовательности; рассчитывается она следующим образом:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{BLEU}&space;=&space;\textrm{BP}&space;\cdot&space;\textrm{exp}&space;(\sum_{n=1}^{N}&space;w_{n}&space;\log&space;p_{n}),)

где BP - штраф за слишком короткие варианты перевода, который считается как отношение количества токенов в переводе, предложенном моделью, к количеству токенов в эталонной последовательности; вторая часть выражения – среднее геометрическое значений модифицированной точности n-грамм:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{n\text{-}gram&space;\in&space;C}&space;Count_{clip}&space;\textrm{(n-gram)}}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{n\text{-}gram\textquotesingle&space;\in&space;C\textquotesingle}&space;Count&space;\textrm{(n-gram\textquotesingle)}})

для n-грамм длиной от 1 до N, умноженных на соответствующие положительные веса w<sub>n</sub>, в сумме дающие 1.

*  В отличие от стандартной BLEU, в формулу расчета точности совпадения n-грамм для метрики BLEU<sub>weight</sub> включается весовой коэффициент (![image](https://latex.codecogs.com/svg.image?\mu_{n}^{i})), значение которого больше для ключевых слов языка программирования, чем для других токенов:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count_{clip}&space;(C(i,&space;i&plus;n))}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count&space;(C\textquotesingle(i,&space;i&plus;n))},)

где C(i, i+n) - n-грамм, начинающийся в позиции i и заканчивающийся в позиции i+n; Count<sub>clip</sub>, как и в случае со стандартной BLEU, - максимальное количество n-грамм, встречающихся как в переведенном коде, так и в наборе эталонных решений.
Список ключевых слов заранее определяется для конкретного языка программирования.

* Синтаксическая структура исходного кода может быть представлена в виде дерева абстрактного синтаксиса (ДАС) - переведенные и эталонные функции, таким образом, можно сравнивать на уровне поддеревьев, полученных с помощью ДАС-парсера. Поскольку нас интересует синтаксическая информация, листья ДАС, в которых находятся переменные функции, не учитываются. Match<sub>ast</sub> рассчитывается по следующей формуле:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Match}_{ast}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{T}_{cand})}{\textrm{Count}(\textrm{T}_{ref})},)

где Count(T<sub>ref</sub>) - общее количество поддеревьев референсного кода, Count<sub>clip</sub>(T<sub>cand</sub>) - количество поддеревьев переведенного кода, совпавших с поддеревьями эталонных функций. Данная метрика позволяет оценить качество получившегося кода с точки зрения синтаксиса.  

* Сравнение переведенного кода и референсного на уровне семантики происходит с использованием "потоков данных" (data flow ```Guo et al., 2020```) - представлений исходного кода в виде графа, вершины которого — переменные, а грани обозначают своего рода "генетические" отношения между вершинами (выражают информацию о том, откуда берется значение каждой переменной). Формула расчета метрики Match<sub>df</sub> имеет следующий вид:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Match}_{df}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{DF}_{cand})}{\textrm{Count}(\textrm{DF}_{ref})},)

где Count(DF<sub>ref</sub>) - общее количество "потоков данных" референсного кода, Count<sub>clip</sub>(DF<sub>cand</sub>) - количество "потоков данных" переведенного кода, совпавших с эталонными.

## Подзадача 2 - Handwritten Text Recognition

В качестве основной метрики для оценки решений участников используется метрика **String Accuracy** - отношение количества полностью совпавших транскрибаций строк к количеству всех строк в выборке. Считается она следующим образом:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\text{String&space;Accuracy}&space;=&space;\frac{\sum_{i=1}^n&space;[\text{pred}_i&space;=&space;\text{true}_i]}{n})

Здесь n - размер тестовой выборки, pred<sub>i</sub> – это строка из символов, которую распознала модель на i-ом изображении в выборке, а true<sub>i</sub> - это правильный перевод i-ого изображения, произведенный аннотатором, [•] - скобка Айверсона:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/Iverson.png)

Метрика String Accuracy изменяется от 0 до 1, где 0 – наихудшее значение, 1 - наилучшее.

## Подзадача 3 - Zero-shot Object Detection

Для оценки качества будет использоваться метрика **F1-score**:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{F1}=&space;2&space;\cdot&space;\frac{\text{Recall}\cdot\text{Precision}}{\text{Recall}&space;&plus;&space;\text{Precision}})

F1-score вычисляется на основе значений Precision (точности) и Recall (полноты), которые, в свою очередь, зависят от набора статистик по прогнозам – true positive (TP, истинно-положительный результат), false positive (FP, ложно-положительный результат) и false negative (FN, ложно-отрицательный результат):

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Precision}=&space;\frac{\text{True\&space;Positive}}{\text{True\&space;Positive}&space;&plus;&space;\text{False\&space;Positive}},)

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Recall}=&space;\frac{\text{True\&space;Positive}}{\text{True\&space;Positive}&space;&plus;&space;\text{False\&space;Negative}})

Правила, по которым прогноз модели относится к одному из типов, следующие: 

* Если данный класс из запроса отсутствует в правильной разметке (то есть является негативным примером), но модель участника сделала для него предсказание — предсказание оценивается как *FP*
* Если данный класс из запроса присутствует в правильной разметке (то есть является положительным примером):
    * модель участника не сделала для него предсказание, то есть передала пустой список, или же количество предсказанных ограничивающих рамок меньше количества верных для данного класса, — непредсказанные ограничиващие рамки оцениваются как *FN*
    * для каждого bbox данного класса из предсказания (класс может иметь несколько соответствующих bbox на изображении):
        * пересечение предсказанного bbox хотя бы с одним из правильных bbox для данного класса по IoU > 0.5 – предсказание оценивается как *TP*
        * пересечение предсказанного bbox с каждым из правильных bbox для данного класса по IoU < 0.5 – предсказание оценивается как *FP*

IoU – это метрика, которая оценивает степень пересечения между двумя ограничивающими рамками. Она вычисляется как отношение площади пересечения к площади объединения этих двух bbox:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{IoU}=&space;\frac{\text{Intersection}}{\text{Union}})

IoU для каждой пары (prediction/true) принимает значение от 0 до 1. В качестве порога отсечения по IoU используется значение 0.5, то есть все предсказанные bbox, значение IoU для которых меньше 0.5, считаются ложными прогнозами.

Метрика F1-score изменяется от 0 до 1, где 0 – наихудшее значение, 1 – наилучшее.

## Подзадача 4 - Visual Question Answering

Для оценки качества предсказания будет использоваться метрика **Accuracy**. Эта метрика показывает долю точных совпадений среди пар предсказанных и истинных ответов, то есть отражает отношение числа совпавших ответов (когда модель участника предсказала такой же ответ, как истинный) к общему числу ответов. Эта метрика изменяется от 0 до 1, где 0 – наихудшее значение, 1 – наилучшее:

![image](https://latex.codecogs.com/svg.image?\color{Blue}\textrm{Accuracy}&space;=\frac{&space;\textrm{True&space;answers}}{&space;\textrm{All&space;answers}})
