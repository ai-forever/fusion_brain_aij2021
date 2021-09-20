# Fusion Brain Challenge

В рамках данной задачи предлагается построить единую multitask-модель, которая бы успешно решала такие относящиеся к различным модальностям подзадачи, как **Code2code translation (С2С), Handwritten Text Recognition (HTR), Zero-shot object detection (zsOD), Visual Question Answering (VQA)** – и сумела бы побить интегральную метрику предложенного Организатором бейзлайна, а также метрики по каждой из подзадач.

Мы предоставляем [концепт единой модели](https://colab.research.google.com/drive/1YAkxWG0dRKPtqy9CZxFPvCNCCXvMGr65?usp=sharing%22%20target%3D%22_parent%22%3E%3Cimg%20src%3D%22https%3A%2F%2Fcolab.research.google.com%2Fassets%2Fcolab-badge.svg%22%20alt%3D%22Open%20In%20Colab), которая обучается на нескольких задачах, относящихся к различным модальностям (визуальной, аудио и текстовой). Концепт вдохновлен статьёй ["Pretrained Transformers as Universal Computations Engines"](https://arxiv.org/pdf/2103.05247.pdf) (```Lu et al., 2021```), в которой исследуется способность предобученных языковых моделей на базе архитектуры Трансформер формировать качественные репрезентации для произвольных последовательностей данных – обобщаться, таким образом, на другие модальности с минимальным дообучением. Основа предлагаемой в концепте архитектуры – предобученная языковая модель GPT-2; эксперименты проводятся как с "замороженной" моделью (Frozen Pretrained Transformer), так и с моделью, все слои которой дообучаются на трёх модальностях одновременно.  

Для того, чтобы представленная командой/участником архитектура считалась единой (multitask-модель), необходимо, чтобы общие для всех модальностей параметры (shared weights) составляли **не менее 30%** всех параметров модели. В противном случае модель считается решающей задач(у/и) по отдельности. 

Загрузка решений станет доступна с **04.10.2021**.

## Общий формат решения

От участников требуется создать zip-архив с обученной моделью и набором скриптов для совершения предсказания модели. Участник загружает этот архив на платформу проведения соревнования. Далее архив распаковывается в docker-контейнер. Для запуска решений подойдет любой образ, доступный для загрузки из DockerHub. При необходимости, вы можете подготовить свой образ, добавить в него необходимое ПО и библиотеки (см. [инструкцию по созданию docker-образов](https://github.com/sberbank-ai/no_fire_with_ai_aij2021/blob/main/sbercloud_instruction.md)); для использования его необходимо будет опубликовать на DockerHub.

В корне архива обязательно должен быть файл ```metadata.json``` следующего содержания:
```
{
    "image": "sberbank/fb-python",
    "entry_point": "python run.py $PATH_INPUT $PATH_OUTPUT"
}
```

Здесь ```image``` – поле с названием docker-образа, в котором будет запускаться решение, ```entry_point``` – команда, при помощи которой запускается решение. Для решения текущей директорией будет являться корень архива. Во время запуска в переменной окружения ```DATASETS_PATH``` расположен путь к актуальным открытым наборам данных, которые доступны из контейнера с решением.

Скрипт для инференса модели должен принимать в качестве аргумента путь к папке, где лежит контент, на котором необходимо сделать предсказание. Допустим, в качестве аргумента передаётся папка ```fusion_brain```. Названия подпапок ```fusion_brain``` соответствуют названиям подзадач, которые необходимо решить единой модели. Внутри каждой из подпапок (C2C, HTR, zsOD, VQA) лежит контент, необходимый для совершения предсказаний.

Структура данных выглядит следующим образом:

* fusion_brain
  * C2C 
    * requests.json
  * HTR
    * images 
  * zsOD
    * images
    * requests.json
  * VQA
    * images
    * questions.json

Единая модель должна сформировать предсказания для каждой из подзадач формата ```prediction_{TASK_NAME}.json```, то есть после инференса модели должно получиться 4 файла: ```prediction_C2C.json, prediction_HTR.json, prediction_zsOD.json, prediction_VQA.json```. Эти файлы должны располагаться в корне загруженной папки с решением.

После этого в контейнер подкладываются правильные ответы формата ```true_{TASK_NAME}.json``` и запускается скрипт подсчета метрик для каждой из подзадач. Финальная метрика считается как сумма метрик для каждой из подзадач (об этом ниже).

# Подзадача 1 - Code2code Translation

## Описание 
Задача перевода с одного языка программирования на другой, стандартно входит в обширный репертуар сферы ML4Code. На данный момент существует несколько различных вариантов решения – как в духе контролируемого обучения, при котором в качестве обучающего датасета используется параллельный корпус (базовая модель бенчмарка CodeXGLUE c CodeBERT в качестве кодировщика в архитектуре типа «кодировщик-декодировщик», ```Lu et al., 2021``` ), так и неконтролируемого, включающего предобучение кросс-лингвальной языковой модели на монолингвальных корпусах (```TransCoder, Lachaux et al., 2020```). 

Особую сложность представляет случай, когда язык-источник и целевой язык имеют различную типизацию. Наша задача относится именно к такому варианту: необходимо выполнить перевод с языка, имеющего статическую типизацию (Java), на язык с динамической типизацией (Python). На вход модели подаётся функция, написанная на языке Java, – модель должна выдать аналогичную функцию на языке Python. 

## Данные

**Train.** В качестве обучающего датасета предлагается использовать train (5,937 программистских задач) и val (845) части параллельного корпуса [AVATAR](https://github.com/wasiahmad/AVATAR), состоящего из пар аналогичных функций/программ, одна из которых написана на языке программирования Java, другая — на Python. В связи с тем, что в датасете содержится от 1 до 5 вариантов решения задач на обоих языках программирования, можно сформировать от 1 до 25 параллельных примеров на каждую задачу. Авторы датасета предлагают выбирать для каждого языка максимум 3 решения - таким образом, на одну проблему приходится максимум 9 обучающих примеров. Мы предлагаем использовать [датасет](https://dsworks.s3pd01.sbercloud.ru/aij2021/%D0%A12%D0%A1_translation/CodeToCodeTranslation_train_java_python.jsonl), который сформирован именно таким способом. 

Файл имеет формат jsonl с полями java и python:

```
{"java":"import java . util . Scanner ; \u00a0 public class A1437 { \u00a0 public static void main ( String [ ] args ) { Scanner in = new Scanner ( System . in ) ; int T = in . nextInt ( ) ; for ( int t = 0 ; t < T ; t ++ ) { int L = in . nextInt ( ) ; int R = in . nextInt ( ) ; boolean possible = R < 2 * L ; System . out . println ( possible ? \" YES \" : \" NO \" ) ; } } \u00a0 }\n","python":"t = int ( input ( ) ) NEW_LINE ans = [ ] NEW_LINE for i in range ( t ) : l , r = [ int ( x ) for x in input ( ) . split ( ) ] NEW_LINE if ( 2 * l ) > r : NEW_LINE INDENT ans . append ( \" YES \" ) else : NEW_LINE ans . append ( \" NO \" ) NEW_LINE DEDENT for j in ans : print ( j ) NEW_LINE\n"}
```

Для создания обучающего параллельного корпуса также можно использовать [CodeNet](https://github.com/IBM/Project_CodeNet), в котором содержатся решения 4,000 программистских задач на С++, С, Python и Java, извлеченных с сайтов AtCoder (в датасете AVATAR используются решения с этого ресурса по части задач) и AIZU Online Judge. Для удобства участников мы предоставляем [архив](https://dsworks.s3pd01.sbercloud.ru/aij2021/%D0%A12%D0%A1_translation/CodeNet_accepted_java_python.tar.gz), в котором содержатся решения с CodeNet на языках Java и Python, разбитые по задачам. Стоит, однако, учитывать, что решения одной программистской задачи на разных языках являются, как минимум, клонами 4 типа (сохранение семантики кода при широкой вариативности синтаксиса), но не являются гарантированно идентичными друг другу с поправкой на различия в языках (буквальным переводом).   

**Test public.** Публичный лидерборд формируется по результатам проверки предсказаний моделей на тестовой выборке (1,693) датасета AVATAR.

**Test private.** Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке. 

## Метрика качества

**CodeBLEU** - метрика, предложенная в ```Ren et al., 2020``` и призванная приспособить метрику BLEU, которая используется для оценки качества перевода на естественном языке, к оценке перевода исходного кода; это достигается благодаря тому, что сравнение n-граммов дополняется сравнением соответствующих деревьев абстрактного синтаксиса эталонного кода и переведенного (таким образом, учитывается информация на уровне синтаксиса кода), а также сопоставляются "потоки данных" (data-flow, информация на уровне семантики кода).

CodeBLEU представляет собой взвешенную комбинацию четырех компонент:

![image](https://latex.codecogs.com/svg.image?\textrm{CodeBLEU}&space;=&space;\alpha&space;\cdot&space;\textrm{BLEU}&space;&plus;&space;\beta&space;\cdot&space;\textrm{BLEU}_{weight}&space;&plus;&space;\gamma&space;\cdot&space;\textrm{Match}_{ast}&space;&plus;&space;\delta&space;\cdot&space;\textrm{Match}_{df},)

где BLEU - стандартная метрика BLEU ```Papineni et al., 2002```, BLEU<sub>weight</sub> - взвешенное сопоставление n-грамм (токены различаются по важности - и совпадение определённых токенов переведенных и "золотых" функций имеет больший вес), Match<sub>ast</sub> – метрика соответствия деревьев абстрактного синтаксиса переведенного кода и эталонного, Match<sub>df</sub> отражает сходство "потоков данных" функций-гипотез и верных функций.

Остановимся подробнее на каждой из компонент метрики:

* BLEU основана на подсчете n-грамм, которые встретились и в переводе, и в референсной последовательности; рассчитывается она следующим образом:

![image](https://latex.codecogs.com/svg.image?\textrm{BLEU}&space;=&space;\textrm{BP}&space;\cdot&space;\textrm{exp}&space;(\sum_{n=1}^{N}&space;w_{n}&space;\log&space;p_{n}),)

где BP - штраф за слишком короткие варианты перевода, который считатается как отношение количества токенов в переводе, предложенном моделью, к количеству токенов в эталонной последовательности; вторая часть выражения – среднее геометрическое значений модифицированной точности n-грамм:

![image](https://latex.codecogs.com/svg.image?\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{n\text{-}gram&space;\in&space;C}&space;Count_{clip}&space;\textrm{(n-gram)}}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{n\text{-}gram\textquotesingle&space;\in&space;C\textquotesingle}&space;Count&space;\textrm{(n-gram\textquotesingle)}})

для n-грамм длиной от 1 до N, умноженных на соответствующие положительные веса w<sub>n</sub>, в сумме дающие 1.

*  В отличие от стандартной BLEU, в формулу расчета точности сопадения n-грамм для метрики BLEU<sub>weight</sub> включается весовой коэффициент (![image](https://latex.codecogs.com/svg.image?\mu_{n}^{i})), значение которого больше для ключевых слов языка программирования, чем для других токенов:

![image](https://latex.codecogs.com/svg.image?\textrm{p}_{n}&space;=&space;\frac{\sum_{C&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count_{clip}&space;(C(i,&space;i&plus;n))}{\sum_{C\textquotesingle&space;\in&space;Candidates}&space;\sum_{i=1}^{l}&space;\mu_{n}^{i}&space;Count&space;(C\textquotesingle(i,&space;i&plus;n))},)

где C(i, i+n) - n-грамм, начинающийся в позиции i и заканчивающийся в позиции i+n;  Count<sub>clip</sub>, как и в случае со стандартной BLEU, - максимальное количество n-грамм, встречающихся как в переведенном коде, так и в наборе эталонных решений.
Список ключевых слов заранее определяется для конкретного языка программирования.

* Синтаксическая структура исходного кода может быть представлена в виде дерева абстрактного синтаксиса (ДАС) - переведенные и эталонные функции, таким образом, можно сравнивать на уровне поддеревьев, полученных с помощью ДАС-парсера. Поскольку нас интересует синтаксическая информация, листья ДАС, в которых находятся переменные функции, не учитываются. Match<sub>ast</sub> рассчитывается по следующей формуле:

![image](https://latex.codecogs.com/svg.image?\textrm{Match}_{ast}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{T}_{cand})}{\textrm{Count}(\textrm{T}_{ref})},)

гдe Count(T<sub>ref</sub>) - общее количество поддеревьев референсного кода, Count<sub>clip</sub>(T<sub>cand</sub>) - количество поддеревьев переведенного кода, совпавших с поддеревьями эталонных функций. Данная метрика позволяет оценить качество получившегося кода с точки зрения синтаксиса.  

* Сравнение переведенного кода и референсного на уровне семантики происходит с использованием "потоков данных" (data flow ```Guo et al., 2020``` ) - представлений исходного кода в виде графа, вершины которого - переменные, а грани обозначают своего рода "генетические" отношения между вершинами (выражают информацию о том, откуда берется значение каждой переменной). Формула расчета метрики Match<sub>df</sub> имеет следующий вид:

![image](https://latex.codecogs.com/svg.image?\textrm{Match}_{df}&space;=&space;\frac{\textrm{Count}_{clip}(\textrm{DF}_{cand})}{\textrm{Count}(\textrm{DF}_{ref})},)

гдe Count(DF<sub>ref</sub>) - общее количество "потоков данных" референсного кода, Count<sub>clip</sub>(DF<sub>cand</sub>) - количество "потоков данных" переведенного кода, совпавших с эталонными.

## Формат решения

От участников требуется создать архив с обученной моделью и набором скриптов для совершения предсказания модели. Участник загружает этот архив на платформу проведения соревнования. Далее архив распаковывается в докер-контейнер, а система подкладывает в пространство контейнера данные для совершения предсказания. Эти данные включают:

* Файл ```requests.json```. Это словарь формата ```{ "0": "import java . util . Scanner ; ..." ,  ... }```. Ключами являются индексы примеров, значениями – строки функций/программ на языке Java, которые необходимо перевести на язык программирования Python. 

Модель участника должна перевести все примеры из файла requests.json и сгенерировать файл ```prediction_С2С.json```. Это словарь формата ```{ "0": "def find ( x , par ) : NEW_LINE INDENT if par [ x ] == x : ..." ,  ... }```. Ключами являются индексы примеров, значениями – переводы функций/программ на язык Python. Обратите внимание, что, поскольку в Python используются отступы для идентификации логических блоков в коде, в строке перевода на Python присутствуют специальные токены ```INDENT```, ```DEDENT```. 

После проведения инференса скрипт подсчета метрик сравнивает файлы ```prediction_С2С.json``` и ```true_С2С.json```, а затем выводит финальное значение метрики CodeBLEU.

# Подзадача 2 - Handwritten Text Recognition

## Описание 

Перед участниками ставится задача распознавания рукописного текста на изображении. На вход модели подается изображение с рукописным текстом на русском или английском языках. Модель должна выдать в качестве ответа текстовую строку, соответствующую контенту изображения, - в данном случае строку «последовал»:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/htr_posledoval.png)

## Данные

**Train.** Для обучения предоставляется [набор данных](https://dsworks.s3pd01.sbercloud.ru/aij2021/htr/train.zip), состоящий из двух разных датасетов. Первый - собранный нами вручную датасет школьных тетрадей. Изображения в нем являются отдельными словами в тексте, написанном на странице тетради. Вторая часть датасета состоит из известного набора данных IAM. Это набор рукописных слов на английском языке.

**Test public.** Публичный лидерборд так же рассчитывается на части датасетов тетрадей и IAM.

**Test private.** Приватный тестовый датасет скрыт от участников. Это тоже набор данных для распознавания текста, в похожем на обучающий датасет формате. Мы, однако, не предоставляем информации о деталях датасета.

## Метрика качества

В качестве основной метрики для оценки решений участников используется формула  **1 - CER**, где CER - это метрика character error rate. Считается она следующим образом:

![image](https://latex.codecogs.com/svg.image?\text{CER}&space;=&space;\frac{\sum_{i=1}^n&space;\text{dist}_{c}(pred_i,&space;true_i)}{\sum_{i=1}^n&space;\text{len}_{c}(true_i)})

Здесь dist<sub>c</sub>  - это расстояние Левенштейна, посчитанное для токенов-символов (включая пробелы), len<sub>c</sub> - длина строки в символах.

Метрика **1 - CER** изменяется от 0 до 1, где 0 – наихудшее значение, 1 - наилучшее.

## Формат решения

От участников требуется создать архив с обученной моделью и набор скриптов для совершения предсказания модели. Участник загружает этот архив на платформу проведения соревнования. Далее архив распаковывается в докер-контейнер, а система подкладывает в пространство контейнера данные для совершения предсказания. Эти данные включают:

* Папка ```images```. Это набор изображений, по которым нужно сделать предсказания. Внутри лежат файлы формата ```0.jpg, 1.jpg ...```. Каждый файл содержит графические изображения символов, которые необходимо перевести в текстовые символы (текстовые строки). 

Модель участника должна сделать предсказания на всех изображениях из папки images, и сгенерировать файл ```prediction_HTR.json```. Это словарь формата ```{"0.txt": "<предсказанный текст на изображении>" ,  "1.txt": "<предсказанный текст на изображении>" , ... }```. Ключами являются соответствующие названия файлов из папки images, значениями - предсказанные строки на соответствующих изображениях. Если по каким-то причинам на файле с изображением name.png не было сделано предсказания, то есть в ключах словаря ```prediction_HTR.json``` отсутствует ключ ```"name.png"```, то перевод заполняется пустой строкой ```""```.

После проведения инференса скрипт подсчета метрик сравнивает файлы ```prediction_HTR.json``` и ```true_HTR.json```, а затем выводит финальное значение метрики по данной задаче.

Файл ```true_HTR.json``` имеет формат ```{ "0.txt": "<правильный текст на изображении>" ,  "1.txt": "<правильный текст на изображении>" , ... }```. Ключами являются соответствующие названия файлов из папки images, значениями - правильный перевод строки на соответствующем изображении. 

# Подзадача 3 - Zero-shot Object Detection

## Описание 

* Необходимо определить класс объекта, изображенного на фотографии (или классы, если их несколько). Например, на фотографии могут оказаться такие сущности/объекты, как "человек", "машина", "яблоко".

* Одновременно с этим нужно определить местоположение и масштаб каждого из объектов на фотографии. Местоположение объекта описывается так называемым bounding box (ограничивающая рамка, bbox). Это прямоугольник, который наиболее аккуратно нарисован вокруг рассматриваемого объекта. Положение прямоугольника задается 4-мя числами – X, Y, W, H:

  * X – горизонтальная координата верхнего левого угла
  * Y – вертикальная координата верхнего левого угла
  * W – ширина прямоугольника
  * H – высота прямоугольника

Предсказаниями модели должны быть координаты bbox и метка класса для каждого объекта на фотографии. Пример результата работы модели object detection представлен на следующем изображении: 

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/od.png)

В рамках нашего соревнования задача сформулирована как zero-shot object detection. Zero-shot в описании задачи означает, что модели нужно сделать предсказание на данных, совершенно отличных от обучающего набора. Стандартная модель object detection во время предсказания должна выдавать один класс из ограниченного набора, четко заданного во время обучения модели. Zero-shot модель должна уметь детектировать классы, которые не встречались в обучающей выборке.

Еще одна особенность предлагаемой задачи заключается в том, что множество классов для каждого изображения передается в формате запроса. Запрос может содержать классы как на русском, так и на английском языках.

Во время стадии предсказания модели на вход подаются две сущности: первая - изображение, вторая - запрос на естественном языке. Формат запроса - текстовая строка, которая содержит список лейблов, среди которых нужно произвести поиск. Пример: «собака, велосипед, машина, торт, самолет». Запрос содержит и верные метки – объекты, которые действительно присутствуют на изображении, - и некоторое количество неправильных меток. Их объединение является общим пространством поиска для модели. Результатом предсказания модели должен стать список предсказанных меток вместе с соответствующими координатами bounding box.

## Данные

**Train.** Для обучения предлагает использовать популярный датасет MS-COCO.

[Изображения](http://images.cocodataset.org/zips/train2017.zip)  
[Аннотации](http://images.cocodataset.org/annotations/annotations_trainval2017.zip)

**Test public.**  Публичный тестовый датасет сформирован из части датасета [VisualGenome](https://visualgenome.org/api/v0/api_home.html); набор классов в нем скрыт от участников. 

**Test private.** Приватный тестовый датасет скрыт от участников, так же как и набор классов в нем.

## Метрика качества

Для оценки качества будет использоваться метрика **mean average precision** (mAP). Для вычисления mAP сначала нужно выбрать IoU (intersection over union). Это метрика, которая оценивает качество совпадения предсказанного bbox и эталонного. Она вычисляется как отношение площади пересечения к площади объединения этих двух bbox:

![image](https://latex.codecogs.com/svg.image?\textrm{IoU}&space;=&space;\frac{&space;\textrm{Intersection}}{&space;\textrm{Union}})

IoU для каждой пары (prediction/true) принимает значение от 0 до 1. В качестве порога отсечения по IoU используется значение 0.5, то есть все предсказанные bbox, значение IoU для которых меньше 0.5, считаются ложными прогнозами.

Для каждого класса с выбранным значением IoU строится кривая Precision-Recall, затем вычисляется площадь под этой кривой. Это значение усредняется по всем классам, и получается финальное значение метрики mAP.

## Формат решения

От участников требуется создать архив с обученной моделью и набор скриптов для совершения предсказания модели. Участник загружает этот архив на платформу проведения соревнования. Далее архив распаковывается в докер-контейнер, а система подкладывает в пространство контейнера данные для совершения предсказания. Эти данные включают:

* Папка ```images```. Это набор изображений, по которым нужно сделать предсказания. Внутри лежат файлы формата ```0.jpg, 1.jpg ...```.
* Файл ```requests.json```. Это словарь формата ```{ "0.jpg": ["дерево", "часы", "книга"] ,  ... }``` . Ключами являются соответствующие названия файлов из папки images, значениями - список классов, которые нужно детектировать на соответствующем изображении (запрос). Как говорилось ранее, список классов может быть как на русском, так и на английском языке. Запрос, таким образом, содержит список лейблов, среди которых нужно произвести поиск (например, «собака, велосипед, машина, торт, самолет»). **Запрос содержит и правильные метки, относящиеся к объектам, которые действительно находятся на изображении, и некоторое количество неправильных лейблов (соответствующих объектов нет на изображении).**

Модель участника должна сделать предсказания на всех изображениях из папки images, и сгенерировать файл ```prediction_zsOD.json```. Это словарь формата ```{"0.jpg": [["dog", 0.5, 473.07, 395.93, 38.65, 28.67], ["cat", 0.6, 0.0, 101.15, 452.3, 319.43]], ...```. Ключами являются названия файлов из папки images, значениями - предсказания модели для соответствующих изображений. Эти предсказания должны содержать обнаруженные классы вместе с координатами ограничивающих рамок для данного изображения: по ключу "название файла с изображением" - список предсказанных на нём объектов. Формат одного элемента в списке: ```["dog", 0.5, 473.07, 395.93, 38.65, 28.67]``` (шесть элементов, разделенных запятыми). Первый элемент - название класса, второй - score модели, затем идут 4 координаты bbox в формате ```xywh```. Во вложенном списке может содержаться неограниченное количество элементов – это все объекты, которые предсказала модель для данного изображения.

Далее система сравнивает файл с предсказаниями с файлом ```true_zsOD.json```, содержащим правильные ответы, и выводит итоговую метрику mAP.

# Подзадача 4 - Visual Question Answering

## Описание 

Необходимо ответить текстом на вопрос по представленному изображению. На вход модели подаются изображение и текстовый вопрос, связанный с этим изображением, - на выходе модель должна выдать ответ на заданный вопрос в виде текста. Например, ответом на вопрос "Из чего сделаны усы?" в данном случае может быть слово "бананы":

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/bananas.png)

Особенность задачи заключается в том, что вопросы не гомогенны: подходящий ответ может как состоять из нескольких слов, так и быть односложным (ответ типа "да/нет") или представлять собой число. Подразумевается, что на один вопрос необходимо дать только один ответ. 

Вопросы могут быть как на английском языке, так и на русском. Предполагается, что язык ответа соответствует языку вопроса, кроме тех случаев, когда вопрос касается текста на изображении (например, "Что написано на футболке?") – в этом случае ответ должен быть на том же языке, на котором написан текст. 

## Данные

**Train.** В качестве обучающей выборки предлагается использовать часть train датасета [VQA v2](https://visualqa.org/download.html): в состав входят вопросы на английском языке (файл *Training questions 2017 v2.0*), изображения из датасета COCO, по которым эти вопросы заданы (файл *Training images*), а также аннотации - ответы на вопросы (файл *Training annotations 2017 v2.0*).

**Test public.**  Публичный тестовый датасет состоит из вопросов как на русском, так и на английском языках: русскоязычная часть представляет собой переведённые 10 тыс. первых семплов из валидационной части датасета VQA v2, английская - другие 10 тыс. примеров из того же датасета, взятых в оригинальном виде. 

**Test private.** Приватный тестовый датасет скрыт от участников. Его формат аналогичен публичной тестовой выборке, в нем присутствуют вопросы на русском и английском языках.

## Метрика качества

Для оценки качества предсказания будет использоваться метрика **accuracy**. Эта метрика показывает долю точных совпадений среди пар предсказанных и истинных ответов, то есть отражает отношение числа совпавших ответов (когда модель участника предсказала такой же ответ, как истинный) к общему числу ответов. Эта метрика изменяется от 0 до 1, где 0 - наихудшее значение, 1 - наилучшее:

![image](https://latex.codecogs.com/svg.image?\textrm{accuracy}&space;=\frac{&space;\textrm{True&space;answers}}{&space;\textrm{All&space;answers}})

## Формат решения

От участников требуется создать архив с обученной моделью и набором скриптов для совершения предсказания модели. Участник загружает этот архив на платформу проведения соревнования. Далее архив распаковывается в докер-контейнер, а система подкладывает в пространство контейнера данные для совершения предсказания. Эти данные включают:

* Папка ```images```. Это набор изображений, к которым относятся вопросы. Внутри лежат файлы формата ```0.jpg, 1.jpg ...```.
* Файл ```questions.json```. Это словарь формата ```{ "0": {"image_id": "1.jpg", "question": "Куда он смотрит?"} ,  ... }```. Ключами являются индексы примеров, значениями – словарь с полями "image_id" (значение – название файла из папки images) и "question" (значение – текст вопроса по соответствующему изображению). Вопросы могут быть заданы как на английском языке, так и на русском. 

Модель участника должна сделать предсказания по всем вопросам и сгенерировать файл ```prediction_VQA.json```. Это словарь формата ```{ "0": "вниз" ,  ... }```. Ключами являются индексы примеров, значениями – предсказанные моделью ответы на соответствующие вопросы. 

После проведения инференса скрипт подсчета метрик сравнивает файлы ```prediction_VQA.json``` и ```true_VQA.json```, а затем выводит финальное значение метрики accuracy. 

# Интегральная метрика

Итоговая оценка складывается из оценок по подзадачам:

![image](https://latex.codecogs.com/svg.image?\textrm{S}&space;=&space;\textrm{S}_{1}&space;&plus;&space;\textrm{S}_{2}&space;&plus;&space;\textrm{S}_{3}&space;&plus;&space;\textrm{S}_{4},)

где S – итоговая оценка участника, S<sub>1</sub> – оценка по подзадаче Code2code Translation, S<sub>2</sub> – оценка по подзадаче Handwritten Text Recognition, S<sub>3</sub> – оценка по подзадаче Zero-shot Object Detection, S<sub>4</sub> – оценка по подзадаче Visual Question Answering. 

Оценка по каждой из подзадач принимает значения от 0 до 1 (исключение составляет метрика CodeBLEU, которая используется для оценки Code2code translation и может принимать значения в диапазоне от 0 до 100 – с целью нормализации метрика умножается на коэффициент 0.01) – таким образом, минимальное значение итоговой оценки составляет 0, максимальное – 4. Расчет оценки по каждой подзадаче округляется до третьего знака после запятой. По значениям итоговой оценки формируется лидерборд по задаче Fusion Brain Challenge. 

# Призовой фонд

Размер возможного выигрыша по-разному рассчитывается для единой mulitask-модели (в которой доля общих параметров, используемых для решения всех подзадач, составляет не менее 30% от всех параметров) и unitask-модели (решающей одну подзадачу). 

## Multitask-модель 

Для каждого призового места есть фиксированная сумма выигрыша (FIX). Размер бонуса зависит от итовой оценки призеров, но не может превышать разницы между максимальным значением (MAX) и фиксированным. В каждом задании необходимо побить бейзлайн. В каждом задании необходимо превзойти метрики бейзлайна по каждой подзадаче, а интегральная метрика должна как минимум на 0.15 (δ) превосходить интегральную метрику бейзлайна. 

Формула расчета денежного приза имеет следующий вид:

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/prize.png)

, где S – итоговая оценка участника, S<sub>baseline</sub> – итоговая оценка бейзлайна, δ = 0.15 – минимальное значение, на которое должна быть превышена итоговая оценка бейзлайна, коэффициент α зависит от места в лидерборде (топ-3 решения) и вычисляется следующим образом:

![image](https://latex.codecogs.com/svg.image?\alpha_{place}&space;=&space;\frac{\textrm{MAX}_{place}&space;-&space;\textrm{FIX}_{place}}{2.3&space;-&space;(\textrm{S}_{baseline}&space;&plus;&space;\delta)},)

где α<sub>place</sub> – коэффициент для первого, второго и третьего мест в лидерборде (индекс place обозначает место в итоговом лидерборде). MAX<sub>place</sub> – максимальный размер вознаграждения для топ3-решений в лидерборде при S ≥ 2.3 (MAX<sub>1</sub> = 3 млн руб., MAX<sub>2</sub> = 1.5 млн руб., MAX<sub>3</sub> = 0.8 млн руб.). FIX<sub>place</sub> – фиксированная сумма выигрыша для топ-решений в лидерборде при (S<sub>baseline</sub> + δ) ≤ S < 2.3 (FIX<sub>1</sub> = 1, FIX<sub>2</sub> = 0.5, FIX<sub>3</sub> = 0.2). Коэффициент α<sub>place</sub> рассчитывается только для случаев, при которых S<sub>baseline</sub> + δ ≤ S < 2.3 (см. таблицу выше).

![image](https://dsworks.s3pd01.sbercloud.ru/aij2021/misc/prize_plot.png)

**Первое место:** от 1 000 000 рублей до 3 000 000 рублей (в зависимости от качества представленного участником решения)  
**Второе место:** от 500 000 рублей до 1 500 000 рублей (в зависимости от качества представленного участником решения)  
**Третье место:** от 200 000 рублей до 800 000 рублей (в зависимости от качества представленного участником решения)  

## Unitask-модель

* 300 000 рублей за первое место в подзадаче Code2code Translation
* 300 000 рублей за первое место в подзадаче Handwritten Text Recognition
* 300 000 рублей за первое место в подзадаче Zero-shot Object Detection
* 300 000 рублей за первое место в подзадаче Visual Question Answering

[Правила проведения Конкурса "Artificial Intelligence Journey Contest"](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/rules.pdf/download)  
[Пользовательское соглашение](https://api.dsworks.ru/dsworks-transfer/api/v1/public/file/terms_of_use.pdf/download)
