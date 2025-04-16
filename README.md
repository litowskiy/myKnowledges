# My Knowledges

# Основные типы данных и их особенности

Есть **изменяемые** типы — list, dict, set
Есть **неизменяемые** — int, str, tuple, bool, frozenset

Когда добавляем элемент или удаляем из изменяемого, ссылка на объект в памяти сохраняется та же, когда в неизменяемом мы меняем, например, значение числа, то у нас создается новый объект.

## Массивы и кортежи

list() и tuple() | [ ] и ()

Их основное различие в том, что **кортежи изменить нельзя, а массивы — можно**. Кортежи лучше применять, когда не будет необходимости менять список, например, времена года и количество месяцев — постоянное, соответственно можно использовать кортеж)

**Преимущество** использования в том, что достаточно много понятных методов для работы и можно обращаться к элементам массива по индексу
**Недостатки**: неэффективный поиск, удаление и insert в массиве, ибо значения в памяти как бы идут подряд, поэтому когда мы, например, удаляем первый (самый левый) элемент массива, все остальные ячейки должны подвинуться на одну влево

### Основные методы:

- .append() — добавление элемента
- .pop() — удаление элемента (при прохождении цикла лучше не использовать)
- .index() — индекс элемента
- len() — длина массива
- Для подсчета каждого из элементов в массиве можно воспользоваться [Counter](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21)

### Временные сложности:

- Временная сложность .pop() до O(n^2) при удалении нулевого элемента (.pop(0)) — (лучше для этого использовать [deque](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21) с методом .popleft())
- Временная сложность .append() O(1)
- Временная сложность .insert() до O(n^2) при добавлении перед нулевым элементом — (лучше для этого использовать [deque](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21) с методом .appendleft())
- Временная сложность поиска .index() до O(n) если нужный элемент в конце — (лучше для поиска использовать [dict](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21))

## Словари

dict или { }

**Преимущество**: мгновенный поиск по ключу, добавление тоже мгновенное
**Недостатки**: элементы в словаре хранятся неупорядоченно, поэтому нельзя обратиться к элементу по индексу

### Основные методы:

- .keys() — вывести ключи словаря
- .values() — вывести значения словаря
- .items() — вывести пары ключ + значение
- .pop() — удаление из словаря
- .get() — возврат значения по ключу
- .setdefault() — установка значения по умолчанию, если не найден по .get()

### Временные сложности:

- Временная сложность .pop() O(1)
- Временная сложность .get() O(1)
- Временная сложность добавления элемента O(1), ибо элементы не складываются ни в каком порядке, а хранятся в удобном для машины виде

В разделе лайфкахов также есть [полезный метод, который связан со словарем](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21)

## Множества

set()

Можно использовать для вывода уникальных элементов

**Преимущество**: использование множества математических операций таких как нахождение пересечения или разницы множества, поиск уникальных элементов
**Недостатки**: аналогичные массиву недостатки относительно временной сложности

### Основные методы:

- .add() — добавление элемента
- .remove() — удаление элемента
- .intersection() — вывод общих элементов множеств
- .difference() — вывод различий в множествах (или a - b)
- .issubset() — является ли одно множество подмножеством другого

### Временные сложности:

Аналогичны массиву

## Строки

str()

Важно, что строки нужно **пересохранять** при применении элементов

### Основные методы:

- сложение, умножение. При этом строка просто будет повторяться n-ное кол-во раз (s+=, s*=)
- s = s[3:] — для “отсечения” строки
- s=s.join(list) — для “сложения” с массивом
- .count() — подсчет количества определенного элемента
- .find() — поиск элемента. Выведет первый нашедший (**НЕ** выкинет Exception, если не найдет)
- .index() — поиск элемента. Выведет первый нашедший (**выкинет** Exception, если не найдет)
- .isalnum() — фильтрация. Оставляет только числа и буквы / .isalpha() — фильтрация. Оставляет только буквы
- .split() — деление по какому-либо символу / .strip() — удаление лишних пробелов по бокам
- .upper() — перевод в верхний регистр / .lower — перевод в нижний регистр

## Linked List

Тип данных, нереализованный в Python по умолчанию. Устроен следующим образом: представим две ячейки. В одной из них значение, в другой — указатель на следующий элемент в списке. Таким образом, этот список значений, следующий один за другим и называется Linked (соединенный) List (список)

### Временные сложности:

- Сложность поиска до O(n) т.к. для нахождения нужного элемента будет проходить по всем следующим до него
- Сложность вставки **в начало** O(1), ибо мы по сути не проходим по элементам вообще, а только добавляем новый элемент и указатель на начало,
- Сложность вставки в середину O(i), где i — количество элементов перед серединой (или любой другой точкой кроме последней)
- Сложность вставки в конец O(n) — пройдет по всем элементам и в конец вставит новый
- Сложность поиска элемента до O(n)

# Лайфхаки

## Counter

Метод counter из библиотеки collections бывает полезным, когда необходимо посчитать частоты каждого из элементов в массиве

```python
from collections import Counter

nums = [1,3,4,2,3,2,5,1,3,4,2,2,2,1,5,3,2,3,1,3,5,6,5,5,3,2,1,2]
counter = Counter(nums)

print(counter) #Counter({2: 8, 3: 7, 1: 5, 5: 5, 4: 2, 6: 1})
```

## Prod

Может пригодиться в задаче, когда нужно найти произведение всех элементов в массиве

```python
from math import prod

nums = [1,3,4,2,3,2,5,1,3,4,2,2,2,1,5,3,2,3,1,3,5,6,5,5,3,2,1,2]
a = prod(nums)

print(a)
```

## Heapq

Кучи бывают полезны, когда задача формулируется примерно так: 
*“вывести **к-тый** элемент по величине / меньшинству”*

По умолчанию используется **minheap**, то есть значения из кучи будут браться в первую очередь наименьшие, но если нужно **maxheap**, то есть брать наибольшие, то достаточно во всем массиве изменить знаки у элементов.

**Важно**, что при выполнении heapq.heapify() у нас упорядочивается только первый элемент массива. То есть на первое место становится минимальный, а только при .heapq.heappop() у нас уже становится первым следующий минимальный элемент

### Основные методы:

- heapq.heapify() — превратить массив в кучу
- heapq.heappop() — достать минимальный / максимальный элемент (в зависимости от кучи)
- heapq.heappush() — поставить элемент в кучу (с учетом того, является ли он теперь минимальным)
- heapq.nlargest() — вывод n наибольших элементов (временная сложность O(n*log(k)))
- heapq.nsmallest() — вывод n наименьших элементов (временная сложность O(n*log(k)))

### Временные сложности:

- Временная сложность heapq.heapify()  O(n)
- Временная сложность heapq.heappop()  O(log(n))
- Временная сложность heapq.heappush()  O(log(n))

В разделе сортировок есть [сортировка с использованием куч](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21), в которой показано практическое применение

## Deque

Используется в случаях, когда есть необходимость часто работать с началом списка (самый левый элемент)

deque имеет специальные методы, которые позволяют **сокращать время работы с началом списка с O(n) до O(1)**

### Основные методы:

- deque() — превратить массив в очередь
- .popleft() — убрать первый элемент
- .appendleft() — по-другому говоря .insert() на нулевую позицию

### Временные сложности:

- Временная сложность deque() O(n)
- Временная сложность .popleft() O(1)
- Временная сложность .appendleft() O(1)

```python
from collections import deque

nums = [1,3,4,2,3,2,5,1,3,4,2,2,2,1,5,3,2,3,1,3,5,6,5,5,3,2,1,2]
nums = deque(nums)

nums.popleft() #1 при временной сложности O(1)
```

## Itertools (combinations, permutations, product)

Используется, когда необходимо посчитать все перестановки, комбинации элементов. Основные методы: combinations, permutations, product

```python
from itertools import combinations

a = ['A', 'B', 'C']
print(list(combinations(a, 2))) #Комбинируем по два значения

#[('A', 'B'), ('A', 'C'), ('B', 'C')]
```

## lambda + map

map — применяет функцию ко всем элементам массива / словаря и пр. итерируемым объектам

```python
#Синтаксис: **map(функция, массив)**
nums = [2,4,6,8]

def double_val(x):
  return x * 2

print(list(map(double_val, nums))) #[4, 8, 12, 16]
```

lambda — анонимная функция. Используется, когда функцию не будет нужно вызывать повторно или переиспользовать. Нужна только в данном контексте, поэтому безымянная. Синтаксис по логике схож с обычной функцией. Пишем ее без ключевого слова def, вызываем с нужным значением

```python
a = 3
b = lambda x: x*2
b(a) #6
```

Комбинация: lambda+map

```python
nums = [1, 4, -3]

return list(map(lambda x: x ** 2)) #применить функцию лямбда
```

# Сортировки

## Sorted() vs .sort()

Временная сложность O(n) у обоих

sorted(mas) — сортируя, создает новый объект в памяти

mas.sort() — сортирует тот же объект

## Пузырьковая

Временная сложность O(n^2)

```python
def bubble_sort(mas):
  for i in range(len(mas)):
    for j in range(len(mas)-1):
      if mas[j] > mas[j+1]:
        mas[j], mas[j+1] = mas[j+1], mas[j]

  return mas

bubble_sort(nums)
```

## Вставками

Временная сложность O(n^2)

```python
def insertion_sort(mas):
  for i in range(1,len(mas)):
    for j in range(i, 0, -1):
      if mas[j] < mas[j-1]:
        mas[j], mas[j-1] = mas[j-1], mas[j]

  return mas

insertion_sort(nums)
```

## Кучами

Временная сложность O(n*log(n))

```python
import heapq

def heap_sort(mas):
  sorted_mas = []
  heapq.heapify(mas)

  for i in range(len(mas)):
    sorted_mas.append(heapq.heappop(mas))

  return sorted_mas

heap_sort(nums)
```

## Быстрая (Рекурсивный подход)

Временная сложность O(n^2)

```python
def quick_sort(mas):
  if len(mas) <= 1:
    return mas

  p = mas[-1]

  L = [x for x in mas[:-1] if x < p]
  R = [x for x in mas[:-1] if x >= p]

  L = quick_sort(L)
  R = quick_sort(R)

  return L + [p] + R

quick_sort(nums)
```

# DSA

## Two Pointers

Рассматривается в контексте более эффективно прохождения по массиву (O(n^2) → O(n))
Применяется только в сортированном массиве

### Навстречу

Задача: найти пару элементов, которая в сумме дает необходимое значение. Дано значение и сам массив элементов, по которому нужно искать

Преимущество подхода показано на практике:

```python
#Общая временная сложность решения: O(n) * O(n) = O(n^2) -- **плохо**
for i in range(len(nums)): #O(n)
  for j in range(len(nums)): #O(n)
    if i != j:
      if nums[i] + nums[j] == target:
        print(nums[i], nums[j]) #1, 4
        break
```

```python
#Общая временная сложность решения: 2*O(n) -> O(n) -- **хорошо**
nums = sorted(nums) #O(n)

l = 0
r = len(nums) - 1

while l < r: #O(n) - за счет единичного прохода по массиву
  if nums[l] + nums[r] < target:
    l+=1
  elif nums[l] + nums[r] > target:
    r-=1
  else:
    print(nums[l], nums[r]) #1, 4
    break
```

## Быстрый и медленный или non-fixed sliding window

Задача: Найти максимально длинную последовательность символов, где нет повторяющихся

В этом случае медленным указателем можно считать операцию .pop()

```python
strr = 'abccbacacbacdebbbacb'
sett = []
count = 0
max_count = 0

for r in range(len(strr)):
  while strr[r] in sett:
    sett.pop(0)
  sett.append(strr[r])
  max_count = max(max_count, len(sett))

print(max_count) #5

#[]
#['a']
#['a', 'b']
#['a', 'b', 'c']
#['c']
#['c', 'b']
#['c', 'b', 'a']
#['b', 'a', 'c']
#['c', 'a']
#['a', 'c']
#['a', 'c', 'b']
#['c', 'b', 'a']
#['b', 'a', 'c']
#['b', 'a', 'c', 'd']
#['b', 'a', 'c', 'd', 'e'] <- максимальная длина = 5
#['a', 'c', 'd', 'e', 'b'] <- тоже
#['b']
#['b']
#['b', 'a']
#['b', 'a', 'c']
```

Второй вариант
Уже более Two Pointers подход. Суть в том, что у нас есть правый указатель r и левый l. один двигается быстро, другой — медленно, ожидает, пока до нужного момента подвинется правый

**Важно: в контексте задач часто используется правый указатель с циклом for. Подсказка поможет не путаться при решении задач, ибо часто хочется использовать циклы while**

```python
strr = 'abccbacacbacdebbbacb'
sett = set()
count = 0
max_count = 0

l = 0

for r in range(len(strr)):
  print(sett)
  while strr[r] in sett:
    sett.remove(strr[l])
    l+=1
  sett.add(strr[r])
  max_count = max(max_count, (r-l)+1)

print(max_count) #5
```

## Sliding window (Fixed)

Задача: найти максимальное среднее среди последовательности элементов длиной k

```python
nums = [1,4,8,3,9,4,2,5,6,8,2]
k = 4
max_avg = 0
for i in range(len(nums)-k+1):
  print(nums[i: k+i])
  avg = sum(nums[i: k+i]) / k
  max_avg = max(avg, max_avg)
	
print(max_avg) #6.0

#[1, 4, 8, 3]
#[4, 8, 3, 9]
#[8, 3, 9, 4]
#[3, 9, 4, 2]
#[9, 4, 2, 5]
#[4, 2, 5, 6]
#[2, 5, 6, 8]
#[5, 6, 8, 2]
```

## Кумулятивная сумма или prefix sum

```python
nums = [1,2,3,4,5]
nums2 = [0] * len(nums)

for i in range(len(nums)):
  nums2[i] = nums[i] + nums2[i-1]

print(nums2) #[1, 3, 6, 10, 15]
```

## Бинарный поиск

Работает только на **сортированном** массиве

```python
nums = [1,3,7,9,22,99]
target = 3

l = 0
r = len(nums) - 1

while l <= r:
  mid = (l+r) // 2
  if nums[mid] < target:
    l = mid + 1
  elif nums[mid] > target:
    r = mid - 1
  else:
    print(l) #1 - вывел индекс элемента, где находится искомый
    break
```

## Рекурсивный подход

Функция вызывает сама себя некоторое количество раз. При использовании такого подхода важно продумывать **edge-кейсы (момент, когда функция должна остановиться)**, иначе функция будет уходить в бесконечную работу. 

В примере ниже представлена функция по вычислению числе Фибоначчи, где функция вызывается много раз, пока не достигнет edge-case, затем она складывает все вычисленные значения и возвращает их пользователю

Такой подход может быть полезен при обходе деревьев **(DFS)** и при использовании ряда сортировок **(Merge Sort, Quick Sort)**

```python
def F(n):
  if n == 0:
    return 0

  if n == 1:
    return 1

  return F(n-2) + F(n-1)

F(6) #8
```

## Бинарные деревья

В контексте деревьев как правило несколько вариантов обходов:

**DFS** (**Depth** first search) — в глубину — с использованием **стеков** (**FIFO**)

- pre order — сначала родитель, потом правое, левое
- in order — сначала листья левые, потом родитель, потом правые
- post order — сначала левые листья, правые, потом родитель

**BFS** (**Breadth** first search) — в ширину — с использованием **очередей** (**LIFO**)

## [**LinkedList (унарное дерево)**](https://www.notion.so/My-Knowledges-1d1540e9016c80da87a8f45db1e69680?pvs=21)

# ООП

## Принципы ООП

### Абстракция

Код должен зависеть от абстракций, а не конкретных реализаций

В примере ниже абстракция в том, что наш метод sound принимает что-угодно, лишь бы наследовалось от Animal. Если бы на вход метода sound поступала dog, то это уже было бы нарушением принципа абстракции

```python
class Animal:
  def make_sound(self):
    pass

class Dog(Animal):
  def make_sound(self):
    return 'Woof'

def sound(animal: Animal): # абстракция (не важно кто, важно, что метод реализует)
  return animal.make_sound()

dog = Dog()
sound(dog)
```

### Наследование

```python
class Animal:
  def make_sound(self):
    pass

class Dog(Animal): # наследование от Animal
  def make_sound(self):
    return 'Woof'
    
class Cat(Animal): # наследование от Animal
  def make_sound(self):
    return 'Meow'
```

### Полиморфизм

Переопределение методов в дочерних классах

```python
class Animal:
  def make_sound(self):
    pass

class Dog(Animal):
  def make_sound(self): # полиморфизм
    return 'Woof'
    
class Cat(Animal): # полиморфизм
  def make_sound(self):
    return 'Meow'
```

### Инкапсуляция

Сокрытие реализации. Доступ к свойствам объекта должен осуществляться с помощью специальных get и set методов

Вариант 1 — классические методы:

```python
class Cat:
	def __init__(self, name)
		self._name = name
		
	def get_name(self):
		return self._name
		
	def set_name(self, new_name):
		self._name = new_name
		
cat = Cat('sharik')
cat.get_name() #sharik
cat.set_name('murzik')
cat.get_name() #murzik
```

Вариант 2 — декоратор @property:
Позволит обращаться к name **через функции как к свойству**

```python
class Cat:
  def __init__(self, name):
    self._name = name
    
  @property
  def name(self):
    return self._name

  @name.setter
  def name(self, new_name):
    self._name = new_name

cat = Cat('sharik')
cat.name #sharik
cat.name = 'murzik'
cat.name #murzik
```

## Принципы SOLID

### Single Responsibility

Каждый класс отвечает за свой отрезок работы. Нет god-object

```python
#Правильно - разные по логике действия разделены
class ImageSaver:
	def save_image(self):
		pass
		
class ImageZipper:
	def zip_image(self):
		pass
		
class ImageSender:
	def send_image(self):
		pass
		
class ImageUnzipper:
	def unzip_image(self):
		pass
```

```python
#Неправильно - все в одной куче (сохранение, сжатие, отправка)
class Image:
	def save_image(self):
		pass
		
	def zip_image(self):
		pass
		
	def send_image(self):
		pass
		
	def unzip_image(self):
		pass
```

### Open-closed principle

Объекты открыты для расширения, но закрыты для модификации

Нарушением этого принципа может быть наличие if-else логики в функции

```python
#Правильно - для подсчета площади новой фигуры не придется изменять код подсчета старых
class Figure(ABC):
	@abstractmethod
	def count_area(self):
		pass

class Triangle(Figure):
	def count_area(self, a, h):
		return 0.5 * a * h
		
class Square(Figure):
	def count_area(self, a):
		return a**2
		
class Round(Figure):
	def count_area(self, r):
		return 3.14 * r**2
```

```python
#Неправильно - придется менять и добавлять if-else выражения
def count_area():
	if figure == 'round':
		return 3.14 * r**2
	elif figure == 'triangle':
		return 0.5 * a * h
	elif Square == 'square':
		return a**2
```

### Liskov Substitution Principle

Объекты подклассов должны быть **взаимозаменяемыми** с объектами базовых классов

```python
class Bird:
  def fly(self):
    pass

class Pigeon(Bird):
  def fly(self):
    return 'pigeon flying'

class Penguin(Bird):
  def fly(self):
    raise NotImplementedError('Penguins dont fly')

def let_it_fly(bird: Bird): #нарушение LSP поскольку пингвины не летают
    print(bird.fly())
```

### Interface segregation principle

Объекты не должны реализовывать те методы от базовых классов, которых у них нет

В примере ниже все правильно. У домашних животных есть клички, у диких — нет. Орел, хоть и животное, но не бегает, а летает в отличие от собаки

Похожим явлением этого принципа будет являться **Duck Typing**

```python
from abc import ABC, abstractmethod

class HomeAnimal(ABC):
  @abstractmethod
  def __init__(self, name):
    self.name = name

class Flyable(ABC):
  @abstractmethod
  def flies(self):
    pass
    
class Runnable(ABC):
  @abstractmethod
  def run(self):
    pass

class Animal(ABC):
  @abstractmethod
  def eats(self):
    pass

class Fox(Animal):
  def eats(self):
    return 'fox eating'

class Eagle(Animal, Flyable):
  def eats(self):
    return 'eagle eating'

  def flies(self):
    return 'eagle flying'

class Dog(Animal, HomeAnimal, Runnable):
  def __init__(self, name):
    self.name = name

  def eats(self):
    return f'{self.name} eating'
```

### Dependency inversion principle

Модули нижних уровней не должны влиять на модули верхних
Написание кода от абстракций, а не конкретных реализаций

В примере ниже видно, нам не важно, что здесь за птица. Главное, что есть метод fly.

```python
class Bird:
  def fly(self):
    pass

class Pigeon(Bird):
  def fly(self):
    return 'pigeon flying'

def let_it_fly(bird: Bird):
    print(bird.fly())
```

## **Duck Typing**

При использовании этого принципа не важно, является ли объект наследником нужного класса. Достаточно, чтобы он реализовывал определенный метод

```python
from abc import ABC, abstractmethod

class Runnable(ABC):
    @abstractmethod
    def run(self):
      pass

class Cat(Runnable):
    def run(self):
        return "Cat runs"

def move(animal: Runnable):
    print(animal.run())
```

```python
from typing import Protocol

class Runnable(Protocol):
    def run(self):
      pass

class Cat:
    def run(self):
        return "Cat runs"

def move(animal: Runnable):
    print(animal.run())
```

# SQL

## Синтаксис select-запросов:

```sql
select * from Table;
```

где:

- select — означает, что выбираем данные к выводу
- * — в данном случае все поля всех колонок таблицы
- from Table — из какой таблицы

## Group by

Оператор используется для группировки значений по какому-либо признаку

```sql
select name, weight_category from sportsmen group by weight_category
```

Теперь данные будут выведены не в том порядке, в котором они хранятся в таблице, а станут группированы по весовой категории (при выводе)

## Where, Having

Оба оператора отвечают за фильтрацию данных, но отличие в том, что where используется до группировки данных (при помощи оператора group by), а having — после

```sql
select name, weight, hospital, building from patients 
where hospital='бурденко' 
group by building 
having weight>80
```

Запрос буквально формулируется так:

“Выведи имя, вес, больницу и здание больницы среди пациентов, где больница — госпиталь им. бурденко, сгруппируй по зданиям. Отфильтруй среди пациентов этой больницы тех, у кого вес больше 80 и выведи их.” 

## Order by

Используется для сортировки данных по возрастанию (ASC) или убыванию (DESC). При использовании допускается указывать не конкретное название столбца, а номер в селект-запросе

```sql
select name, wins, losses from mma_fighters order by 2 desc
```

Здесь происходит сортировка по убыванию количества побед (2 столбец — wins)

## Limit, offset

Используется отступа (offset) и ограничении количества строк при выводе (limit) 

```sql
select name, wins, losses from mma_fighters order by 2 desc limit 5 offset 5
```

Пропустит топ-5 бойцов, но выведет с 5 по 10 номер по количеству побед. Важно, что порядок написания именно как в запросе, иначе — синтаксическая ошибка

## Join и Union

Используются, когда нужно соединить по горизонтали с сопоставлением по какому-либо признаку/ключу (join) или для объединения по вертикали (union / union all)

left join — левая таблица остается полной, из правой берутся только значения, которые имеют пару из левой

right join — правая таблица остается полной, из левой берутся только значения, которые имеют пару из правой

inner join — берутся только те значения, которые имеют пару

full join — остаются все значения. И те, что имеют пару, и те, что не имеют

cross join — сопоставление каждого с каждым

Union — убирает дубликаты при соединении, union all — оставляет все данные

```sql
select p.patient_id, a.name 
from patients p join admissions a on p.patiend_id = a.patient_id
```

## Alias

Допишу как вернусь