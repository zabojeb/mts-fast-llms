def func1():
    return "Функция 1"

def func2():
    return "Функция 2"

def func3():
    return "Функция 3"

def call_functions(function_list):
    results = []
    for func in function_list:
        results.append(func())  # Вызываем каждую функцию из списка
    return results


functions = [func1, func2, func3]


output = call_functions(functions)
print(output)